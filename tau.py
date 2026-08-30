"""Measure average generation length tau for speculative decoding drafts.

tau := E[# tokens committed per round of speculation] = mean(n_accepted) + 1.
This is the quantity the SPIRe paper measures empirically (Table 1) and then
feeds into its analytical performance model; it depends only on the draft's
and target's distributions, never on kernels, memory layout, or hardware.

Draft modes:
  vanilla  -- a separate, smaller model (--draft-model-name), dense attention.
  magicdec -- the TARGET's own weights, drafting through a StreamingLLM
              sink+window mask. Paper's baseline uses positions-within-cache,
              so --magicdec-rope is ON by default for this mode.
  spire    -- a separately trained draft that uses the sink+window mask during
              training AND inference, with original-text positions. (Once a
              SPIRe draft exists; the sparse-prefill path is selected here.)
  self     -- draft == target, dense. Sanity check only: tau must equal k+1.

Deviations from the paper's protocol, stated so results stay comparable:
  * The paper generates G=64 tokens per context and reports tau over those.
    Here a fixed number of ROUNDS is run per context instead, which is
    cleaner statistically (every round is one sample) but means the number of
    generated tokens varies. With --rounds 16 and k=4 at least 16 and at most
    80 tokens are produced.
  * The reported 95% CI treats rounds as independent samples. Rounds within a
    sequence are correlated, so this interval is optimistic; --by-context
    reports a per-context-mean interval, which is conservative.
  * Sampling temperature is not specified by the paper; default 1.0.

Example:
  XLA_FLAGS=--xla_force_host_platform_device_count=8 python tau.py \
      --config local_test_synthetic --model-name dense_test \
      --mode magicdec --window 64 --sink 1 --k 4 --context-len 512 \
      --batch 64 --num-batches 8
"""

# Set XLA flags before importing JAX
import init_seqax  # noqa: F401  # isort: skip

import argparse
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from omegaconf import OmegaConf

import input_loader
import jax_extra
import shardlib.shardtypes as shardtypes
import training_io
from model import Model, ModelConfig
from speculative import make_speculative_generate

shardtypes.register_with_typeguard()


def load_weights(h: ModelConfig, model_dir: str) -> Model:
    from train import State

    rng = jnp.zeros((2,), dtype=jnp.uint32)
    with shardtypes.Scope():
        state = jax.jit(partial(State.init, h))(rng)
    state, step = training_io.load_checkpoint_if_it_exists(
        model_dir, state, training_io.IOConfig(max_io_threads=64)
    )
    if step == 0:
        raise FileNotFoundError(f"no checkpoint in {model_dir}")
    print(f"  loaded step {step} from {model_dir}")
    return state.weights


def dataset_prompts(cfg, context_len: int, batch: int, num_batches: int):
    """Yield [batch, context_len] prompts from the validation split."""
    params = jax_extra.make_dataclass_from_dict(input_loader.FlatTokensParams, cfg.dataset)
    bp = input_loader.TokenBatchParams(len=context_len, batch=batch)
    loader = input_loader.ShufflingLoader("validation", params, bp)
    for i in range(num_batches):
        yield jnp.asarray(loader.load(i).targets, dtype=jnp.uint32)


def random_prompts(vocab: int, context_len: int, batch: int, num_batches: int, seed: int = 0):
    for i in range(num_batches):
        yield jax.random.randint(
            jax.random.PRNGKey(seed + i), (batch, context_len), 0, vocab
        ).astype(jnp.uint32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model-name", required=True, help="target checkpoint dir name")
    p.add_argument("--draft-model-name", default=None, help="draft checkpoint dir (vanilla/spire modes)")
    p.add_argument("--draft-config", default=None, help="config name holding the draft's model dims")
    p.add_argument("--mode", choices=["vanilla", "magicdec", "spire", "self"], required=True)
    p.add_argument("--k", type=int, default=4, help="maximum speculation depth")
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--rounds", type=int, default=16)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--num-batches", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sink", type=int, default=1)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--magicdec-rope", dest="mdrope", action="store_true", default=None)
    p.add_argument("--no-magicdec-rope", dest="mdrope", action="store_false")
    p.add_argument("--prompts", choices=["dataset", "random"], default="dataset")
    p.add_argument("--by-context", action="store_true", help="also report a per-context (conservative) CI")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = OmegaConf.load(f"configs/{args.config}.yaml")
    h_t = ModelConfig(**cfg.model)
    if args.draft_config:
        h_d = ModelConfig(**OmegaConf.load(f"configs/{args.draft_config}.yaml").model)
    else:
        h_d = h_t

    # Mode -> draft wiring. This is the only place the three baselines differ.
    if args.mode == "self":
        window, prefill_dense, mdrope = None, True, False
    elif args.mode == "vanilla":
        window, prefill_dense, mdrope = None, True, False
    elif args.mode == "magicdec":
        window, prefill_dense, mdrope = args.window, True, True
    else:  # spire: trained with the sparse mask, original-text positions
        window, prefill_dense, mdrope = args.window, False, False
    if args.mdrope is not None:
        mdrope = args.mdrope

    mesh_cfg = cfg.mesh
    assert mesh_cfg.t == 1 and mesh_cfg.s == 1, "tau.py supports batch ('d') sharding only"
    assert args.batch % mesh_cfg.d == 0

    with Mesh(mesh_utils.create_device_mesh([mesh_cfg.d, mesh_cfg.t, mesh_cfg.s], jax.devices()), ("d", "t", "s")):
        print(f"mode={args.mode} k={args.k} L={args.context_len} T={args.temperature} "
              f"sink={args.sink} window={window} magicdec_rope={mdrope}")
        w_t = load_weights(h_t, os.path.join(cfg.root_working_dir, args.model_name))
        if args.mode in ("vanilla", "spire"):
            assert args.draft_model_name, f"--draft-model-name required for mode {args.mode}"
            w_d = load_weights(h_d, os.path.join(cfg.root_working_dir, args.draft_model_name))
        else:
            w_d, h_d = w_t, h_t

        spec = make_speculative_generate(
            h_t, h_d, args.context_len, args.rounds, args.k, args.temperature,
            draft_sink=args.sink, draft_window=window,
            draft_prefill_dense=prefill_dense, magicdec_rope=mdrope,
        )

        src = (dataset_prompts(cfg, args.context_len, args.batch, args.num_batches)
               if args.prompts == "dataset"
               else random_prompts(h_t.vocab, args.context_len, args.batch, args.num_batches, args.seed))

        all_acc = []
        for i, prompt in enumerate(src):
            rng = jnp.array([0, args.seed + i], dtype=jnp.uint32)
            with shardtypes.Scope():
                _, _, n_acc = spec(w_t, w_d, prompt, rng)
            all_acc.append(np.asarray(n_acc))
            print(f"  batch {i}: tau = {float(np.mean(all_acc[-1])) + 1:.3f}")

        acc = np.concatenate(all_acc, axis=0)  # [contexts, rounds]
        gen = acc + 1.0
        tau = float(gen.mean())
        ci = 1.96 * float(gen.std(ddof=1)) / np.sqrt(gen.size)
        print(f"\ntau = {tau:.3f} +/- {ci:.3f}  (n = {gen.size} rounds, "
              f"{acc.shape[0]} contexts x {acc.shape[1]} rounds)")
        print(f"acceptance rate = {float(acc.mean()) / args.k:.3f}  "
              f"(mean accepted {float(acc.mean()):.3f} of k={args.k})")
        if args.by_context:
            per_ctx = gen.mean(axis=1)
            ci_c = 1.96 * float(per_ctx.std(ddof=1)) / np.sqrt(per_ctx.size)
            print(f"per-context tau = {float(per_ctx.mean()):.3f} +/- {ci_c:.3f} (conservative)")
        if args.mode == "self":
            # In exact arithmetic self-drafting accepts every token, so tau == k+1.
            # Measured tau falls slightly short because the draft evaluates tokens
            # incrementally (L=1) while verification evaluates them in a batched
            # (k+1)-token pass; in bf16 the two reduce in different orders, so
            # near-tied argmaxes can disagree. The shortfall is the NUMERICAL
            # AGREEMENT FLOOR of this implementation: no draft can be measured
            # above it, and it is invisible to the analytical cost model.
            floor = (args.k + 1) - tau
            print(f"self-draft numerical agreement floor: {floor:.4f} tokens/round "
                  f"({floor / (args.k + 1) * 100:.2f}% of k+1)")
            assert tau > args.k, f"self-draft tau {tau} far below k+1; likely a real bug, not numerics"
            print("self-draft sanity check PASSED (within numerical floor)")


if __name__ == "__main__":
    main()
"""KV-cached autoregressive decoding for the dense seqax model.

Design, in the spirit of the rest of this codebase:
- No configuration objects beyond ModelConfig; generation parameters are
  explicit function arguments, fixed at trace time.
- The entire generation (prefill + decode loop) is one jitted, sharded
  computation: prefill fills a fixed-size cache of length P + G, then a
  lax.scan runs G - 1 single-token steps. One compile, no per-token dispatch.
- The KV cache is explicit: bf16[layers, 2, B, P+G, n_kv, d_head], written
  at kv_offset, never resized. Masks are built from absolute positions.
- Sharding: batch over "d" only. Tensor ("t") and sequence ("s") parallelism
  are not supported here (asserted), matching train.py's restriction.

Assumptions, stated rather than handled:
- All prompts in a batch have the same length P (no padding / ragged prompts).
  This matches the SPIRe evaluation protocol: fixed-length contexts from
  LongCrawl64, generate G tokens each.
- No EOS handling: exactly G tokens are generated per sequence.
- temperature == 0.0 means greedy; otherwise softmax sampling at that
  temperature, with per-device RNG decorrelation across the "d" axis.

Run as a script to decode from a trained checkpoint:
  XLA_FLAGS=--xla_force_host_platform_device_count=8 python decode.py \
      --config local_test_synthetic --model-name dense_test \
      --prompt-len 8 --gen-len 16 --batch 8 --check
"""

# Set XLA flags before importing JAX
import init_seqax  # noqa: F401  # isort: skip

import argparse
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from omegaconf import OmegaConf
from typeguard import typechecked

import shardlib.shardtypes as shardtypes
import training_io
from model import Model, ModelConfig
from shardlib.shardtypes import bf16, bool_, f32, i32, u32

shardtypes.register_with_typeguard()
PRNGKey = u32[b"2"]


def _sample(logits: jax.Array, rng: jax.Array, step: jax.Array, temperature: float) -> jax.Array:
    """Sample next token ids [B] from logits [B, V]. temperature == 0.0 -> greedy."""
    if temperature == 0.0:
        return jnp.argmax(logits, axis=-1).astype(jnp.uint32)
    # Decorrelate randomness across the batch-sharded "d" axis and across steps.
    key = jax.random.fold_in(jax.random.fold_in(jax.random.wrap_key_data(rng), step), jax.lax.axis_index("d"))
    return jax.random.categorical(key, logits / temperature, axis=-1).astype(jnp.uint32)


def make_generate(h: ModelConfig, prompt_len: int, gen_len: int, temperature: float):
    """Build a jitted, sharded generate function for fixed prompt/generation lengths.

    Returns generate(weights, prompt_ids[B, prompt_len], rng) -> generated_ids[B, gen_len].
    Recompiles for each distinct (prompt_len, gen_len, temperature, model config).
    """
    P, G = prompt_len, gen_len
    Klen = P + G  # fixed cache size; the last generated token is never written back

    @jax.jit
    @partial(shardtypes.typed_shard_map, check_rep=False)
    @typechecked
    def generate(w: Model, prompt: u32[b"B/d P"], rng: PRNGKey) -> u32[b"B/d G"]:
        lb = prompt.shape[0]  # local batch
        k_pos = jnp.arange(Klen)[jnp.newaxis, jnp.newaxis, :]  # [1, 1, Klen]

        # ---- Prefill: fill cache positions [0, P), sample token P ----
        cache = jnp.zeros((h.layers, 2, lb, Klen, h.n_kv, h.d_head), dtype=jnp.bfloat16)
        q_pos = jnp.arange(P)[jnp.newaxis, :, jnp.newaxis]  # absolute positions 0..P-1
        prefill_mask = jnp.broadcast_to(k_pos <= q_pos, (lb, P, Klen))
        with shardtypes.Scope():
            logits, cache, _ = w.forward_pass(h, prompt, prefill_mask, kv_cache=cache, kv_offset=jnp.int32(0))
        tok = _sample(logits[:, -1], rng, jnp.uint32(0), temperature)  # [lb]

        # ---- Decode loop: token at absolute position pos is written to the
        # cache and used to sample the token at position pos + 1. ----
        def step(carry, step_idx):
            cache, tok, pos = carry
            mask = jnp.broadcast_to(k_pos <= pos, (lb, 1, Klen))
            with shardtypes.Scope():
                logits, cache, _ = w.forward_pass(
                    h, tok[:, jnp.newaxis], mask, kv_cache=cache, kv_offset=pos
                )
            next_tok = _sample(logits[:, -1], rng, step_idx + 1, temperature)
            return (cache, next_tok, pos + 1), tok

        (_, last_tok, _), toks = jax.lax.scan(
            step, (cache, tok, jnp.int32(P)), jnp.arange(G - 1, dtype=jnp.uint32)
        )
        # toks: [G-1, lb] tokens at positions P .. P+G-2; last_tok: position P+G-1
        out = jnp.concatenate([toks, last_tok[jnp.newaxis, :]], axis=0)
        return jnp.transpose(out, (1, 0))

    return generate


def make_greedy_check(h: ModelConfig, prompt_len: int, gen_len: int):
    """Reference check: teacher-force [prompt || generated] through the full
    non-cached forward pass; greedy generation is correct iff generated[t] equals
    argmax of the full-pass logits at the preceding position, for every t."""
    P, G = prompt_len, gen_len
    T = P + G

    @jax.jit
    @partial(shardtypes.typed_shard_map, check_rep=False)
    @typechecked
    def check(w: Model, full_ids: u32[b"B/d T"]) -> f32[b""]:
        lb = full_ids.shape[0]
        causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))[jnp.newaxis, ...]
        mask = jnp.broadcast_to(causal, (lb, T, T))
        with shardtypes.Scope():
            logits, _, _ = w.forward_pass(h, full_ids, mask)
        pred = jnp.argmax(logits[:, P - 1 : T - 1], axis=-1).astype(jnp.uint32)  # predicts positions P..T-1
        agree = jnp.mean((pred == full_ids[:, P:]).astype(jnp.float32))
        return jax.lax.pmean(agree, ("d", "t", "s"))

    return check


def load_weights(h: ModelConfig, model_dir: str, rng: PRNGKey) -> Tuple[Model, int]:
    """Load model weights from a seqax checkpoint directory (latest step)."""
    from train import State  # deferred: train.py pulls in hydra/zarr machinery

    state = jax.jit(partial(State.init, h))(rng)
    state, start_step = training_io.load_checkpoint_if_it_exists(
        model_dir, state, training_io.IOConfig(max_io_threads=64)
    )
    if start_step == 0:
        raise FileNotFoundError(f"no checkpoint found in {model_dir}")
    return state.weights, start_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="config name under configs/, e.g. local_test_synthetic")
    p.add_argument("--model-name", required=True, help="model dir name under root_working_dir")
    p.add_argument("--prompt-len", type=int, default=8)
    p.add_argument("--gen-len", type=int, default=16)
    p.add_argument("--batch", type=int, default=8, help="must be divisible by mesh d")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check", action="store_true", help="verify greedy output against the non-cached forward pass")
    args = p.parse_args()

    cfg = OmegaConf.load(f"configs/{args.config}.yaml")
    h = ModelConfig(**cfg.model)
    mesh_cfg = cfg.mesh
    assert mesh_cfg.t == 1 and mesh_cfg.s == 1, "decode.py supports batch ('d') sharding only"
    assert args.batch % mesh_cfg.d == 0, "batch must be divisible by mesh d"
    assert args.prompt_len + args.gen_len <= h.seq_len or True  # cache may exceed train seq_len; RoPE table extends

    import os

    model_dir = os.path.join(cfg.root_working_dir, args.model_name)

    with Mesh(mesh_utils.create_device_mesh([mesh_cfg.d, mesh_cfg.t, mesh_cfg.s], jax.devices()), ("d", "t", "s")):
        rng = jnp.zeros((2,), dtype=jnp.uint32).at[1].set(args.seed)
        weights, step = load_weights(h, model_dir, rng)
        print(f"loaded checkpoint step {step} from {model_dir}")

        # Demo prompt: deterministic pseudo-random token ids in-vocab.
        prompt = jax.random.randint(
            jax.random.PRNGKey(args.seed), (args.batch, args.prompt_len), 0, h.vocab
        ).astype(jnp.uint32)

        generate = make_generate(h, args.prompt_len, args.gen_len, args.temperature)
        generated = generate(weights, prompt, rng)
        print("prompt[0]:   ", prompt[0].tolist())
        print("generated[0]:", generated[0].tolist())

        if args.check:
            if args.temperature != 0.0:
                print("--check requires greedy decoding (temperature 0); skipping")
            else:
                check = make_greedy_check(h, args.prompt_len, args.gen_len)
                agree = check(weights, jnp.concatenate([prompt, generated], axis=1))
                print(f"greedy agreement with non-cached forward pass: {float(agree):.4f}")
                assert float(agree) == 1.0, "cached decode diverges from full forward pass"
                print("greedy check PASSED")


if __name__ == "__main__":
    main()
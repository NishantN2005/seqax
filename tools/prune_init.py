"""Build the SPIRe draft's initial checkpoint by pruning the trained target.

The paper initializes the draft from the target rather than randomly: draft layer
i takes target layer i+6, and the embedding, unembedding, and both outer
layernorms are copied wholesale. Figure 5 puts the value of this at 0.296 tau
(3.401 full vs 3.105 random-init) -- the second largest single contributor, after
distillation.

This is a pure tree slice because the SPIRe draft matches the target in every
dimension except depth (d_model 512, n_kv 8, d_head 128, d_ff 4096, 2 layers vs
8). Every per-layer weight in `Model` carries a leading `layers` axis, so the
prune is `w[6:8]`. Nothing is reshaped or projected.

The optimizer state is deliberately NOT carried over: Adam moments from the
target's final cosine-decayed steps describe a different loss surface than the
draft's fresh warmup, and `State.init` zeros them anyway. We write weights only
and let training start its own moments.

Example:
    PYTHONPATH=. python tools/prune_init.py \
        --target-config spire_target_1024 --target-name spire_target_1024 \
        --draft-config spire_draft_spire  --draft-name  spire_draft_spire_init
"""

import argparse
import os

import init_seqax  # noqa: F401

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from omegaconf import OmegaConf

import shardlib.shardtypes as shardtypes

shardtypes.register_with_typeguard()

import training_io  # noqa: E402
from model import Model, ModelConfig  # noqa: E402

# Per-layer fields live on TransformerLayer and carry a leading `layers` axis.
# Everything else on Model is depth-independent and copied unchanged.
SHARED_FIELDS = ("embed", "unembed", "ln_embed", "ln_final")


def prune(target: Model, h_t: ModelConfig, h_d: ModelConfig, first_layer: int) -> Model:
    """Slice `h_d.layers` consecutive target layers starting at `first_layer`."""
    for dim in ("d_model", "n_q_per_kv", "n_kv", "d_head", "d_ff", "vocab"):
        tv, dv = getattr(h_t, dim), getattr(h_d, dim)
        if tv != dv:
            raise ValueError(
                f"prune_init requires matching {dim}: target {tv} != draft {dv}. "
                "The SPIRe draft differs from the target only in layer count."
            )
    last = first_layer + h_d.layers
    if last > h_t.layers:
        raise ValueError(f"target has {h_t.layers} layers; cannot take [{first_layer}:{last})")

    # Every per-layer weight on Transformer carries a leading `layers` axis.
    transformer = jax.tree.map(lambda w: w[first_layer:last], target.transformer)
    kw = {f: getattr(target, f) for f in SHARED_FIELDS}
    # Feedback-memory weights are NOT inherited: the target's are shape
    # [target_layers, 0] and the draft needs [draft_layers, n_mem]. Zeros make the
    # pathway an exact no-op at step 0, so a memory-enabled draft starts exactly
    # where the memory-free one did and can only diverge by learning.
    w_memory = jnp.zeros((h_d.layers, h_d.n_mem), dtype=jnp.float32)
    draft = Model(transformer=transformer, w_memory=w_memory, **kw)

    print(f"pruned target layers [{first_layer}:{last}) -> draft layers [0:{h_d.layers})")
    for f in SHARED_FIELDS:
        print(f"  copied {f}: {getattr(target, f).shape}")
    print(f"  sliced w_q: {target.transformer.w_q.shape} -> {draft.transformer.w_q.shape}")
    print(f"  w_memory: {draft.w_memory.shape} (zero-init, exact no-op)")
    return draft


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-config", required=True, help="config name under configs/")
    p.add_argument("--target-name", required=True, help="target run dir under root_working_dir")
    p.add_argument("--draft-config", required=True)
    p.add_argument("--draft-name", required=True, help="run dir to write the pruned checkpoint into")
    p.add_argument(
        "--first-layer",
        type=int,
        default=None,
        help="first target layer to take; default = target.layers - draft.layers (the LAST n layers, "
        "which is what the paper's 'layers 7-8' means for an 8-layer target and 2-layer draft)",
    )
    args = p.parse_args()

    cfg_t = OmegaConf.load(f"configs/{args.target_config}.yaml")
    cfg_d = OmegaConf.load(f"configs/{args.draft_config}.yaml")
    h_t = ModelConfig(**cfg_t.model)
    h_d = ModelConfig(**cfg_d.model)
    first = args.first_layer if args.first_layer is not None else h_t.layers - h_d.layers

    root = cfg_t.root_working_dir
    mc = cfg_t.mesh
    io = training_io.IOConfig(max_io_threads=64)

    with Mesh(mesh_utils.create_device_mesh([mc.d, mc.t, mc.s], jax.devices()), ("d", "t", "s")):
        from train import State  # deferred: pulls in hydra/zarr machinery

        rng = jnp.zeros((2,), dtype=jnp.uint32)

        # shardlib's dimension names are GLOBAL and persist across calls, so
        # initializing an 8-layer target and then a 2-layer draft in one process
        # collides ("expected 8, got 2") unless each gets a fresh Scope.
        with shardtypes.Scope():
            state_t = jax.jit(lambda r: State.init(h_t, r))(rng)
            state_t, step_t = training_io.load_checkpoint_if_it_exists(
                os.path.join(root, args.target_name), state_t, io
            )
        if step_t == 0:
            raise FileNotFoundError(f"no target checkpoint in {os.path.join(root, args.target_name)}")
        print(f"loaded target step {step_t}")

        draft_weights = prune(state_t.weights, h_t, h_d, first)

        # Fresh optimizer moments; only the weights are inherited.
        with shardtypes.Scope():
            state_d = jax.jit(lambda r: State.init(h_d, r))(rng)
        state_d = State(weights=draft_weights, adam_mu=state_d.adam_mu, adam_nu=state_d.adam_nu)

        out = os.path.join(root, args.draft_name)
        training_io.save_checkpoint(out, 0, state_d, io)
        print(f"\nwrote pruned-init checkpoint to {out}/0000000000")
        print("Train from it with: clone_from=<that dir name>  reset_optimizer=true")


if __name__ == "__main__":
    main()

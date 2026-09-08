"""The SPIRe memory pathway as train.py actually drives it.

tests/test_spire_memory.py checks the operator inside forward_pass. This checks
the training path on top of it: that loss_fn slices the right target activations,
builds the block split, reaches the loss, and produces gradients on the key and
value projections -- the parameters the pathway exists to feed.

The teeth test is the important one. A memory pathway that is silently
disconnected still trains, still reports a falling loss, and still produces a
checkpoint; that is precisely how a 20,841-step run was spent measuring an
operator the paper does not describe.

    PYTHONPATH=. python tests/test_spire_train.py
"""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
import init_seqax  # noqa: E402, F401

os.environ["JAX_PLATFORMS"] = "cpu"

from functools import partial  # noqa: E402
from typing import Tuple  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import mesh_utils  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

import shardlib.shardtypes as shardtypes  # noqa: E402
from shardlib.shardtypes import f32  # noqa: E402

shardtypes.register_with_typeguard()

from input_loader import TokenBatch  # noqa: E402
from model import Model, ModelConfig  # noqa: E402
from train import loss_fn  # noqa: E402

VOCAB, L, B = 128, 16, 4
COMMON = dict(vocab=VOCAB, seq_len=L, d_model=32, n_q_per_kv=1, n_kv=4, d_head=8,
              d_ff=64, rope_max_timescale=100, attention_mask="streaming_llm",
              sink_size=1, window=4)
h_teacher = ModelConfig(layers=4, **COMMON)
h_plain = ModelConfig(layers=2, **COMMON)
h_mem = ModelConfig(layers=2, n_mem=2, memory_mode="kv", memory_block_k=4, **COMMON)


def make_batch(rng):
    ids = jax.random.randint(rng, (B, L), 1, VOCAB).astype(jnp.uint32)
    starts = np.zeros((B, L), dtype=bool)
    starts[:, 0] = True
    return TokenBatch(targets=ids, is_seq_start=jnp.asarray(starts))


def run(w, h, batch, teacher, grad=False):
    def go_inner(w, batch):
        loss, _ = loss_fn(w, h, batch, None, teacher, h_teacher, 0.5)
        return jax.lax.psum(loss, ("d", "t", "s"))

    @jax.jit
    @partial(shardtypes.typed_shard_map, check_rep=False)
    def go(w: Model, batch: TokenBatch) -> f32[b""]:
        return go_inner(w, batch)

    @jax.jit
    @partial(shardtypes.typed_shard_map, check_rep=False)
    def go_grad(w: Model, batch: TokenBatch) -> Model:
        return jax.grad(go_inner)(w, batch)

    with shardtypes.Scope():
        return float(go(w, batch)) if not grad else go_grad(w, batch)


with Mesh(mesh_utils.create_device_mesh([1, 1, 1], jax.devices()[:1]), ("d", "t", "s")):
    rng = jax.random.PRNGKey(0)
    key = jax.random.key_data(rng).astype(jnp.uint32)
    with shardtypes.Scope():
        w_t = jax.jit(Model.init, static_argnums=0)(h_teacher, key)
    with shardtypes.Scope():
        w_plain = jax.jit(Model.init, static_argnums=0)(h_plain, key)
    with shardtypes.Scope():
        w_mem = jax.jit(Model.init, static_argnums=0)(h_mem, key)
    batch = make_batch(jax.random.PRNGKey(1))

    l_plain = run(w_plain, h_plain, batch, w_t)
    l_mem = run(w_mem, h_mem, batch, w_t)
    print(f"MixedLoss without memory = {l_plain:.6f}")
    print(f"MixedLoss with memory    = {l_mem:.6f}")
    assert np.isfinite(l_mem), f"memory pathway produced a non-finite loss: {l_mem}"
    print("1. loss is finite                                                   PASSED")

    # Teeth: identical weights, identical batch, identical teacher. The ONLY
    # difference is that the draft reads the target's activations as its keys and
    # values. If that does not move the loss, the pathway is not connected.
    assert abs(l_mem - l_plain) > 1e-4, (
        f"memory changed the loss by {abs(l_mem - l_plain):.2e}: the target "
        "activations are not reaching the loss, so training would silently "
        "reproduce the no-memory ablation")
    print(f"2. memory moves the loss (delta = {abs(l_mem - l_plain):.4f})                       PASSED")

    # The pathway feeds the key and value projections specifically. Zero gradient
    # there would mean the substitution is being computed and then discarded.
    g = run(w_mem, h_mem, batch, w_t, grad=True)
    gk = float(jnp.abs(g.transformer.w_k).max())
    gv = float(jnp.abs(g.transformer.w_v).max())
    assert gk > 0 and gv > 0, f"no gradient on w_k ({gk}) / w_v ({gv})"
    print(f"3. gradients reach w_k ({gk:.3e}) and w_v ({gv:.3e})          PASSED")

    # memory_block_k is what makes the split block-structured rather than a
    # per-sequence prefix; forgetting it must fail loudly, not train silently.
    try:
        bad = ModelConfig(layers=2, n_mem=2, memory_mode="kv", memory_block_k=0, **COMMON)
        run(w_mem, bad, batch, w_t)
        raise SystemExit("FAIL: memory_mode='kv' with memory_block_k=0 was accepted")
    except AssertionError as e:
        assert "memory_block_k" in str(e), e
    print("4. memory_mode='kv' without memory_block_k is rejected              PASSED")

    print("\nall SPIRe training-path tests passed")

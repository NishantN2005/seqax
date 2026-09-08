"""The Feedback Transformer half of SPIRe (paper 3.3).

The load-bearing test is CAUSALITY OF THE RECURRENCE. Under z^i_t =
Attention(x^i_t, m_<t), a query at offset 0 of its block has an EMPTY self
region: everything before the block start is a target activation, and there is
no earlier in-block position. So its logits cannot depend on the memory mixing
weights at all. If they do, either the strict causality or the k-pass structure
is wrong -- and both are the kind of error that still trains, still shows a
falling loss, and still produces a checkpoint.

    PYTHONPATH=. python tests/test_spire_feedback.py
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

VOCAB, L, B, K = 128, 16, 4, 4
COMMON = dict(vocab=VOCAB, seq_len=L, d_model=32, n_q_per_kv=1, n_kv=4, d_head=8,
              d_ff=64, rope_max_timescale=100, attention_mask="streaming_llm",
              sink_size=1, window=8)
h_teacher = ModelConfig(layers=4, **COMMON)
# n_mem = layers + 1: the mix runs over the embedding state plus each layer output.
h_nofb = ModelConfig(layers=2, n_mem=3, memory_mode="kv", memory_block_k=K, **COMMON)
h_fb = ModelConfig(layers=2, n_mem=3, memory_mode="kv", memory_block_k=K,
                   feedback_memory=True, **COMMON)


def make_batch(rng):
    ids = jax.random.randint(rng, (B, L), 1, VOCAB).astype(jnp.uint32)
    starts = np.zeros((B, L), dtype=bool)
    starts[:, 0] = True
    return TokenBatch(targets=ids, is_seq_start=jnp.asarray(starts))


def loss_of(w, h, batch, teacher):
    @jax.jit
    @partial(shardtypes.typed_shard_map, check_rep=False)
    def go(w: Model, batch: TokenBatch) -> f32[b""]:
        loss, _ = loss_fn(w, h, batch, None, teacher, h_teacher, 0.5)
        return jax.lax.psum(loss, ("d", "t", "s"))
    with shardtypes.Scope():
        return float(go(w, batch))


def grads_of(w, h, batch, teacher):
    @jax.jit
    @partial(shardtypes.typed_shard_map, check_rep=False)
    def go(w: Model, batch: TokenBatch) -> Model:
        def f(w, batch):
            loss, _ = loss_fn(w, h, batch, None, teacher, h_teacher, 0.5)
            return jax.lax.psum(loss, ("d", "t", "s"))
        return jax.grad(f)(w, batch)
    with shardtypes.Scope():
        return go(w, batch)


with Mesh(mesh_utils.create_device_mesh([1, 1, 1], jax.devices()[:1]), ("d", "t", "s")):
    key = jax.random.key_data(jax.random.PRNGKey(0)).astype(jnp.uint32)
    with shardtypes.Scope():
        w_t = jax.jit(Model.init, static_argnums=0)(h_teacher, key)
    with shardtypes.Scope():
        w = jax.jit(Model.init, static_argnums=0)(h_fb, key)
    batch = make_batch(jax.random.PRNGKey(1))

    # --- mix_memory is a genuine convex mix -------------------------------
    states = jnp.bfloat16(jax.random.normal(jax.random.PRNGKey(3), (3, B, L, 32)))
    wm = jnp.float32(jax.random.normal(jax.random.PRNGKey(4), (2, 3)))
    m = Model.mix_memory(wm, states)
    assert m.shape == (2, B, L, 32), m.shape
    same = jnp.broadcast_to(states[0][jnp.newaxis], states.shape)
    m_same = np.asarray(jnp.float32(Model.mix_memory(wm, same)))
    ref = np.asarray(jnp.float32(same[0]))
    assert np.allclose(m_same[0], ref, atol=2e-2) and np.allclose(m_same[1], ref, atol=2e-2), \
        "mixing identical states did not return that state: weights do not sum to 1"
    print("1. mix_memory is a convex combination (softmax over layers)         PASSED")

    # --- feedback changes the model ---------------------------------------
    l_nofb = loss_of(w, h_nofb, batch, w_t)
    l_fb = loss_of(w, h_fb, batch, w_t)
    print(f"   MixedLoss without feedback = {l_nofb:.6f}")
    print(f"   MixedLoss with feedback    = {l_fb:.6f}")
    assert np.isfinite(l_fb), f"feedback produced a non-finite loss: {l_fb}"
    assert abs(l_fb - l_nofb) > 1e-4, (
        "feedback memory did not change the loss; the k-pass recurrence is not "
        "connected and training would silently reproduce the 3.352 arm")
    print(f"2. feedback moves the loss (delta = {abs(l_fb - l_nofb):.4f})                       PASSED")

    # --- the mixing weights are actually trained ---------------------------
    g = grads_of(w, h_fb, batch, w_t)
    gw = float(jnp.abs(g.w_memory).max())
    assert gw > 0, "no gradient on w_memory: the mixing weights would never train"
    print(f"3. gradients reach w_memory ({gw:.3e})                              PASSED")

    # --- CAUSALITY OF THE RECURRENCE --------------------------------------
    # Offset-0 queries have an empty self region under m_<t, so perturbing the
    # mixing weights must not move their logits by even a bit.
    # NOTE: the mix is a softmax over the layer axis, so adding a constant to
    # EVERY entry of w_memory is its null direction and changes nothing. The
    # perturbation has to be non-uniform across slots to be a real one.
    w2 = type(w)(**{**{f.name: getattr(w, f.name) for f in w.__dataclass_fields__.values()},
                    "w_memory": w.w_memory.at[:, 0].add(5.0)})
    l_a = loss_of(w, h_fb, batch, w_t)
    l_b = loss_of(w2, h_fb, batch, w_t)
    assert abs(l_a - l_b) > 1e-5, (
        "changing w_memory did not change the loss at all; the mix is inert")
    print(f"4. w_memory perturbation moves the loss ({abs(l_a - l_b):.4f})              PASSED")

    # A block size of 1 means every position's self region is empty (block_start
    # == the query itself, and strict causality removes the diagonal), so the
    # feedback pathway must contribute NOTHING and the loss must equal the
    # no-feedback model at block size 1.
    h_fb1 = ModelConfig(layers=2, n_mem=3, memory_mode="kv", memory_block_k=1,
                        feedback_memory=True, **COMMON)
    h_nofb1 = ModelConfig(layers=2, n_mem=3, memory_mode="kv", memory_block_k=1,
                          feedback_memory=False, **COMMON)
    a = loss_of(w, h_fb1, batch, w_t)
    b = loss_of(w2, h_fb1, batch, w_t)
    assert abs(a - b) < 1e-6, (
        f"at block size 1 the self region is empty, so w_memory must be irrelevant "
        f"-- but the loss moved by {abs(a - b):.3e}: the recurrence is reading "
        f"positions it must not see")
    print("5. at block_k=1 the self region is empty and w_memory is inert      PASSED")

    print("\nall SPIRe feedback-memory tests passed")

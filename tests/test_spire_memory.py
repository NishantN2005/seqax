"""SPIRe's memory pathway: the operator, not the plumbing.

Three times on this project an internal metric agreed with itself while the
deployed behaviour was broken -- the missing BOS (eval loss healthy, generation
degenerate), the memory leak (training loss BETTER than correct), and Phase B
(loss 0.75, tau 1.000). Each time the cause was a test that compared the
implementation against itself.

So the two central tests here are EXACT REDUCTIONS to the plain model along paths
where the memory pathway must provably contribute nothing, plus a teeth test that
fails if those reductions are vacuous.

  1. decode path, no substitution   memory_mask all False  =>  bitwise plain
  2. block path, single block       block_k = L            =>  bitwise plain
  3. teeth                          block_k = 4, real mem  =>  MUST differ
  4. the split is a partition       m_tgt and m_self disjoint, union = mask
  5. memory reaches the output      changing memory changes the logits
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # CPU: exactness, and no contention with a training GPU
import init_seqax  # noqa: F401,E402

from functools import partial  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import mesh_utils  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from typeguard import typechecked  # noqa: E402

import shardlib.shardtypes as shardtypes  # noqa: E402
from shardlib.shardtypes import bf16, bool_, f32, u32  # noqa: E402

shardtypes.register_with_typeguard()

from model import Model, ModelConfig  # noqa: E402

B, L, NMEM = 8, 16, 3
BASE = dict(vocab=256, seq_len=L, layers=2, d_model=64, n_q_per_kv=1, n_kv=8,
            d_head=16, d_ff=128, rope_max_timescale=256)
h_plain = ModelConfig(**BASE)
h_kv = ModelConfig(**BASE, n_mem=NMEM, memory_mode="kv")


def run(h, w, ids, mem=None, mmask=None, block_k=None):
    @jax.jit
    @partial(shardtypes.typed_shard_map, check_rep=False)
    @typechecked
    def fwd(w: Model, x: u32[b"B/d L"], mem: bf16[b"n_mem B/d L M"],
            mmask: bool_[b"B/d L"]) -> f32[b"B/d L V"]:
        causal = jnp.tril(jnp.ones((L, L), dtype=jnp.bool_))[jnp.newaxis, ...]
        mask = jnp.broadcast_to(causal, (x.shape[0], L, L))
        with shardtypes.Scope():
            out = w.forward_pass(
                h, x, mask,
                memory=(None if mem_off else mem),
                memory_mask=(None if mmask_off else mmask),
                memory_block_k=block_k,
            )
        return out[0]

    mem_off = mem is None
    mmask_off = mmask is None
    # Size the placeholder to THIS config's n_mem: shardlib dimension names are
    # global, so a [3, ...] array against an n_mem=0 model is a type error.
    mem_ = jnp.zeros((h.n_mem, B, L, BASE["d_model"]), jnp.bfloat16) if mem is None else mem
    mmask_ = jnp.zeros((B, L), jnp.bool_) if mmask is None else mmask
    with shardtypes.Scope():
        return np.asarray(fwd(w, jnp.asarray(ids, jnp.uint32), mem_, mmask_))


with Mesh(mesh_utils.create_device_mesh([1, 1, 1], jax.devices()[:1]), ("d", "t", "s")):
    rng = jax.random.PRNGKey(0)
    key = jax.random.key_data(rng).astype(jnp.uint32)
    # Each init binds `n_mem`, so each needs its own Scope.
    with shardtypes.Scope():
        w_plain = jax.jit(Model.init, static_argnums=0)(h_plain, key)
    with shardtypes.Scope():
        w_kv = jax.jit(Model.init, static_argnums=0)(h_kv, key)
    ids = jax.random.randint(rng, (B, L), 0, BASE["vocab"]).astype(jnp.uint32)
    mem = jax.random.normal(jax.random.PRNGKey(7), (NMEM, B, L, BASE["d_model"]), jnp.float32)
    mem = jnp.bfloat16(mem)

    base = run(h_plain, w_plain, ids)

    # w_memory is zeros and consumes no rng, so the two inits must agree on every
    # other weight; if they did not, the comparisons below would be meaningless.
    assert np.array_equal(np.asarray(w_kv.embed), np.asarray(w_plain.embed)), \
        "n_mem changed the weight init; the exactness tests would be comparing different models"

    # 1. Decode path with nothing substituted. kv_src = where(False, mem, x) = x,
    #    so the keys and values are the draft's own and this must be the plain model.
    got = run(h_kv, w_kv, ids, mem=mem, mmask=jnp.zeros((B, L), jnp.bool_), block_k=0)
    assert np.array_equal(got, base), (
        "kv pathway with no substitution is not the plain model; max |diff| = "
        f"{np.abs(got - base).max()}")
    print("1. decode path, memory_mask all False -> bitwise identical to plain  PASSED")

    # 2. Block path with one block spanning the sequence: block_start(t) = 0 for
    #    every t, so no key is target-sourced and the joint softmax must collapse
    #    onto the self source alone. This is the test of the two-source machinery.
    got = run(h_kv, w_kv, ids, mem=mem, block_k=L)
    assert np.array_equal(got, base), (
        "two-source attention with an empty target side is not the plain model; "
        f"max |diff| = {np.abs(got - base).max()}")
    print("2. block path, block_k = L -> bitwise identical to plain            PASSED")

    # 3. Teeth. If a real block split did not change the output, 1 and 2 would be
    #    passing vacuously -- the memory could be disconnected entirely.
    got4 = run(h_kv, w_kv, ids, mem=mem, block_k=4)
    assert not np.array_equal(got4, base), (
        "block_k=4 with nonzero memory produced the plain model's logits: the "
        "target activations are not reaching attention, so tests 1-2 are vacuous")
    print(f"3. block_k = 4 differs from plain (max |diff| = {np.abs(got4 - base).max():.4f})  PASSED")

    # 4. The split must be a partition of the ordinary mask: every key is read
    #    from exactly one source. An overlap would double-count a key in the
    #    denominator; a gap would silently drop context.
    for bk in (1, 2, 4, 8):
        q = np.arange(L)[:, None]
        kp = np.arange(L)[None, :]
        causal = kp <= q
        from_target = kp < (q // bk) * bk
        m_t = causal & from_target
        m_s = causal & ~from_target
        assert not (m_t & m_s).any(), f"block_k={bk}: sources overlap"
        assert ((m_t | m_s) == causal).all(), f"block_k={bk}: sources do not cover the mask"
        # Every query must see at most bk self-sourced keys: the paper drafts k
        # tokens per round, so a wider self region would train a regime decode
        # never meets.
        assert m_s.sum(axis=1).max() <= bk, f"block_k={bk}: self region wider than k"
    print("4. target/self masks partition the attention mask for k in 1,2,4,8   PASSED")

    # 5. The substitution must actually carry information. Same everything,
    #    different target activations -> different logits.
    mem2 = jnp.bfloat16(jax.random.normal(jax.random.PRNGKey(9), mem.shape, jnp.float32))
    got5 = run(h_kv, w_kv, ids, mem=mem2, block_k=4)
    assert not np.array_equal(got5, got4), (
        "changing the target activations did not change the logits; the memory is "
        "being ignored")
    print("5. logits depend on the target activations                          PASSED")

    # 6. The projection shortcut must be EXACT. At decode the draft skips its own
    #    forward pass over the prompt entirely and builds its cache by projecting
    #    the target's activations. That is only legitimate if the result is
    #    identical to what a forward pass would have written.
    @jax.jit
    @partial(shardtypes.typed_shard_map, check_rep=False)
    @typechecked
    def cache_via_forward(w: Model, x: u32[b"B/d L"],
                          mem: bf16[b"n_mem B/d L M"]) -> bf16[b"layers 2 B/d L K D"]:
        causal = jnp.tril(jnp.ones((L, L), dtype=jnp.bool_))[jnp.newaxis, ...]
        mask = jnp.broadcast_to(causal, (x.shape[0], L, L))
        cache = jnp.zeros((h_kv.layers, 2, x.shape[0], L, h_kv.n_kv, h_kv.d_head), jnp.bfloat16)
        with shardtypes.Scope():
            out = w.forward_pass(h_kv, x, mask, kv_cache=cache,
                                 kv_offset=jnp.zeros((x.shape[0],), jnp.int32),
                                 memory=mem, memory_block_k=None)
        return out[1]

    @jax.jit
    @partial(shardtypes.typed_shard_map, check_rep=False)
    @typechecked
    def cache_via_projection(w: Model, mem: bf16[b"n_mem B/d L M"]) -> bf16[b"layers 2 B/d L K D"]:
        with shardtypes.Scope():
            return w.memory_to_kv(h_kv, mem)

    with shardtypes.Scope():
        c_fwd = np.asarray(cache_via_forward(w_kv, jnp.asarray(ids, jnp.uint32), mem))
    with shardtypes.Scope():
        c_proj = np.asarray(cache_via_projection(w_kv, mem))
    assert c_fwd.shape == c_proj.shape, f"{c_fwd.shape} vs {c_proj.shape}"
    assert np.array_equal(c_fwd, c_proj), (
        "memory_to_kv does not reproduce the cache a forward pass writes; the "
        f"decode shortcut is not exact. max |diff| = {np.abs(c_fwd - c_proj).max()}")
    print("6. memory_to_kv == the cache a forward pass writes (bitwise)         PASSED")

    print("\nall SPIRe memory-pathway tests passed")

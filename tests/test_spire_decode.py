"""SPIRe's decode path against an independent oracle.

Decode and training are different machinery -- one walks a KV cache one token at
a time, the other runs a masked forward pass over the whole sequence -- and the
project's recurring failure has been the two silently disagreeing while every
self-consistent check stayed green. So the reference here is built from TRAINING
semantics and shares no code with the decode loop.

The alignment that makes them comparable: choose the prompt so the first drafted
position sits at offset 0 of a block. Then block_start == that position, every
earlier key is a target activation, and (under strict m_<t) the self region is
empty -- exactly the decode condition for the first proposal. The second
proposal, at offset 1, additionally reads the feedback memory written for the
first, which is what exercises the recurrence.

    PYTHONPATH=. python tests/test_spire_decode.py
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
import init_seqax  # noqa: E402, F401

from functools import partial  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import mesh_utils  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import PartitionSpec as PS  # noqa: E402

import shardlib.shardtypes as shardtypes  # noqa: E402
from shardlib.shardtypes import bf16, f32, u32  # noqa: E402

shardtypes.register_with_typeguard()

from model import Model, ModelConfig, streaming_visibility  # noqa: E402
from speculative import make_speculative_generate  # noqa: E402

VOCAB, B, P, K = 128, 4, 7, 4          # Pf = P+1 = 8, a multiple of K
SINK, WINDOW = 1, 8
COMMON = dict(vocab=VOCAB, seq_len=64, d_model=32, n_q_per_kv=1, n_kv=4,
              d_head=8, d_ff=64, rope_max_timescale=100)
h_t = ModelConfig(layers=4, **COMMON)
h_d = ModelConfig(layers=2, n_mem=3, memory_mode="kv", memory_block_k=K,
                  feedback_memory=True, attention_mask="streaming_llm",
                  sink_size=SINK, window=WINDOW, **COMMON)
Pf = P + 1
LO = h_t.layers - h_d.layers - 1        # 1
R = 1
S = 1 + R * (K + 1)
KLEN = Pf + S + K + 1


MODEL_SPEC = None   # filled once a Mesh exists


def _sm(local, in_specs, out_specs):
    mesh = jax._src.mesh.thread_resources.env.physical_mesh
    return jax.jit(jax.experimental.shard_map.shard_map(
        local, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False))


def target_ref(w, ids):
    """Dense causal target pass, returning logits and per-layer activations.

    Explicit partition specs rather than annotation-derived ones: `layers` is a
    global shardlib dimension name, so a 4-layer target and a 2-layer draft
    cannot both describe their outputs with it in the same process.
    """
    n = ids.shape[1]

    def local(w, x):
        m = jnp.broadcast_to(jnp.tril(jnp.ones((n, n), jnp.bool_))[jnp.newaxis], (x.shape[0], n, n))
        with shardtypes.Scope():
            out = w.forward_pass(h_t, x, m, emit_activations=True)
        return out[0], out[3]

    fn = _sm(local, (MODEL_SPEC, PS("d", None)),
             (PS("d", None, None), PS(None, "d", None, None)))
    return fn(w, jnp.asarray(ids, jnp.uint32))


def draft_ref(w, ids, mem, self_mem):
    """Draft forward under TRAINING semantics: no cache, block split, strict m_<t."""
    n = ids.shape[1]

    def local(w, x, mem, sm):
        q = jnp.arange(n)[jnp.newaxis, :, jnp.newaxis]
        kk = jnp.arange(n)[jnp.newaxis, jnp.newaxis, :]
        m = jnp.broadcast_to(streaming_visibility(q, kk, SINK, WINDOW), (x.shape[0], n, n))
        with shardtypes.Scope():
            out = w.forward_pass(h_d, x, m, memory=mem, w_memory=w.w_memory,
                                 memory_block_k=K, self_memory=sm,
                                 emit_activations=True)
        return out[0], out[3], out[4]

    fn = _sm(local,
             (MODEL_SPEC, PS("d", None), PS(None, "d", None, None), PS(None, "d", None, None)),
             (PS("d", None, None), PS(None, "d", None, None), PS("d", None, None)))
    return fn(w, jnp.asarray(ids, jnp.uint32), mem, self_mem)


with Mesh(mesh_utils.create_device_mesh([1, 1, 1], jax.devices()[:1]), ("d", "t", "s")):
    MODEL_SPEC = shardtypes.make_partition_specs(Model)
    key = jax.random.key_data(jax.random.PRNGKey(0)).astype(jnp.uint32)
    with shardtypes.Scope():
        w_t = jax.jit(Model.init, static_argnums=0)(h_t, key)
    with shardtypes.Scope():
        w_d = jax.jit(Model.init, static_argnums=0)(h_d, key)
    # A zero w_memory would make the mix uniform; a random one exercises it.
    w_d = type(w_d)(**{**{f.name: getattr(w_d, f.name) for f in w_d.__dataclass_fields__.values()},
                       "w_memory": jax.random.normal(jax.random.PRNGKey(5), (h_d.layers, h_d.n_mem))})
    prompt = np.asarray(jax.random.randint(jax.random.PRNGKey(2), (B, P), 1, VOCAB).astype(jnp.uint32))

    # ---- decode path ----
    gen = make_speculative_generate(
        h_t, h_d, P, R, K, 0.0, draft_sink=SINK, draft_window=WINDOW,
        draft_prefill_dense=False, magicdec_rope=False, klen=KLEN, return_drafts=True)
    with shardtypes.Scope():
        _, _, _, drafts, _ = gen(w_t, w_d, jnp.asarray(prompt), jnp.zeros((2,), jnp.uint32))
    drafts = np.asarray(drafts)                       # [B, K]

    # ---- oracle: first proposal (offset 0, empty self region) ----
    ids0 = np.concatenate([np.zeros((B, 1), np.uint32), prompt], axis=1)   # [B, Pf]
    tl, _ = target_ref(w_t, ids0)
    cur = np.asarray(jnp.argmax(tl[:, -1], -1)).astype(np.uint32)          # token at position Pf
    ids1 = np.concatenate([ids0, cur[:, None]], axis=1)                    # [B, Pf+1]
    _, tacts = target_ref(w_t, ids1)
    mem1 = jnp.bfloat16(tacts[LO:LO + h_d.n_mem])
    zero_sm = jnp.zeros((h_d.layers, B, ids1.shape[1], COMMON["d_model"]), jnp.bfloat16)
    dl, dacts, demb = draft_ref(w_d, ids1, mem1, zero_sm)
    want1 = np.asarray(jnp.argmax(dl[:, Pf], -1)).astype(np.uint32)

    agree1 = float((drafts[:, 0] == want1).mean())
    print(f"first proposal: decode vs training-semantics reference = {agree1:.3f} agreement")
    assert agree1 == 1.0, (
        f"decode disagrees with training semantics on the FIRST draft.\n"
        f"  decode {drafts[:, 0].tolist()}\n  ref    {want1.tolist()}")
    print("1. first proposal matches (prefill projection, strict mask, rope)   PASSED")

    # ---- oracle: second proposal (offset 1, reads the feedback memory) ----
    ids2 = np.concatenate([ids1, want1[:, None]], axis=1)                  # [B, Pf+2]
    _, tacts2 = target_ref(w_t, ids2)
    mem2 = jnp.bfloat16(tacts2[LO:LO + h_d.n_mem])
    zero_sm2 = jnp.zeros((h_d.layers, B, ids2.shape[1], COMMON["d_model"]), jnp.bfloat16)
    dl2, dacts2, demb2 = draft_ref(w_d, ids2, mem2, zero_sm2)
    # Feedback memory for position Pf, from the draft's own states there.
    states = jnp.concatenate([demb2[jnp.newaxis], dacts2], axis=0)         # [layers+1, B, L, M]
    m_at = Model.mix_memory(w_d.w_memory, states)                          # [layers, B, L, M]
    sm2 = zero_sm2.at[:, :, Pf, :].set(m_at[:, :, Pf, :])
    dl3, _, _ = draft_ref(w_d, ids2, mem2, sm2)
    want2 = np.asarray(jnp.argmax(dl3[:, Pf + 1], -1)).astype(np.uint32)

    agree2 = float((drafts[:, 1] == want2).mean())
    print(f"second proposal: decode vs reference = {agree2:.3f} agreement")
    assert agree2 == 1.0, (
        f"decode disagrees on the SECOND draft, which is the first one that reads "
        f"feedback memory.\n  decode {drafts[:, 1].tolist()}\n  ref    {want2.tolist()}")
    print("2. second proposal matches (feedback memory written and read)       PASSED")

    # Teeth: if the feedback memory were ignored, proposal 2 would equal the value
    # computed with an all-zero self memory. It must not.
    want2_nofb = np.asarray(jnp.argmax(dl2[:, Pf + 1], -1)).astype(np.uint32)
    assert not np.array_equal(want2, want2_nofb), (
        "the reference gives the same second proposal with and without feedback "
        "memory, so test 2 cannot detect whether decode uses it")
    print("3. feedback memory changes the second proposal (test has teeth)     PASSED")

    print("\nall SPIRe decode tests passed")

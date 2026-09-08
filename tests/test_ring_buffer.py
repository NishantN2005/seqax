"""The compact sliding-window draft cache against the full masked cache.

SPIRe's claim is that the draft's KV cache is constant in the decoding sequence
length. A mask over a full-length cache gives the right distributions but reads
every position anyway, so it can demonstrate tau and nothing about cost. The
compact path stores sink + window entries and cycles them.

The oracle is the full-cache path, which is already validated by test_spec and
test_spire_decode. If the ring's slot arithmetic, its slot->position map, its
RoPE positions or its eviction order were wrong, the two would disagree.

Geometry is chosen so the ring definitely WRAPS -- window 8 against ~40 absolute
positions, so every slot is overwritten four times. A ring that never wraps would
pass any test vacuously.

    PYTHONPATH=. python tests/test_ring_buffer.py
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
import init_seqax  # noqa: E402, F401

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import mesh_utils  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

import shardlib.shardtypes as shardtypes  # noqa: E402

shardtypes.register_with_typeguard()

from model import Model, ModelConfig  # noqa: E402
from speculative import make_speculative_generate  # noqa: E402

VOCAB, B, P, K, R = 128, 4, 15, 4, 4
SINK, WINDOW = 1, 8
COMMON = dict(vocab=VOCAB, seq_len=64, d_model=32, n_q_per_kv=1, n_kv=4,
              d_head=8, d_ff=64, rope_max_timescale=100)
h_t = ModelConfig(layers=4, **COMMON)
h_spire = ModelConfig(layers=2, n_mem=3, memory_mode="kv", memory_block_k=K,
                      feedback_memory=True, attention_mask="streaming_llm",
                      sink_size=SINK, window=WINDOW, **COMMON)
Pf = P + 1
S = 1 + R * (K + 1)
KLEN = Pf + S + K + 1
C_D = SINK + WINDOW

C_D = SINK + WINDOW + K
print(f"geometry: Klen={KLEN}, compact C_d={SINK}+{WINDOW}+{K}={C_D} "
      f"({KLEN / C_D:.1f}x smaller); positions {Pf}..{Pf + S} cycle a "
      f"{WINDOW + K}-slot ring {S / (WINDOW + K):.1f} times")


def run(w_t, w_d, h_d, prompt, *, compact, slack=None, **kw):
    gen = make_speculative_generate(
        h_t, h_d, P, R, K, 1.0, draft_sink=SINK, draft_window=WINDOW,
        klen=KLEN, return_drafts=True, compact_draft_cache=compact,
        _ring_slack=slack, **kw)
    with shardtypes.Scope():
        out, n_gen, n_acc, drafts, qs = gen(w_t, w_d, jnp.asarray(prompt),
                                            jnp.zeros((2,), jnp.uint32))
    return tuple(np.asarray(x) for x in (out, n_gen, n_acc, drafts, qs))


with Mesh(mesh_utils.create_device_mesh([1, 1, 1], jax.devices()[:1]), ("d", "t", "s")):
    key = jax.random.key_data(jax.random.PRNGKey(0)).astype(jnp.uint32)
    with shardtypes.Scope():
        w_t = jax.jit(Model.init, static_argnums=0)(h_t, key)
    with shardtypes.Scope():
        w_spire = jax.jit(Model.init, static_argnums=0)(h_spire, key)
    w_spire = type(w_spire)(**{**{f.name: getattr(w_spire, f.name)
                                  for f in w_spire.__dataclass_fields__.values()},
                               "w_memory": jax.random.normal(jax.random.PRNGKey(5), (2, 3))})
    prompt = np.asarray(jax.random.randint(jax.random.PRNGKey(2), (B, P), 1, VOCAB).astype(jnp.uint32))

    cases = [
        ("SPIRe (memory pathway + feedback)", w_spire, h_spire,
         dict(draft_prefill_dense=False, magicdec_rope=False)),
        ("MagicDec (target weights, cache-relative rope)", w_t, h_t,
         dict(draft_prefill_dense=True, magicdec_rope=True)),
        ("MagicDec (absolute rope)", w_t, h_t,
         dict(draft_prefill_dense=True, magicdec_rope=False)),
    ]
    # An untrained draft has near-uniform logits, so its argmax is decided by float
    # noise and token equality would prove nothing. The DISTRIBUTIONS are the
    # observable: a correct ring differs from the full cache only in the order the
    # masked-out zeros enter the softmax reduction, while a ring that evicts a live
    # position, mislabels a slot, or misplaces RoPE feeds attention different
    # inputs entirely.
    TOL = 1e-4
    n = 0
    for name, w_d, h_d, kw in cases:
        full = run(w_t, w_d, h_d, prompt, compact=False, **kw)
        comp = run(w_t, w_d, h_d, prompt, compact=True, **kw)
        dq = float(np.abs(full[4] - comp[4]).max())
        assert dq < TOL, (
            f"{name}: the compact ring feeds attention different inputs than the "
            f"full masked cache -- max |q_full - q_compact| = {dq:.3e} "
            f"(tol {TOL:g}). A correct ring differs only by summation order.")
        for label, a, b in (("n_generated", full[1], comp[1]),
                            ("n_accepted", full[2], comp[2])):
            assert np.array_equal(a, b), f"{name}/{label}: {a.tolist()} vs {b.tolist()}"
        n += 1
        print(f"{n}. {name:<46} max |dq| = {dq:.1e}  PASSED")

    # Teeth 1: the ring has to have wrapped, or eviction was never exercised.
    total = int(full[1].sum())
    assert total > WINDOW, f"only {total} tokens generated; the ring never wrapped"
    print(f"{n + 1}. ring wrapped: positions {Pf}..{Pf + S} over a {WINDOW}-token window     PASSED")

    # Teeth 2: the +k slack is load-bearing, not padding. At slack 0 the ring is
    # sink + window -- the size the paper's cost model charges for -- and the
    # speculative writes that end up rejected destroy positions the next query
    # still needs. This MUST diverge; if it does not, the tolerance above is
    # meaningless and the whole comparison is vacuous.
    narrow = run(w_t, w_spire, h_spire, prompt, compact=True, slack=0,
                 draft_prefill_dense=False, magicdec_rope=False)
    full_s = run(w_t, w_spire, h_spire, prompt, compact=False,
                 draft_prefill_dense=False, magicdec_rope=False)
    dq_bad = float(np.abs(full_s[4] - narrow[4]).max())
    assert dq_bad > TOL, (
        f"a ring of only sink+window agreed with the full cache to {dq_bad:.3e}; "
        f"either this geometry never rejects a draft, or the comparison cannot "
        f"detect eviction of a live position")
    print(f"{n + 2}. sink+window alone DIVERGES (max |dq| = {dq_bad:.1e}): "
          f"the +k slack is required   PASSED")

    print("\nall ring-buffer tests passed")

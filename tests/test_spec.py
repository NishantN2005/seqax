"""Step-10/11 tests: speculative decoding correctness (losslessness).

A. Self-drafting (draft == target, dense): greedy SD output must equal plain
   greedy decoding AND every draft token must be accepted (n_acc == k).
B. Vanilla SD (independent small draft): greedy SD output must STILL equal
   plain greedy decoding — losslessness does not depend on draft quality.
C. MagicDec-style (draft = target weights, sink+window mask): same guarantee.
D. temperature > 0: the marginal distribution of the first SD-generated token
   matches plain sampling (statistical check over a large batch).
"""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
import init_seqax  # noqa: F401
import numpy as np
import jax, jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh

import dataclasses

import shardlib.shardtypes as shardtypes
shardtypes.register_with_typeguard()
from model import Model, ModelConfig
from decode import make_generate
from speculative import make_speculative_generate

h = ModelConfig(vocab=256, seq_len=32, layers=2, d_model=64, n_q_per_kv=1, n_kv=8,
                d_head=16, d_ff=128, rope_max_timescale=256)
h_small = ModelConfig(vocab=256, seq_len=32, layers=1, d_model=32, n_q_per_kv=1, n_kv=4,
                      d_head=8, d_ff=64, rope_max_timescale=256)
B, Pn, K, R = 8, 12, 4, 6
G = 1 + R * (K + 1)

with Mesh(mesh_utils.create_device_mesh([8, 1, 1], jax.devices()), ("d", "t", "s")):
    rng = jnp.zeros((2,), dtype=jnp.uint32)
    with shardtypes.Scope():
        w_t = jax.jit(Model.init, static_argnums=0)(h, rng)
    with shardtypes.Scope():
        w_d = jax.jit(Model.init, static_argnums=0)(h_small, jnp.ones((2,), dtype=jnp.uint32))
    # Sharpen output distributions so no logit gap is within bf16 noise of a tie.
    # Batched (verify) and incremental (draft/plain-decode) passes reduce in
    # different orders; near-ties would make exact-equality greedy tests flaky.
    w_t = dataclasses.replace(w_t, unembed=w_t.unembed * 8.0)
    w_d = dataclasses.replace(w_d, unembed=w_d.unembed * 8.0)
    prompt = jax.random.randint(jax.random.PRNGKey(0), (B, Pn), 0, h.vocab).astype(jnp.uint32)

    # Reference: plain greedy decoding, enough tokens to cover the max any test needs.
    ref = make_generate(h, Pn, G, 0.0)(w_t, prompt, rng)

    def compare(tag, out, n_gen):
        n = int(jnp.min(n_gen))
        ok = bool(jnp.all(out[:, :n] == ref[:, :n]))
        print(f"{tag}: min tokens generated {n}, losslessness {'PASSED' if ok else 'FAILED'}")
        assert ok, tag
        return n

    # ---- A. self-draft, dense ----
    sg = make_speculative_generate(h, h, Pn, R, K, 0.0)
    out, n_gen, n_acc = sg(w_t, w_t, prompt, rng)
    compare("A self-draft", out, n_gen)
    assert bool(jnp.all(n_acc == K)), "self-draft greedy must accept every draft token"
    print(f"A self-draft: all {K} drafts accepted every round PASSED")

    # ---- B. vanilla small draft ----
    sg = make_speculative_generate(h, h_small, Pn, R, K, 0.0)
    out, n_gen, n_acc = sg(w_t, w_d, prompt, rng)
    compare("B vanilla", out, n_gen)
    print(f"B vanilla: tau = {float(jnp.mean(n_acc)) + 1:.3f} (draft is random-init; expect ~1)")

    # ---- C. MagicDec-style: target drafts itself through sink+window ----
    sg = make_speculative_generate(h, h, Pn, R, K, 0.0, draft_sink=1, draft_window=4,
                                   draft_prefill_dense=True)
    out, n_gen, n_acc = sg(w_t, w_t, prompt, rng)
    compare("C magicdec", out, n_gen)
    print(f"C magicdec: tau = {float(jnp.mean(n_acc)) + 1:.3f}")

    # ---- C2. MagicDec with cache-relative rope (paper footnote 4) ----
    sg = make_speculative_generate(h, h, Pn, R, K, 0.0, draft_sink=1, draft_window=4,
                                   draft_prefill_dense=True, magicdec_rope=True)
    out, n_gen, n_acc_mdr = sg(w_t, w_t, prompt, rng)
    compare("C2 magicdec+cache-rope", out, n_gen)
    print(f"C2 magicdec+cache-rope: tau = {float(jnp.mean(n_acc_mdr)) + 1:.3f}")
    assert not bool(jnp.all(n_acc_mdr == n_acc)), "cache-relative rope should change draft behaviour"
    print("C2: rope convention changes acceptance (as expected) PASSED")

    # ---- D. temperature: first-SD-token marginal matches plain sampling ----
    hv = ModelConfig(vocab=32, seq_len=16, layers=1, d_model=32, n_q_per_kv=1, n_kv=4,
                     d_head=8, d_ff=64, rope_max_timescale=64)
    Bd, Pd, T = 4096, 4, 0.9
    with shardtypes.Scope():
        w_v = jax.jit(Model.init, static_argnums=0)(hv, rng)
    with shardtypes.Scope():
        w_vd = jax.jit(Model.init, static_argnums=0)(hv, jnp.ones((2,), dtype=jnp.uint32))
    prm = jnp.tile(jnp.array([[3, 1, 4, 1]], dtype=jnp.uint32), (Bd, 1))  # identical prompts
    # plain sampling: token index 1 (second generated) across the batch
    with shardtypes.Scope():
        plain = make_generate(hv, Pd, 2, T)(w_v, prm, rng)[:, 1]
    sgd = make_speculative_generate(hv, hv, Pd, 1, 2, T)
    with shardtypes.Scope():
        spec, _, _ = sgd(w_v, w_vd, prm, jnp.full((2,), 5, jnp.uint32))
    spec = spec[:, 1]  # first SD-produced token (always valid: n >= 1 per round... index1 needs n_gen>=2)
    hp = np.bincount(np.array(plain), minlength=hv.vocab) / Bd
    hs = np.bincount(np.array(spec), minlength=hv.vocab) / Bd
    dmax = float(np.max(np.abs(hp - hs)))
    print(f"D temperature: max marginal freq diff = {dmax:.4f} (n={Bd})")
    assert dmax < 0.03, "SD marginal deviates from target sampling"
    print("D temperature distributional check PASSED")
"""The transcribed cost model against the paper's own published figures.

`calculate_itm` is a transcription of spire_appendix.ipynb, and the whole project
turns on whether it says what the paper says: every "measured vs predicted"
number is a comparison against this function. A transcription error would not
show up as a crash, it would show up as a finding.

So this checks it against values the paper published rather than against itself:
the three body-parameter counts printed in the notebook, and the three throughput
multipliers highlighted in Figure 4 at B=64, L=512, k=4, HOI=600 (H100 SXM, which
is the card the paper plots).

    PYTHONPATH=. python tests/test_cost_model.py
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

from bench import (body_params, calculate_itm,  # noqa: E402
                   draft_kv_projection_params, kv_elements_per_token)
from model import ModelConfig  # noqa: E402

C = dict(vocab=50304, seq_len=1024, rope_max_timescale=10000)
# spire_appendix.ipynb cell 4, verbatim.
h_t = ModelConfig(layers=8, d_model=512, n_q_per_kv=1, n_kv=8, d_head=128, d_ff=4096, **C)
h_v = ModelConfig(layers=4, d_model=256, n_q_per_kv=1, n_kv=8, d_head=64, d_ff=2048, **C)
h_s = ModelConfig(layers=2, d_model=512, n_q_per_kv=1, n_kv=8, d_head=128, d_ff=4096, **C)

# --- 1. body parameters, printed by the notebook --------------------------
want = {"target": 67_117_056, "vanilla draft": 8_390_656, "SPIRe draft": 16_779_264}
got = {"target": body_params(h_t), "vanilla draft": body_params(h_v),
       "SPIRe draft": body_params(h_s)}
for name in want:
    assert got[name] == want[name], f"{name}: {got[name]:,} != {want[name]:,}"
print(f"1. body parameters exact: {got['target']:,} / {got['vanilla draft']:,} / "
      f"{got['SPIRe draft']:,}   PASSED")
# The paper says 1/8 and 1/4 "body parameters". That holds exactly for the matmul
# weights and NOT for the raw totals: layernorms scale with layers x d_model, not
# with the quadratic terms, so they leave 8,192 vs 2,048 residue. Strip them and
# the target is 2**26 with the drafts at 2**23 and 2**24, exactly.
def matmuls(h):
    return body_params(h) - h.layers * 2 * h.d_model

assert matmuls(h_t) == 2 ** 26 and matmuls(h_v) == 2 ** 23 and matmuls(h_s) == 2 ** 24
assert matmuls(h_v) * 8 == matmuls(h_t), "vanilla draft is not 1/8 the target"
assert matmuls(h_s) * 4 == matmuls(h_t), "SPIRe draft is not 1/4 the target"
print("   1/8 and 1/4 exact in the matmul weights (layernorms aside)        PASSED")

# --- 2. Figure 4's highlighted column --------------------------------------
B, L, k, HOI = 64, 512, 4, 600.0
N_t, N_v, N_s = body_params(h_t), body_params(h_v), body_params(h_s)
sink, win = 1, L // 8          # window_factor 8; at L=512 that is the paper's 64
assert win == 64

itm = {
    # vanilla: dense draft, reads the whole context
    "vanilla": calculate_itm(B, L, h_t, k, HOI, N_t, N_v, L, h_v, 0),
    # MagicDec: the TARGET's weights, reading a sink+window slice of its cache
    "magicdec": calculate_itm(B, L, h_t, k, HOI, N_t, N_t, sink + win, h_t, 0),
    # SPIRe: quarter-size draft, sparse cache, plus the memory-vector projection
    "spire": calculate_itm(B, L, h_t, k, HOI, N_t, N_s, sink + win, h_s,
                           draft_kv_projection_params(h_s)),
}
taus = {"vanilla": 2.647, "magicdec": 3.891, "spire": 3.401}   # Table 1, L=512
fig4 = {"vanilla": 1.36, "magicdec": 2.05, "spire": 2.78}      # Figure 4, B=64 L=512

print(f"\n   {'method':<10}{'ITM':>8}{'tau/ITM':>9}{'Figure 4':>10}{'err':>8}")
for m in ("vanilla", "magicdec", "spire"):
    t = taus[m] / itm[m]
    err = abs(t - fig4[m]) / fig4[m]
    print(f"   {m:<10}{itm[m]:>8.3f}{t:>9.2f}{fig4[m]:>10.2f}{err * 100:>7.1f}%")
    assert err < 0.01, (
        f"{m}: throughput multiplier {t:.3f} against the paper's {fig4[m]} "
        f"({err * 100:.1f}% off). The cost model is not saying what the paper says.")
print("2. all three throughput multipliers match Figure 4 within 1%        PASSED")

# --- 3. the claims the figures rest on -------------------------------------
# SPIRe beats MagicDec by ~35% and vanilla by ~100% at this cell.
r_md = (taus["spire"] / itm["spire"]) / (taus["magicdec"] / itm["magicdec"]) - 1
r_v = (taus["spire"] / itm["spire"]) / (taus["vanilla"] / itm["vanilla"]) - 1
print(f"3. SPIRe over MagicDec {r_md * 100:.0f}% (paper: 35%), "
      f"over vanilla {r_v * 100:.0f}% (paper: 100%)                 PASSED")
assert 0.30 < r_md < 0.40 and 0.95 < r_v < 1.10

# --- 4. SPIRe's memory-projection term: implemented, and inert where it counts -
# The appendix adds FLOPs for projecting a memory vector into a key and a value.
# At the paper's own headline cell that term changes NOTHING, because SPIRe's
# draft is memory-bound there by about 8x -- 2.03e10 against 2.42e9 -- so the
# roofline max is decided by the cache and the weights, not by FLOPs. The term is
# correct to include and it moves no number the paper reports. It is also the
# term that prices feedback memory, the component Figure 5 ranks 5th of 6.
no_proj = calculate_itm(B, L, h_t, k, HOI, N_t, N_s, sink + win, h_s, 0)
assert no_proj == itm["spire"], (
    "the projection term changed ITM at B=64 L=512, where the draft should be "
    "firmly memory-bound -- check the roofline")
flops = 2 * N_s * B + 2 * draft_kv_projection_params(h_s) * B
mem = (N_s + B * (sink + win) * kv_elements_per_token(h_s)) * HOI
print(f"4. projection term inert at B={B} L={L}: draft is memory-bound "
      f"{mem / flops:.1f}x    PASSED")

# But it must still be WIRED IN, or test 2 would pass with the term missing
# entirely. In the compute-bound corner (large batch, very short context) it bites.
Bc, Lc = 512, 32
kc = 1 + Lc // 8
with_p = calculate_itm(Bc, Lc, h_t, k, HOI, N_t, N_s, kc, h_s,
                       draft_kv_projection_params(h_s))
without_p = calculate_itm(Bc, Lc, h_t, k, HOI, N_t, N_s, kc, h_s, 0)
assert with_p != without_p, (
    "the projection term is inert even in the compute-bound corner, so it is not "
    "wired in at all and test 2 cannot detect its absence")
print(f"   and it DOES bite when compute-bound (B={Bc} L={Lc}): "
      f"{without_p:.3f} -> {with_p:.3f}      PASSED")

# --- 5. HOI cancels when every roofline term is memory-bound ---------------
alt = {m: v for m, v in zip(("vanilla", "magicdec", "spire"), (
    calculate_itm(B, L, h_t, k, 756.0, N_t, N_v, L, h_v, 0),
    calculate_itm(B, L, h_t, k, 756.0, N_t, N_t, sink + win, h_t, 0),
    calculate_itm(B, L, h_t, k, 756.0, N_t, N_s, sink + win, h_s,
                  draft_kv_projection_params(h_s))))}
same = [m for m in itm if abs(alt[m] - itm[m]) < 1e-9]
print(f"5. HOI 600 (SXM) vs 756 (PCIe) identical for: {', '.join(same) or 'none'}")
assert "vanilla" in same and "magicdec" in same, (
    "HOI should cancel from the ITM ratio wherever all three roofline terms are "
    "memory-bound; it did not, so the card would change our conclusions")

print("\nall cost-model tests passed")

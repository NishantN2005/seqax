# Results

Everything measured so far in the SPIRe reproduction. Companion to
[fidelity-ledger.md](fidelity-ledger.md), which records what these numbers can and
cannot be compared against.

As of 2026-09-03: both baselines are complete, **Phase A of the SPIRe draft**
(pruned init + sparse-mask training + MixedLoss with distillation and α) is trained
and measured, and **Phase B** (Phase A plus target-activation substitution with a
rollout split) is trained and measured. Phase B is a NEGATIVE result -- see
Finding 9.

---

## Models trained

| | target | vanilla draft | SPIRe Phase A | SPIRe Phase B |
|---|---|---|---|---|
| layers / d_model | 8 / 512 | 4 / 256 | 2 / 512 | 2 / 512 |
| d_head / d_ff | 128 / 4096 | 64 / 2048 | 128 / 4096 | 128 / 4096 |
| body params | 67,117,056 | 8,390,656 (= 1/8) | 16,779,264 (= 1/4) | 16,779,264 |
| total params | 118,628,352 | 34,146,304 | 68,290,560 | 68,290,560 |
| training tokens | 2,372,567,040 | 682,926,080 | 1,365,811,200 | 1,365,811,200 |
| steps @ batch 64 × 1024 | 36,203 | 10,421 | 20,841 | 20,841 |
| attention at training | dense | dense | **sink 1, window 64** | sink 1, window 64 |
| initialization | random | random | **pruned: target [6:8)** | pruned: target [6:8) |
| loss | hard CE | hard CE | **0.5·distill_CE + 0.5·(−α)** | same |
| target-activation memory | — | — | none | **n_mem 3, causal, rollout split** |
| **final eval loss** | **2.534** (σ 0.231) | **3.067** (σ 0.240) | **3.000** | **3.661** |
| perplexity | 12.6 | 21.5 | 20.1 | 38.9 |
| **τ @ L=512, T=1.0** | — | 2.566 | **2.851** | 2.769 |
| wall time (H100 PCIe) | ~5.5 h | ~45 min | ~2.5 h | ~3 h |

Phase B's eval loss is measured **memory-free** (eval supplies no teacher), which is
also the decode condition. It is worse than Phase A's because Phase B spent only
~37.5% of training in that regime against Phase A's 100% — the likely cause of its
lower τ. See Finding 9.

Note the SPIRe draft beats the vanilla draft on held-out hard-target CE (3.000 vs
3.067) **while attending to only 64 tokens rather than 1024**. Eval loss is measured
with the hard-target objective for all three, so the comparison is like-for-like
even though SPIRe trains on a different loss.

Both body-parameter counts reproduce `spire_appendix.ipynb`'s printed values
exactly. Eval is over 50,331,648 held-out tokens (768 steps).

### Learning-rate probe (target, 1,200-step scaled-down schedule)

| LR | final loss |
|---|---|
| 6.5e-3 | 3.609 |
| **3e-3** | **3.581** |
| 1.5e-3 | 3.649 |

Flat to within 2%, no divergence anywhere — gradient clipping to global norm 1.0
is what stabilizes this range, not the LR choice. 3e-3 adopted.

---

## Numerical agreement floor

Greedy self-drafting (draft ≡ target) must give τ = k+1 = 5.0 exactly.

**Measured: τ = 4.975 ± 0.023** → floor of **0.0254 tokens/round (0.51% of k+1)**.

The draft evaluates incrementally (L=1) while verification evaluates k+1 tokens
batched; in bf16 those reduce in different orders and near-tied argmaxes disagree.
This is a systematic **downward** bias on every τ below, it is the ceiling no draft
can exceed, and the analytical cost model cannot see it.

---

## τ measurements (k=4, sink=1, window=L/8, 128 contexts × 8 rounds)

### Vanilla speculative decoding

| L | τ (T=0.0) | acc. | τ (T=1.0) | acc. |
|---|---|---|---|---|
| 256 | 2.800 ± 0.099 | 0.450 | 2.349 ± 0.091 | 0.337 |
| **512** | 2.931 ± 0.101 | 0.483 | **2.566 ± 0.096** | 0.392 |
| 1024 | 2.949 ± 0.100 | 0.487 | 2.646 ± 0.097 | 0.412 |

### MagicDec (target weights, sparse mask at decode, cache-relative RoPE)

| L | τ (T=0.0) | acc. | τ (T=1.0) | acc. |
|---|---|---|---|---|
| 256 | 2.921 ± 0.102 | 0.480 | 3.307 ± 0.098 | 0.577 |
| **512** | 3.462 ± 0.101 | 0.615 | **3.763 ± 0.096** | 0.691 |
| 1024 | 3.586 ± 0.099 | 0.646 | 3.725 ± 0.095 | 0.681 |

### SPIRe draft, Phase A (pruned init + sparse training + distill + α; no target activations)

| L | window | τ (T=0.0) | acc. | τ (T=1.0) | acc. |
|---|---|---|---|---|---|
| 256 | 32 | 2.590 ± 0.098 | 0.397 | 2.536 ± 0.095 | 0.384 |
| **512** | **64** | **2.856 ± 0.100** | 0.464 | **2.851 ± 0.102** | 0.463 |
| 1024 | 128 | 1.458 ± 0.049 | 0.115 | 1.558 ± 0.057 | 0.139 |

### SPIRe draft, Phase B (Phase A + target-activation substitution + rollout split)

| L | window | τ (T=0.0) | acc. | τ (T=1.0) | acc. |
|---|---|---|---|---|---|
| 256 | 32 | 1.116 ± 0.021 | 0.029 | 1.214 ± 0.033 | 0.053 |
| **512** | **64** | **2.725 ± 0.100** | 0.431 | **2.769 ± 0.099** | 0.442 |
| 1024 | 128 | 1.576 ± 0.052 | 0.144 | 1.649 ± 0.059 | 0.162 |

Phase B vs Phase A at T=1.0: **−52% at L=256, −2.9% at L=512, +5.8% at L=1024.**
Target-activation substitution is credited with +0.393 τ in the paper; under our
reading it costs 0.082 at the matched-window point. See Finding 9.

### Against the paper (L=512)

| | paper | ours T=0.0 | ours T=1.0 |
|---|---|---|---|
| vanilla | 2.647 | 2.931 (**+10.7%**) | 2.566 (**−3.1%**) |
| magicdec | 3.891 | 3.462 (**−11.0%**) | 3.763 (**−3.3%**) |
| spire (full) | 3.401 | — | 2.851 (−16.2%) |
| **spire, no FM, no target acts** | **2.959** | 2.856 (−3.5%) | **2.851 (−3.6%)** |
| spire, no FM (Phase B target) | 3.352 | 2.725 (−18.7%) | 2.769 (−17.4%) |

Phase A implements neither feedback memory nor target-activation substitution, so
`spire_no_fm_no_target_acts = 2.959` is its correct comparison point — not full
SPIRe, and not `spire_no_feedback_memory`. Against it we are **3.6% low at T=1.0**,
matching the 3.1–3.3% shortfall of both baselines. See Finding 6.

---

## Findings

### 1. The paper's unstated sampling temperature is recoverable — it is T=1.0

The paper never records its sampling temperature. Measuring both settles it.

At **T=1.0** both baselines are low by nearly the same margin (−3.1%, −3.3%). At
**T=0.0** they err in *opposite directions* by similar magnitude (+10.7%, −11.0%).

A consistent small bias across two independent baselines is what our known
deviations predict (~10% data subset, unspecified hyperparameters, the −0.025
numerical floor). Equal-and-opposite errors are the signature of a wrong shared
setting. The gap structure agrees: the paper's MagicDec-over-vanilla margin is
1.244, ours is 1.197 at T=1.0 and only 0.531 at T=0.0.

**Conclusion: the paper sampled at T≈1.0, and at that temperature we reproduce both
baselines to within 3.3%.** This inference was only available because both
temperatures were measured — either alone yields one plausible number and no way to
know whether it is the right one.

### 2. Temperature interacts with draft type, and the direction reverses

MagicDec improves with sampling (3.462 → 3.763); vanilla degrades (2.931 → 2.566).

The mechanism is clean. MagicDec's draft **is** the target, so q ≈ p and Leviathan's
`min(1, p/q)` accepts readily at T=1; greedy instead demands exact argmax agreement,
which the sparse mask sometimes flips. The vanilla draft is a genuinely different,
smaller model: its argmax often matches on easy tokens, but its full distribution
does not, and sampling penalizes precisely that.

**This is a confound in the paper's headline comparison.** The MagicDec-over-vanilla
advantage is 0.53 greedy and 1.20 at T=1 — it more than doubles on an unstated
hyperparameter.

### 3. τ is not constant in L — the cost model assumes it is

MagicDec at T=1.0 measures 3.307 → 3.763 → 3.725 across L = 256/512/1024: a **14%
range**. `calculate_itm` takes the single L=512 measurement and reuses it unchanged
out to L=8192.

The dense (vanilla) draft keeps improving with context while the sparse (MagicDec)
draft plateaus — the MagicDec-minus-vanilla gap runs 0.958 → 1.197 → 1.079. That is
the predicted mechanism: MagicDec's window grows only as L/8, so the *absolute*
hidden context expands with L. It is exactly what feedback memory exists to
counteract.

**Caveat:** MagicDec's 512→1024 dip (3.763 vs 3.725) sits well inside the
confidence intervals. This is *consistent with* the hypothesis, not established by
it; it needs longer L to become a result.

### 4. A decode off-by-one invalidated all prior τ (found and fixed)

`train.py` feeds `inputs = shift_right(targets)` with position 0 masked to token 0.
`decode.py` fed the raw prompt, so `logits[-1]` predicted the token *just supplied*
and greedy generation repeated its own last token forever — while eval loss was a
healthy 2.534.

Every τ measured before 2026-08-31 is invalid: draft and target shared the bug, so
acceptance was measured on a degenerate repeating stream where it is trivially near
perfect. See fidelity-ledger §5.1 for why a fully green test suite missed it.

### 6. Target-activation substitution, not feedback memory, is the large missing piece

Decomposing Figure 5 (all at L=512, against full SPIRe's 3.401):

| removed | τ | cost |
|---|---|---|
| feedback memory (learned per-layer mixing) | 3.352 | **0.049** |
| ...and target-activation substitution | 2.959 | **0.393** |
| distillation (hard-target loss instead) | 3.069 | 0.332 |
| pruned init (random instead) | 3.105 | 0.296 |

The config lists `substitute_past_memory_with: target_activations` as a sub-option
of feedback memory, but the two are ablated *separately*. The reading that makes
both rows coherent: "no feedback memory" drops the learned per-layer mixing while
the draft still receives target activations; "no FM, no target acts" drops both.

**Target activations are therefore the single largest component in the paper —
0.393, ahead of distillation (0.332) and pruned init (0.296).** An earlier plan here
priced the whole feedback-memory bundle at 0.049 and deferred it as low-value. That
conflated the mechanism with the supervision it carries: the mechanism is worth
0.049, the supervision 0.393. Phase A skipped both, which is precisely why it lands
at 2.856 rather than 3.4.

Consequence for Phase B: build **target-activation substitution first**. It is also
the better-specified half — `m_t^i = y_t^{i+6-1}`, the target's layer-(i+6−1)
activations feeding draft layer i, a direct extension of the pruned-init
correspondence already implemented.

### 7. Sparse-draft τ is highly sensitive to train/eval window mismatch, asymmetrically

Phase A's draft trained at window 64. Evaluated at window = L/8:

| eval window | vs trained | τ (T=0.0) | acceptance |
|---|---|---|---|
| 32 (L=256) | narrower | 2.590 | 0.397 |
| **64 (L=512)** | **matched** | **2.856** | **0.464** |
| 128 (L=1024) | wider | **1.458** | **0.115** |

τ peaks exactly where the windows coincide. Narrowing costs 9%; **widening costs
49%** and drops acceptance to 0.115 — the draft becomes barely better than no
speculation at all. The asymmetry makes sense: a narrower window is a subset of the
attention pattern the draft learned, whereas a wider one places it fully
off-distribution.

This is not reported in the paper, and it has a direct methodological consequence:
**a sparse draft cannot be trained once and evaluated across a range of L** without
either retraining per L or accepting a large, L-dependent τ penalty that is easily
mistaken for a property of the method. It also retroactively justifies treating
fidelity-ledger §3.6 as blocking rather than a detail.

### 8. Distillation makes τ nearly temperature-invariant

| method (L=512) | τ(T=0) | τ(T=1) | Δ |
|---|---|---|---|
| vanilla | 2.931 | 2.566 | −0.365 |
| MagicDec | 3.462 | 3.763 | +0.301 |
| **SPIRe Phase A** | **2.856** | **2.851** | **−0.005** |

A distilled draft is trained to match the target's *full distribution*, so
Leviathan's `min(1, p/q)` behaves nearly the same whether tokens are drawn greedily
or sampled. An independently-trained draft agrees on argmax far more often than on
the distribution, so temperature moves it substantially.

This means the temperature confound in Finding 2 applies **asymmetrically across
methods**: the unstated temperature perturbs the two baselines by ±0.3–0.4 while
leaving SPIRe essentially unmoved. Any comparison of SPIRe against a baseline at an
unrecorded temperature inherits that asymmetry as a systematic bias.

### 9. Target-activation substitution is unusable without the rollout, and does not reproduce

Four training runs went into this component. The first three failed, each in a way
worth recording, and the fourth is methodologically sound and gives a small
**negative** result.

**Run 1 — information leak (τ = 1.000, acceptance 0.000).** The memory bank was
target layers [5, 8), which includes layer 7: the target's FINAL hidden state.
Pruned init also hands the draft a copy of the target's unembedding, so the draft
could route that state straight to its output and reproduce the target's
distribution without learning anything. It did exactly that, putting 0.381 and 0.134
on the leaked slot and ~0.0004 elsewhere. **Training loss went to 0.75 -- BELOW a
correct run** -- because a shortcut genuinely minimizes the training objective.

Root cause: every draft layer could read every memory slot. Fixed with a
**depth-causal mask** -- layer j may read only slots m ≤ j. Masked entries receive
zero gradient and stay at their zero init, so a trained `w_memory` with a non-zero
upper triangle now proves the mask was not in force.

**Run 2 — masked, no rollout (eval loss 12.88, τ ≈ 1.00).** The mask held (upper
triangle exactly 0) and the weights learned were legitimate: 0.418 on target layer
6 for draft layer 1, which is precisely the input that layer consumed inside the
full target. But trained with memory at every position and evaluated without it,
the model was *worse than uniform* (ln 50304 = 10.83). Not degraded -- non-functional.

**Run 3 — prefill-side decode memory (τ ≈ 1.00, unchanged).** Injecting the target's
activations during the draft's prefill does not help, because prefill memory
improves the KV cache while **τ depends entirely on per-token predictions during
drafting**, and every drafted position is in-flight by construction. Even the first
draft step cannot be helped: the newly committed token is one the target *produced*
rather than consumed, so no valid activation exists at that position.

**Run 4 — rollout split (τ = 2.769 at L=512).** Zeroing memory past a per-step
random split, mimicking the prompt/generation boundary, restores a working draft.
But it lands **below** Phase A (2.851), not at the paper's 3.352.

Two conclusions:

1. **`train_rollout_k` is load-bearing, not an optimization.** Without training-time
   exposure to the deployment memory condition, target-activation substitution
   yields a model that trains beautifully and cannot be deployed. The paper reports
   +0.393 τ for this component and specifies the rollout in one line without noting
   that the second is what makes the first work at all.

2. **Under our reading, the mechanism does not reproduce.** Most likely cause: τ is
   determined *entirely* by the memory-free regime, yet our split sampled uniformly
   over [L/4, L], so only ~37.5% of training positions exercised it -- against
   Phase A's 100%. Phase B has a better-informed prefix and a less-practiced
   generator, and τ scores only the generator. Biasing the split toward shorter
   prefixes is the obvious next experiment; we stopped instead, at four runs and
   ~10 GPU-hours, with Phase A as the better result.

This is evidence about *our reading* of an underspecified mechanism, not a
refutation of the paper. The spec pins neither the decode-time memory source nor
the split.

A third observation worth keeping: memory training made the draft markedly more
brittle to window mismatch. At L=256 Phase A lost 9% while Phase B lost 52%,
compounding Finding 7.

### 10. Measured ITM exceeds the cost model by ~2x, and the cause is not simple overhead

**First end-to-end timing of the paper's cost model** (objective #1). Vanilla
speculative decoding, H100 PCIe, HOI=756, k=4. ITM is measured as a SLOPE between
two generation lengths (G=32 vs 160, 7 reps), which cancels prefill and compile
cost exactly; cells whose inter-quartile jitter exceeds 10% are rejected rather
than reported.

| B | L | t_step (ms) | t_round (ms) | ITM meas | ITM pred | err |
|---|---|---|---|---|---|---|
| 4 | 512 | 1.414 | 9.468 | 6.696 | 1.667 | +302% |
| 16 | 512 | 2.752 | 12.499 | 4.542 | 1.833 | +148% |
| 64 | 512 | 8.620 | 42.997 | 4.988 | 1.944 | +157% |
| 4 | 1024 | 1.774 | 6.746 | 3.803 | 1.750 | +117% |
| 16 | 1024 | 4.114 | 15.378 | 3.738 | 1.900 | +97% |
| 64 | 1024 | 13.810 | 53.944 | 3.906 | 1.971 | +98% |

**The model under-predicts cost by 2-3x, in the direction of optimism**, and the
error *plateaus* near +98% rather than converging.

Absolute anchor: at B=64, L=512 the target step touches 604M elements = 1.21 GB in
bf16, which at 2 TB/s should take 0.6 ms. Measured 8.62 ms -- **14x off the memory
roofline the model assumes.**

**A fixed per-pass overhead does NOT explain it.** Solving
`t_step = c + w_t`, `t_round = 6c + 5·w_d + w_t` (a round is k=4 drafts + the
hole-closing pass + verify) per cell gives c ranging 0.81-6.34 ms, scaling with B,
and one cell yields a negative w_t. Hypothesis rejected.

What the data points to instead: per-pass costs that scale with **B x Klen but not
with model size** -- mask construction, RoPE table, cache indexing. A speculative
round pays them 6 times against a plain step's once, and they do not shrink when
the draft does, which would also explain why the draft never realizes its 4-6x
element-count advantage. **Untested**; profiling is the next step.

**Caveat, prominent:** this implementation is unoptimized -- mask-based attention
rather than a compact cache, no fused kernels, a 67M model. A production serving
stack sits far closer to roofline. The defensible claim is "the model omits costs
that dominate at these scales," not "the model is wrong."

Not measured: sparse drafts. SPIRe's and MagicDec's entire cost argument is that
the draft does not carry the KV cache, and ours is a mask, so the draft still reads
full Klen from HBM. Timing those would measure our implementation, not the method.
The compact ring buffer remains the blocker (fidelity-ledger 2.5).

### 5. The paper's dataset source no longer exists

`gs://longcrawl64` returns `NoSuchBucket` from every GCS endpoint. Work continues on
the `clankur/longcrawl64` mirror, whose document counts sum to exactly the published
6,661,465. See fidelity-ledger §2.1–2.3.

---

## Not yet done

- **Target-activation substitution** — built and measured across four runs; see
  Finding 9. Negative under our reading. A split-ratio sweep is the obvious next
  experiment if this is revisited.
- **Feedback memory** (learned per-layer mixing beyond the substitution) — worth
  0.049 τ, only ~1.6× our run-to-run jitter; resolving it at 95% confidence needs
  roughly 2,000 contexts rather than the 128 used here. Not attempted.
- **Compact sparse KV cache** — currently a mask, so draft memory traffic is
  unreduced and Phase 4 draft-side timing would be meaningless.
- **Phase 4** — throughput sweeps over batch/context/k vs `calculate_itm` at
  HOI = 756.
- **Phase 5** — long-context acceptance vs position.

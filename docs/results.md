# Results

Everything measured so far in the SPIRe reproduction. Companion to
[fidelity-ledger.md](fidelity-ledger.md), which records what these numbers can and
cannot be compared against.

As of 2026-09-08: **all three of the paper's draft models are trained and Table 1
is reproduced end to end** at the paper's own configuration -- fixed window 64,
sink 1, k=4, G=64 generated tokens, ~1,200 contexts per cell.

The "Phase A / Phase B" naming below is superseded. Phase A is the Figure 5
ablation *without feedback memory and without attending to target activations*
(published 2.959); Phase B implemented an operator the paper does not describe
(memory added into the residual stream at the same position, rather than
replacing the keys and values read from previous positions) and is withdrawn.
The model that carries every measurement from here is `spire_draft_spire_full`,
which has all six techniques Figure 5 ablates.

---

## Models trained

| | target | vanilla draft | SPIRe ablation | **SPIRe (full)** |
|---|---|---|---|---|
| config | `spire_target_1024` | `spire_draft_vanilla` | `spire_draft_spire` | `spire_draft_spire_full` |
| layers / d_model | 8 / 512 | 4 / 256 | 2 / 512 | 2 / 512 |
| d_head / d_ff | 128 / 4096 | 64 / 2048 | 128 / 4096 | 128 / 4096 |
| body params | 67,117,056 | 8,390,656 (= 1/8) | 16,779,264 (= 1/4) | 16,779,264 (= 1/4) |
| total params | 118,628,352 | 34,146,304 | 68,290,560 | 68,299,782 |
| training tokens | 2,372,567,040 | 682,926,080 | 1,365,811,200 | 1,365,811,200 |
| steps @ batch 64 × 1024 | 36,203 | 10,421 | 20,841 | 20,841 |
| attention at training | dense | dense | sink 1, window 64 | sink 1, window 64 |
| initialization | random | random | pruned: target [6:8) | pruned: target [6:8) |
| loss | hard CE | hard CE | 0.5·distill_CE + 0.5·(−α) | same |
| target activations as K/V | — | — | no | **yes** |
| feedback memory | — | — | no | **yes, n_mem 3** |
| **τ @ L=512, T=1.0** | — | **2.594** | **2.834** | **3.140** |
| paper's value for it | — | 2.647 | 2.959 | 3.401 |
| wall time (H100 PCIe) | ~5.5 h | ~45 min | ~2.5 h | **~7 h** |

The full model's wall time is 2.2× the ablation's at the same step count: feedback
memory runs k sequential forward passes per step, each rematerialised in the
backward. 0.833 steps/s against 1.83.

Its **eval loss is not comparable** and is omitted deliberately. Eval supplies no
teacher, so `memory=None` and the draft falls back to computing its own keys and
values everywhere — which is not its deployment condition, where the target's
activations are available for every committed position. τ at decode is the metric
that describes this model.

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

## τ: Table 1 reproduced (fixed window 64, sink 1, k=4, G=64 tokens)

The paper's own configuration, measured on ~1,200 validation contexts per cell
with the paper's stopping rule (score the rounds a context needs to emit 64
tokens, not a fixed round count). Raw `[contexts, rounds]` accept matrices are
saved under `~/tau_raw/`, so any statistic here can be recomputed without a GPU.

### T = 1.0 — the paper's regime

| L | Vanilla (ours) | paper | MagicDec (ours) | paper | SPIRe (ours) | paper |
|---|---|---|---|---|---|---|
| 960 | 2.580 ± 0.017 | 2.644 | 3.554 ± 0.021 | 3.793 | 3.106 ± 0.020 | 3.382 |
| **512** | **2.594 ± 0.018** | 2.647 | **3.648 ± 0.021** | 3.891 | **3.140 ± 0.020** | 3.401 |
| 256 | 2.590 ± 0.018 | 2.637 | 3.781 ± 0.021 | 4.005 | 3.139 ± 0.020 | 3.427 |

Gap to the paper: vanilla −1.8% to −2.4%, MagicDec −5.6% to −6.3%, SPIRe −7.7%
to −8.4%. The ordering MagicDec > SPIRe > vanilla holds at every context length,
as published.

### The dependence on L reproduces more closely than the level

| | spread across L, ours | spread across L, paper |
|---|---|---|
| vanilla | 0.5% | 0.4% |
| SPIRe | 1.1% | 1.3% |
| **MagicDec** | **6.4%, decreasing in L** | **5.6%, decreasing in L** |

MagicDec is the only method whose τ genuinely moves with context length, and it
moves in the same direction and by nearly the same amount as the paper's. It is
also the only method whose draft is the full target model, so it has the most to
lose from a sparse cache as the context it cannot see grows. This supersedes the
earlier claim that τ is not constant in L, which was measured while the window
was being varied along with L and is withdrawn.

### T = 0.0 — greedy does not discriminate

| L | Vanilla | MagicDec | SPIRe |
|---|---|---|---|
| 960 | 3.351 ± 0.034 | 3.410 ± 0.035 | 3.250 ± 0.034 |
| 512 | 3.333 ± 0.034 | 3.434 ± 0.035 | 3.224 ± 0.033 |
| 256 | 3.240 ± 0.033 | 3.688 ± 0.035 | 3.352 ± 0.034 |

The three methods compress into a narrow band and **the ordering inverts** --
vanilla beats SPIRe at L=512 and L=960. Greedy decoding makes all three drafts
agree with the target far more often, which flatters the weak draft most. Any
paper reporting τ without stating a temperature is under-determined by roughly
the size of the effect it claims.

---

## τ vs speculation depth, all three models (objective #2)

L=512, T=1.0, 600 contexts, G=64. The paper reports k=4 only.

| | k=1 | 2 | 3 | **4** | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| MagicDec | 1.853 | 2.560 | 3.156 | **3.688** | 4.095 | 4.490 | 4.788 | 5.119 |
| SPIRe | 1.761 | 2.346 | 2.776 | **3.136** | 3.407 | 3.594 | 3.744 | 3.858 |
| Vanilla | 1.659 | 2.092 | 2.382 | **2.581** | 2.708 | 2.798 | 2.859 | 2.951 |

**k=4 is the depth at which these three methods look most alike.** They saturate
at very different rates, and the paper evaluates all of them at the one depth
where that difference is least visible:

| | k=4 → k=8 | marginal gain at k=8 |
|---|---|---|
| MagicDec | +39% | **+0.331** |
| SPIRe | +23% | +0.114 |
| Vanilla | +14% | +0.092 |

At k=8 MagicDec is still gaining **3× faster** than either small draft and has
clearly not saturated. Its draft *is* the target, so acceptance survives deep into
a round; the small drafts are simply not reached that far.

The consequence for the paper's framing: **MagicDec's lead over SPIRe more than
doubles**, +0.552 at k=4 to +1.261 at k=8, while SPIRe's lead over vanilla grows
much less, +0.555 to +0.907. Read at k=4 the two gaps are nearly identical (0.552
vs 0.555); read at k=8 they are not close.

Efficiency, τ/(k+1) -- the fraction of each round's capacity actually used:

| | k=1 | k=4 | k=8 |
|---|---|---|---|
| MagicDec | 0.926 | 0.738 | 0.569 |
| SPIRe | 0.880 | 0.627 | 0.429 |
| Vanilla | 0.830 | 0.516 | 0.328 |

All three decay, but at k=8 vanilla wastes two thirds of every round while
MagicDec still uses over half. That ordering is stable across the whole sweep,
and it is what makes deep speculation worth more to a large draft -- the trade
the cost model has to price, since a bigger draft costs more per token drafted.

τ here is measured under the same G=64 rule as Table 1, so these numbers are
directly comparable to it: the k=4 column reproduces the Table 1 row to within
measurement noise.

## What the memory pathway is worth, measured

All four numbers below come from the same harness, so they are directly
comparable -- an earlier version of this comparison mixed a fixed-round
measurement with a token-budget one.

| model | ours | paper | gap |
|---|---|---|---|
| SPIRe ablation (no feedback memory, no target activations) | 2.834 ± 0.019 | 2.959 | −4.2% |
| SPIRe full (all six techniques) | 3.140 ± 0.020 | 3.401 | −7.7% |
| **the memory pathway is worth** | **+0.306** | **+0.442** | **69% captured** |

The pathway works and it is the largest single contributor we added, but it
delivers about two thirds of its published benefit. Three candidate explanations
were tested and eliminated:

1. **Measurement convention.** Re-running the ablation under the current harness
   gives 2.834 against the 2.851 recorded under the old one -- 0.6%, inside
   run-to-run jitter. Not the cause.
2. **Mixing-weight initialisation.** `w_memory` is zero-initialised, which under
   a softmax is a *uniform* mix rather than a no-op, so the draft might have
   started somewhere unhelpful and stayed. It did not: the trained weights are

   |  | embed | layer 0 out | layer 1 out |
   |---|---|---|---|
   | draft layer 0 | 0.264 | 0.634 | 0.102 |
   | draft layer 1 | 0.115 | 0.555 | 0.330 |

   against 0.333 uniform. Both layers concentrated on layer 0's output. Not the
   cause.
3. **A window off-by-one.** Feedback memory reads `m_<t` strictly, so a window of
   64 leaves 63 *readable* memory vectors; the paper may intend 64. Measured at
   both: SPIRe 3.127 → 3.131 (+0.13%, 0.14σ), MagicDec 3.688 → 3.717. SPIRe is
   indifferent and MagicDec — which has no feedback memory and does read its own
   slot — moved seven times more, the opposite of the prediction. Not the cause.

What remains, in order of plausibility:

- **Document diversity.** Ledger §2.3: the surviving mirror holds ~10% of
  LongCrawl64, and at the paper's `dataset_seqlen` of 1024 it yields 0.65B tokens
  against a 2.37B Chinchilla budget. We use 4096, which buys the tokens at the
  cost of ~4× fewer distinct documents. **This is not fixable by retraining** --
  the documents do not exist in the mirror. It also fits the shape of the gap:
  the dense draft is −2% while both sparse drafts are −4% to −8%, and a draft
  learning to exploit a narrow window plus target activations plausibly needs
  more diversity than one that simply reads everything.
- **Attribution between the two memory components.** Our +0.306 cannot be split
  into target-activations and feedback memory without the intermediate arm
  (published 3.352), which was deliberately not trained. That run is ~3.2h (no
  k-pass loop) and is the outstanding diagnostic.

Note that the τ shortfall does **not** propagate into the cost-model validation:
`ITM_measured = τ_obs / speedup` and `speedup = t_plain / (t_round/τ_obs)`, so τ
cancels and the measured ITM is a pure timing ratio.

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

### 3. ~~τ is not constant in L~~ — WITHDRAWN, and replaced by a real version

~~MagicDec at T=1.0 measures 3.307 → 3.763 → 3.725 across L = 256/512/1024: a 14%
range, and `calculate_itm` reuses the single L=512 value out to L=8192.~~

Withdrawn on two counts. The measurement varied the *window* as L/8 alongside L,
so it compared drafts with 32-, 64- and 128-token caches and attributed the result
to context length; and the paper's Table 1 already reports τ at three context
lengths, so this was never an untested assumption.

What survives, measured properly at fixed window 64, is narrower and stands up:
**MagicDec's τ does drift with L, and only MagicDec's.** Across L = 256/512/960 we
measure 3.781 → 3.648 → 3.554, a 6.4% range decreasing in L, against the paper's
4.005 → 3.891 → 3.793 (5.6%, same direction). Vanilla moves 0.5% and SPIRe 1.1%.
MagicDec is the one method whose draft is the full target model, so it has the
most to lose as the context its window cannot reach grows. The cost model's
assumption that τ is L-invariant is a good approximation for two of the three
methods and a ~6% one for MagicDec.

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

### 10. The cost model is optimistic by a flat ~25%, and its RATIOS reproduce exactly

**Objective #1, complete.** All three draft models timed end to end on an H100
PCIe at the paper's operating point: B=64, L=512, T=1.0, k=1..8, HOI=756, with
`Klen` pinned to 768 across every cell so the target's cache -- the denominator of
every ITM -- is identical for all three. Validation passes in all three modes:
ITM monotonic in k, worst jitter 1.7% against a 10% threshold, token ranges
matched to 3.8-8.0%.

#### Measured / predicted ITM

| | k=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | mean |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla | 1.33 | 1.30 | 1.31 | 1.30 | 1.28 | 1.27 | 1.26 | 1.25 | **1.29** |
| MagicDec | 1.32 | 1.29 | 1.26 | 1.25 | 1.23 | 1.21 | 1.18 | 1.17 | **1.24** |
| SPIRe | 1.15 | 1.17 | 1.19 | 1.23 | 1.23 | 1.26 | 1.25 | 1.27 | **1.22** |

Across all 24 cells the model is optimistic by **1.15x to 1.33x, mean 1.25x**.
That is a roughly constant factor, not a structural failure, and it supersedes
both withdrawn estimates (2-3x, and 30-40%).

#### Throughput, measured

`τ / ITM`, pairing measured ITM with the 600-context τ of §"τ vs speculation
depth" rather than the 64-context sample the timing batch provides:

| | k=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| SPIRe | 1.454 | 1.805 | 2.004 | **2.078** | **2.162** | 2.141 | 2.152 | 2.096 |
| MagicDec | 1.145 | 1.376 | 1.499 | **1.559** | 1.576 | 1.578 | **1.586** | 1.568 |
| Vanilla | 1.005 | **1.091** | 1.067 | **1.021** | 0.968 | 0.915 | 0.858 | 0.815 |

#### The paper's headline claims reproduce

| at k=4 | measured | paper Fig. 4 | measured/predicted |
|---|---|---|---|
| Vanilla | 1.021 | 1.36 | 0.75 |
| MagicDec | 1.559 | 2.05 | 0.76 |
| SPIRe | 2.078 | 2.78 | 0.75 |

Every absolute lands at **0.75-0.76** of prediction -- again strikingly uniform.
The paper's actual claims are *ratios between methods*, and a common factor
cancels out of a ratio:

| | measured | paper |
|---|---|---|
| SPIRe over vanilla | **+104%** | +104% |
| SPIRe over MagicDec | **+33%** | +36% |

The paper claims "over 100% versus vanilla, over 35% versus MagicDec". On
hardware, at its own operating point: **+104% and +33%**. At each method's own
best depth, +98% and +36%.

**This is the central result.** The cost model mispredicts every absolute
throughput by a quarter, and predicts the differences between methods almost
exactly. A roofline model that omits a constant overhead fraction will do
precisely that, and it means the paper's comparative conclusions survive
measurement even though its absolute numbers do not.

#### Two things only measurement shows

**Vanilla speculative decoding is a net loss in this regime.** 1.021x at k=4 --
barely break-even -- falling below 1.0 from k=5 and reaching 0.815x at k=8. The
cost model predicts 1.36x. At B=64, L=512 the dense draft's KV reads cost more
than its acceptance buys, and no analytical term in the model captures it.

**k=4 is not the optimum for any of the three.** SPIRe peaks at k=5 (2.162x),
MagicDec at k=7 (1.586x), vanilla at k=2 (1.091x). The paper evaluates
everything at k=4. Combined with the acceptance sweep, deeper speculation is
worth more to the larger draft on both sides of the ratio.

#### On the validation gate

All three modes print `DISAGREE`, on the τ cross-check alone. It is a sampling
artifact and it is understood: the timing harness measures τ on one batch of 64
contexts while the reference averages 600. Re-running the acceptance harness on
that *same batch* gives 2.322 / 3.456 / 2.846 against the timing run's 2.414 /
3.546 / 2.960 -- agreement to 2.6-4.0%, the residual being the fixed-round versus
G=64 stopping rule. That batch is simply harder than the split average, uniformly
across all three methods.

It does not touch the ITM column: `ITM = τ_obs / speedup` while
`speedup = t_plain / (t_round/τ_obs)`, so **τ cancels** and measured ITM is a pure
timing ratio.

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

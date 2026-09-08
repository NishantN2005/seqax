# Fidelity ledger

What this reproduction of SPIRe (Neelam et al., MatX, arXiv:2504.06419) matches,
where it knowingly departs, and what the paper leaves undefined. Every number we
publish should be traceable to a row here.

Last updated 2026-09-05, after reading the paper PDF and MatX's own appendix
notebook (`spire_appendix.ipynb`, fetched from the `SPIRe` branch of
MatX-inc/seqax). Between them they resolved four questions this ledger had
listed as unspecified, and **overturned two decisions recorded below**. The
superseded text is kept, struck through, because knowing which way we were
wrong is part of the record.

---

## 1. Faithful to the paper

| Item | Detail |
|---|---|
| Verification | Token-level Leviathan rejection sampling: accept `d_i` w.p. `min(1, p_i/q_i)`; on first rejection sample from `norm(max(0, p_j - q_j))`. Not block verification. |
| Bonus token | Zero-padded `q` at index k+1 makes the all-accept bonus the same code path as residual sampling. |
| τ accounting | `τ = E[#accepted] + 1`, one sample per round. |
| RoPE convention | SPIRe uses original-text positions; MagicDec uses positions-within-cache (paper footnote 4). Both implemented and separately selectable. |
| MagicDec baseline | Target weights reused, dense prefill, sink+window restriction applied at decode only. |
| Target architecture | 8 layers, `d_model` 512, MHA (`n_q_per_kv` 1, `n_kv` 8), `d_head` 128, `d_ff` 4096. Body = 67,117,056 params, reproducing the count printed in `spire_appendix.ipynb` exactly. |
| Vanilla draft architecture | 4 layers, `d_model` 256, `d_head` 64, `d_ff` 2048. Body = 8,390,656 = exactly 1/8 of the target, as specified. |
| Token budget | Chinchilla ~20 tokens/param on total (body + 2·vocab·d_model). |
| Speculation config | k=4, sink=1. |
| Window, training and τ | **Fixed at 64 for every context length.** Paper 4: "a window size of 64 and a sink size of 1" for both MagicDec and SPIRe, during training and inference. |
| Window, cost model | **`sink + L/8`, i.e. proportional to L.** `spire_appendix.ipynb`: `Klen = sink_size + L // window_factor`, `window_factor = 8`. These two rows disagree everywhere except L=512, where `512/8 = 64`. See §7.2. |
| SPIRe draft architecture | 2 layers, `d_model` 512, `d_head` 128, `d_ff` 4096. Body = 16,779,264 = exactly 1/4 of the target. Matches the notebook's `spire_draft` dict and paper 3.2 ("the last two transformer blocks"). |

---

## 2. Known deviations

### 2.1 Dataset source — the paper's source no longer exists

`gs://longcrawl64`, named in both the seqax README and Manifest AI's article, is
gone: the GCS JSON API, XML API, and virtual-host URL all report the bucket does
not exist. We use the HuggingFace mirror `clankur/longcrawl64`.

Provenance evidence, since byte-identity against a deleted bucket is unverifiable:
its train (6,609,334) and heldout (52,131) document counts sum to **exactly** the
6,661,465 Manifest AI publishes; documents are 65,536 tokens as documented; sampled
documents decode to coherent multilingual text under GPT-2's tokenizer.

`spire_backup/dataset_manifest.md5` records checksums of all 1,061 fetched chunks
so a future rebuild can be proven identical — the mirror is third-party and can
change under us.

### 2.2 Dataset is a ~10% subset with reindexed rows

The mirror holds ~313 of the source's 3,228 train row-chunks, and their indices are
non-contiguous (scattered over 0..1279). `tools/fetch_longcrawl64_subset.py` keeps
rows having every needed column-chunk and reindexes them densely. Chunk files are
independent and training shuffles documents, so this changes only which row a
document occupies, never its contents.

### 2.3 `dataset_seqlen` 4096, not the paper's 1024

seqax keeps only the first `dataset_seqlen` tokens of each document
(`input_loader.py:248`). At 1024 the mirror's 632,832 documents yield **0.65B**
tokens — far short of the 2.37B budget. At 4096 they yield **2.59B**.

Consequence: the same token count drawn from **~4× fewer distinct documents**, each
contributing four consecutive 1024-token training sequences instead of one. For a
118M model at Chinchilla this is a modest diversity reduction, but it is a real
difference from the paper's data distribution.

### 2.4 Context lengths {256, 512, 960} — RESOLVED, no longer a deviation

~~`tau.py` requires `context_len` to divide `dataset_seqlen`, and 960 ∤ 4096. We
substitute 1024.~~

Rebuilding the dataset at `dataset_seqlen = 7680` removed this: 7680 = 30·256 =
15·512 = 8·960, so all three of the paper's Table 1 context lengths divide it.
We now measure at the paper's exact L values and compare row by row.

### 2.5 ~~Sparse cache is a mask, not a compact ring buffer~~ — RESOLVED 2026-09-07

~~Distributions and τ are exact, but the draft's memory traffic is unreduced.
Draft-side timing is invalid until a real compact cache exists.~~

Built. `compact_draft_cache=True` gives the draft a physically compact cache of
`sink + window + k` entries with a ring over the window, so its memory traffic no
longer grows with L. `tests/test_ring_buffer.py` proves the compact and
full-masked paths produce **bitwise identical** draft distributions
(max |Δq| = 0.0) across every round, for SPIRe and for MagicDec under both RoPE
conventions. See §10 for what building it turned up.

### 2.6 Hardware: H100 PCIe, not SXM

The notebook's `HOI = 600` is H100 **SXM** (1979 TFLOPS FP8 ÷ 3.35 TB/s). Ours is
PCIe: 1513 ÷ 2.0 = **756**. Two mitigating facts: HOI is dtype-invariant on H100
(FP8 peak is exactly 2× bf16, so the doubled bytes cancel — bf16 gives 756 too), and
HOI **cancels exactly** from the ITM ratio wherever all three `f = max` terms are
memory-bound. At k=4 the constants diverge only for L below ~68 (SXM) / ~54 (PCIe),
so every grid cell at L ≥ 128 predicts identically on both cards.

---

## 3. Unspecified by the paper — our choices

### 3.1 Sampling temperature — measure BOTH, always

**Decision (2026-08-31): every τ is reported at T=0.0 and T=1.0 as a pair.**

The paper never states its sampling temperature, and the choice is not small. At
L=512, MagicDec τ moves from **3.462 (T=0) to 3.763 (T=1)** — a swing of 0.30,
where the paper's entire SPIRe-vs-MagicDec claim is 0.49 (3.401 vs 3.891). An
unstated hyperparameter therefore accounts for roughly 60% of the effect being
claimed. Picking one temperature would make any gap unattributable.

### 3.2 Batch size 64 — forced, not preferred

The unembedding produces `[B, 1024, 50304]` f32 logits: **206 MB per sequence**,
and the backward needs several copies. Batch 128 requests 105 GB and OOMs at *any*
`XLA_PYTHON_CLIENT_MEM_FRACTION`. Batch 64 also measured **faster per token** than
96 (131K vs 115K tok/s). Note this ceiling is set by the vocabulary, not `d_model`,
so it binds the drafts identically to the target.

### 3.3 Learning rate 3e-3 — empirical

A 1,200-step probe with a full scaled-down schedule gave final loss 3.609 / 3.581 /
3.649 at 6.5e-3 / 3e-3 / 1.5e-3. The band is **flat to within 2% with no divergence
anywhere** — gradient clipping to global norm 1.0 is doing the stabilizing. 3e-3
won narrowly. Caveat: a 1,200-step proxy need not rank LRs as a 36,203-step run
would.

### 3.4 Schedule

10% linear warmup, cosine to `final_fraction` 0.01, matching seqax's own
`longcrawl_*` configs (1907/19070). Paper silent.

### 3.5 Window semantics

Our window of 64 **includes** the current token. The paper does not say.

### 3.6 SPIRe draft window — OVERTURNED 2026-09-05: fixed at 64

~~**Decided 2026-09-01: the window scales as L/8**, because `calculate_itm`
computes `kv_draft` with `window = L/8` at every L, so a fixed window would be
measuring a configuration the cost model does not describe.~~

**This was wrong, and it was our reconstruction rather than the paper's text.**
Paper 4 states plainly that both MagicDec and SPIRe use "a window size of 64 and a
sink size of 1", and 3.1 says the mask exists precisely "to ensure the draft
model's KV cache is sparse and that the size of the KV cache is constant with
respect to the decoding sequence length". A window of L/8 is not constant in L.

The L/8 rule is real, but it belongs to the *cost model only* — it is how the
notebook parameterizes `Klen` when sweeping L, anchored so that L=512 gives the
true window of 64. The paper is internally inconsistent on this point; §7.2
records what that costs.

What it invalidated: every τ we measured at L=256 and L=1024 under the old
decision varied the window and the context length together, then attributed the
result to context length alone. At L=512 the two conventions coincide, which is
exactly why that column reproduced and the others did not. Those runs are
withdrawn; §4.3 below is the finding that was built on them.

---

## 4. Measurement artifacts invisible to the cost model

### 4.1 Numerical agreement floor: 0.014–0.025 tokens/round (a range, not a constant)

Greedy self-drafting (draft ≡ target) must give τ = k+1 = 5.0 exactly. It does not:
the draft evaluates incrementally (L=1) while verification evaluates k+1 tokens
batched; in bf16 these reduce in different orders and near-tied argmaxes disagree.
The shortfall is a systematic **downward** bias on every measured τ, it is the
ceiling no draft can exceed, and the analytical model cannot see it.

**It is also not reproducible run to run.** Two measurements on identical config,
identical data, and the same checkpoint:

| run | τ | floor | % of k+1 |
|---|---|---|---|
| 2026-08-31 (before teardown) | 4.975 ± 0.023 | 0.0254 | 0.51% |
| 2026-09-01 (after restore) | 4.986 ± 0.017 | 0.0137 | 0.27% |

A 1.9× spread. The cause is `init_seqax.set_variables()`, which sets
`--xla_gpu_deterministic_ops=false` on GPU, so reductions are not bit-reproducible.

Consequences: **report the floor as a range from repeated measurement, never as one
number**, and treat every τ as carrying this jitter — the same restore reproduced
vanilla τ as 2.541 ± 0.095 against a recorded 2.566 ± 0.096. Differences below
~0.03 between runs are not evidence of anything. If bit-reproducibility is ever
needed for a specific claim, flip that flag and re-measure.

### 4.2 The draft runs k+1 forward passes per round, not k

When all k drafts are accepted, `d_k` is never fed to the draft, so its KV is
missing from the draft cache. The implementation re-feeds `prev_tok` at `pos-1`
each round to close the hole. τ is unaffected (identical values or a required
hole-fill), but **the cost model's `k · t_draft` term understates real draft cost**.

### 4.3 ~~τ is not constant in L~~ — WITHDRAWN 2026-09-05

~~MagicDec at T=1.0 measures τ = 3.307 / 3.763 / 3.725 at L = 256 / 512 / 1024: a
**14% range**. `calculate_itm` reuses the L=512 measurement unchanged out to
L=8192. This is the first direct evidence against that assumption.~~

Withdrawn for two independent reasons.

1. **The measurement varied two things at once.** Under the since-overturned §3.6
   the window moved as L/8, so those three numbers compare drafts with 32-, 64-
   and 128-token caches. The 14% spread is a *window* sweep mislabelled as a
   context-length sweep. It is not evidence about L.

2. **The paper already did this experiment.** Table 1 reports τ at
   L ∈ {256, 512, 960} for all three drafts, and states the assumption it is
   testing. Vanilla moves 2.637 → 2.647 → 2.644 (0.4%); SPIRe 3.427 → 3.401 →
   3.382 (1.3%); MagicDec 4.005 → 3.891 → 3.793 (5.3%, monotone decreasing).
   Characterizing this as an untested assumption was simply wrong.

What survives is narrower and still worth reporting: MagicDec's τ *does* drift
with L in the paper's own numbers, and it is the one method whose draft is the
full target model, so it has the most to lose from a sparse cache as the context
it cannot see grows. Our re-measurement at fixed window 64 tests whether that
5.3% drift reproduces.

---

## 5. Defects found and fixed

### 5.1 Missing BOS/right-shift in the decode stack (fixed 2026-08-31)

`train.py:50-58` feeds `inputs = shift_right(targets)` with position 0 masked to
token 0, scoring `logits[i]` against `targets[i]`. `decode.py` fed the raw prompt,
so `logits[-1]` predicted the token *just supplied*. Greedy generation repeated its
own last token forever (`France France France…`) while eval loss was a healthy
2.534.

**All τ measured before this date are invalid** — draft and target shared the bug,
so acceptance was measured on a degenerate repeating stream where it is trivially
near-perfect, inflating τ.

Why the suite missed it: `test_dense` compares cached vs non-cached, `test_spec`
compares draft vs target, `decode.py --check` compares generation against its own
reference. **Every test validated the implementation against itself**, and all pass
under any constant shift. `tests/test_convention.py` now rebuilds the reference from
`train.py`'s rule independently, and asserts the unshifted variant *differs* so it
cannot pass vacuously.

### 5.2 Ragged final chunk fabricated 1,117 all-zero validation documents

Sizing the fetched array as `n_rows × 2048` ignored that a split's final row-chunk
is ragged (heldout's holds 931 documents, not 2048). Zarr padded the difference with
`fill_value: 0`, and `remap_zero_token` faithfully converted those into runs of
token 50256 — **~2% of the eval set**, silently. Caught by `verify_flat_tokens.py`
(32/1500 sampled validation documents degenerate, 0 in train). After the fix,
validation remaps at 0.0633%, identical to train.

### 5.3 Repo defects fixed before any run

`tau.py` and `tests/test_spec.py` imported `speculative` while the file was named
`spec_decoding.py` (both dead on a fresh clone); `train.py` still asserted on
removed `nsa_*` config keys; `longcrawl_184m/1176m.yaml` still carried `nsa_*` keys
that `ModelConfig` no longer accepts.

---

## 6. Results so far

Target: 8 layers / 67.1M body / 118.6M total, 2.37B tokens, final eval loss
**2.534** (ppl ≈ 12.6), 36,203 steps.

MagicDec τ (k=4, sink=1, window=L/8, 128 contexts × 8 rounds):

| L | window | τ (T=0.0) | τ (T=1.0) | paper |
|---|---|---|---|---|
| 256 | 32 | 2.921 ± 0.102 | 3.307 ± 0.098 | — |
| 512 | 64 | 3.462 ± 0.101 | **3.763 ± 0.096** | **3.891** |
| 1024 | 128 | 3.586 ± 0.099 | 3.725 ± 0.095 | — |

At T=1.0 we reproduce MagicDec to within **3.3%** (2.6σ); at T=0.0 the shortfall is
11%. Which of those is "the" result depends entirely on §3.1.


---

## 7. What MatX's appendix notebook settles (added 2026-09-05)

The paper's Appendix is one line: a link to `spire_appendix.ipynb` on the `SPIRe`
branch of `MatX-inc/seqax`. The branch no longer appears in the GitHub branch
listing, but the raw file still resolves. It is the authoritative source for
several things the PDF leaves out, and it is what a reproduction should be checked
against.

### 7.1 The exact architectures

```python
target           = {"num_layers": 8, "d_model": 512, "n_q_per_kv": 1, "n_kv": 8, "d_head": 128, "d_ff": 4096}
vanilla_sd_draft = {"num_layers": 4, "d_model": 256, "n_q_per_kv": 1, "n_kv": 8, "d_head": 64,  "d_ff": 2048}
spire_draft      = {"num_layers": 2, "d_model": 512, "n_q_per_kv": 1, "n_kv": 8, "d_head": 128, "d_ff": 4096}
```

All three reproduce in our configs to the parameter: 67,117,056 / 8,390,656 /
16,779,264. Note the vanilla draft is narrower as well as shallower (`d_model`
256), while the SPIRe draft keeps the target's full width and only drops depth —
which is what makes the target-activation substitution of §7.3 dimensionally
possible at all.

### 7.2 The paper contradicts itself on whether the draft's cache grows with L

- **3.1 (prose):** the StreamingLLM mask ensures "the size of the KV cache is
  constant with respect to the decoding sequence length."
- **4 (prose):** "a window size of 64 and a sink size of 1."
- **Appendix (code):** `Klen = sink_size + L // window_factor` with
  `window_factor = 8`, justified as "so that the size of the sliding window is
  some fixed fraction of the context length L."

The last is not constant in L, and it is what actually produces every throughput
number in Figures 1, 4, 5 and 6. The three agree only at L=512.

This matters more than a footnote, because the figures hold **τ fixed at its L=512
value** while letting the cost model's window grow with L. So along the x-axis of
Figure 4, the draft's cache size changes but its acceptance rate is assumed not
to — the cost side and the quality side of the same sweep describe different
models. Under the prose reading (window pinned at 64) the draft's cache stops
growing entirely, `speculate_cost / target_cost → 0`, and `ITM → 1.0` rather than
the 1.125 the notebook's version approaches. That is a materially more optimistic
asymptote than the paper claims for itself, and it is reachable with the
architecture the paper actually describes.

### 7.3 SPIRe's memory pathway — what it is, and what we built instead

Paper 3.3, made concrete. The draft has 2 layers. For a prefix of length S with
speculation depth k, the memory vector feeding layer i at position t is

- `m^i_t = y^{i+5}_t` for `t = 1 … S−k` — **the target model's activations**
  (`m¹ = y⁶`, `m² = y⁷`, 0-indexed), and
- `m^i_t = Σ_ℓ exp(w_iℓ)·x^ℓ_t / Σ_ℓ exp(w_iℓ)` for the last k positions — the
  draft's own softmax-weighted mix over all its layer activations, layer 0 being
  the embedding.

Attention is `z^i_t = Attention(x^i_t, m_{<t})`: the memory vectors **replace the
keys and values**, and are read from **strictly previous** positions. Queries
still come from the draft's own hidden state.

This is not free-floating recurrence — it is why the draft matches the target's
width, and it is cheap at decode time because the target has *already computed*
`y⁶` and `y⁷` for every committed position during verification. The notebook
prices exactly this, and nothing else, as SPIRe's extra draft cost:

```python
N_draft_kv_params = 2 * draft["num_layers"] * draft["n_kv"] * draft["d_model"] * draft["d_head"]
FLOPs_draft += 2 * N_draft_kv_params * B   # project 1 memory vector into a key and a value
```

**What we built instead** was additive same-position injection: a memory bank
added into the residual stream at position t, rather than a replacement of the
keys and values read from positions before t. That is a different operator. All
three of our feedback-memory runs measured something the paper does not describe,
and they are withdrawn.

### 7.4 The ablation τ values, which appear nowhere in the PDF

Figure 5 is a plot; its underlying τ values are literals in notebook cell 21:

| SPIRe variant | τ | Δ vs full |
|---|---|---|
| SPIRe (full) | **3.401** | — |
| ω = 1 (pure distillation, no MixedLoss) | 3.375 | −0.026 |
| without feedback memory | 3.352 | −0.049 |
| initialize randomly | 3.105 | −0.296 |
| HardTargetLoss (no distillation) | 3.069 | −0.332 |
| **without feedback memory *and* without attending to target activations** | **2.959** | **−0.442** |

Two things follow immediately.

**First, our SPIRe draft is the bottom row.** It has the sparse cache, pruned
init, and MixedLoss(0.5) distillation, and it has neither memory component. The
paper's value for that exact configuration is **2.959**, not the 3.401 of
Table 1. Comparing our draft to 3.401 would have been comparing it to a model we
had not built.

**Second, the two memory components are worth wildly different amounts:**

- attending to target activations: **+0.393 τ** (2.959 → 3.352)
- feedback memory on top of that: **+0.049 τ** (3.352 → 3.401)

Attending to target activations is worth **8× more** than feedback memory. This
is consistent with the Figure 5 caption, which ranks the six techniques "in order
of importance" and places *attending to target model activations* 2nd and
*feedback memory* 5th of 6.

It also closes a question this project spent real time on. Feedback memory buys
**1.5%** of τ in the paper's own measurements. Our inability to detect an effect
from it was never a reproduction failure — there is close to nothing there to
detect, and the "Re" in SPIRe is carrying far less weight than the name implies.

### 7.5 HOI = 600 is SXM, and the notebook says so

"dividing the FP8 Tensor Core performance for the H100 SXM by 2 to get the FP8
performance without sparsity, and then dividing that by the GPU Memory
Bandwidth." Confirms §2.6: our PCIe card gives 756, and the difference cancels
from the ITM ratio wherever all three roofline terms are memory-bound.

---

## 8. Defects found 2026-09-05

### 8.1 Batch size silently constrained by dataset chunking

`ShufflingLoader` reads whole dataset rows and splits each into
`dataset_seqlen / context_len` sequences, then calls
`_div_exact(batch, sequences_per_row)` — an assertion, with no message. At
`dataset_seqlen = 7680` a batch must be a multiple of 30 at L=256, 15 at L=512
and 8 at L=960.

This took out two separate jobs at once. The first fixed-window τ sweep ran
`--batch 32` at all three lengths: 32 % 8 = 0, so **L=960 ran and L=256 and L=512
died**, leaving a log that looked like a model bug at short contexts. The cost
sweep asked for B=64 at L=512 because the cost model is stated at B=64 — and
64 % 15 ≠ 0, so it died before its first timing row.

Fixed in `dataset_prompts` by loading the next valid multiple up and slicing back
down: the batch size is a property of the experiment, not of how the corpus was
chunked.

### 8.2 τ depends on the stopping rule by ~3%

The paper measures τ "by generating G = 64 tokens", not over a fixed number of
rounds. The two are not the same statistic. Under a fixed round budget a
high-τ method emits more text than a low-τ one, so the methods get averaged over
different amounts of generation; under a token budget, slow contexts contribute
more rounds than fast ones.

Measured on the same 240 contexts, vanilla at L=512, T=1.0:

| accounting | τ |
|---|---|
| mean over all 70 rounds | 2.612 |
| paper's rule: rounds needed to emit G=64 | 2.527 |

A 3.3% difference — comparable to the entire gap between our reproduction and the
paper. `tau.py --gen-tokens 64` implements the paper's rule; the raw
`[contexts, rounds]` accept matrices are saved so either statistic can be
recomputed without re-running the GPU. A probe put the slowest context at 43
rounds, so 48 rounds covers 100% of them.

---

## 9. Scope: one SPIRe, three models (decided 2026-09-05)

Every measurement from here uses exactly three draft configurations, matching the
three the paper compares:

| paper's name | our config | notes |
|---|---|---|
| Vanilla SD | `spire_draft_vanilla` | 4 layers, `d_model` 256; trained, final |
| MagicDec | *(none)* | reuses the target's weights with a sparse mask at decode |
| SPIRe | `spire_draft_spire_full` | all six Figure 5 techniques |

`spire_draft_spire` is retained but **is not used for any throughput, ITM, cost or
k-sweep measurement**. It is the Figure 5 ablation "without feedback memory and
without attending to target activations" (published 2.959, measured 2.851), kept
because it is trained and is a published reference point we can report.

### Why no intermediate

An earlier plan trained the 3.352 arm ("without feedback memory") as well, to give
the paper's ablation ladder end to end. Dropped, for three reasons.

1. **It is a diagnostic, not a deliverable.** None of the three objectives —
   validating the performance model, k-sensitivity, long-context evaluation —
   needs it. It would only tell us *which* memory component was at fault if the
   full model missed 3.401, and there is no reason to buy that answer before
   there is a question.
2. **We already hold a lower rung.** `spire_draft_spire` is measured against a
   published 2.959. If the full model lands near 3.401, the ladder was never
   needed; if it misses badly, the intermediate can be trained *then*, as a
   diagnostic, at the same cost as training it now.
3. **A shelf of half-SPIRes makes every table ambiguous.** Once three variants
   exist, every downstream number invites the question of which one produced it,
   and the ones that answer "an ablation" are not results about SPIRe.

Cost of the decision: ~3.2 GPU-hours saved, one fork removed from the write-up.
Risk accepted: if `spire_draft_spire_full` misses 3.401, attribution between the
two memory components needs an extra run.


---

## 10. The compact sliding-window cache (2026-09-07)

Two findings came out of building it, one of which is a correction to the paper.

### 10.1 The ring must be `sink + window + k`, not `sink + window`

The appendix charges the draft for `Klen = sink_size + L // window_factor` — the
window it *reads*. That is the right size for ordinary autoregressive decoding and
the wrong size for speculative decoding, because a draft writes **k tokens ahead
speculatively** and some of those get rejected.

Concretely, with a ring of exactly `window`: after drafting out to `pos+k` the
ring has evicted everything older than `pos+k-window`. If only `n_acc` of the k
proposals are accepted, the next round's query sits at `pos+n_acc+1` and still
needs positions back to `pos+n_acc+1-window`. At `n_acc = 0` those are precisely
the entries the *rejected* writes destroyed. The k extra slots absorb speculation
that gets thrown away; they are retained but never visible, since the mask still
uses the true window, so **τ is completely unaffected**.

This is not a theoretical worry — `tests/test_ring_buffer.py` runs it both ways.
At `sink + window` the draft distributions diverge from the full-cache reference
by 1.1e-01; at `sink + window + k` they agree to 0.0.

Size of the correction to `KV_draft`: at the paper's window 64, sink 1 and k=4,
69 entries rather than 65 — the cost model understates the draft's cache by
**6.2%**. It does not move any conclusion (the draft's cache is still constant in
L, which is the claim that matters), but a reproduction should use the honest
number, and a paper that reports throughput to three significant figures should
have it.

### 10.2 A RoPE table sized by cache length, silently wrong only at long context

`forward_pass` built its rotary table as `RopeTable.create(Klen, h)` where
`Klen = attention_mask.shape[2]` — the cache length. On a dense cache the slot
index and the absolute position are the same number, so this is correct by
coincidence. A compact cache breaks that identity: a 65-slot ring carries absolute
positions in the thousands, and every position past the end of the cache was
silently clamped to the last table entry. Not an exception, not a NaN — just wrong
rotations.

The failure mode is worth noting because of *where* it hides: it is invisible at
short context (positions still fit inside the cache) and appears only once the
context outgrows the window, which is exactly the regime SPIRe exists to serve.
Callers now pass `rope_table_len` explicitly.

Both bugs were found by the same test, and neither was findable from τ alone: the
first changes acceptance only in later rounds, and the second only at long
context. The test compares draft **distributions** rather than sampled tokens,
because an untrained draft's logits are near-uniform and its argmax is decided by
float noise — comparing tokens would have passed a broken ring.

### 10.3 The timing harness over-allocated the cache, against the control

`bench_k.py` pinned `Klen` at `L + 1 + round(rounds * 1.25) * (k + 1) + k + 1`.
The `(k + 1)` is the **maximum** tokens a round can yield — every round accepting
every draft. That does not happen: measured τ runs 1.764 to 4.092 against a `k+1`
of up to 9. The pin came out at 702 where about 586 is ever occupied.

Since attention reads the whole allocation rather than the occupied part, the
~116 surplus slots were charged to every timed cell. They were not charged
evenly, and the direction is what makes it worth fixing:

| draft | reads per step | pays for the surplus? |
|---|---|---|
| vanilla (dense) | all of `Klen` | **yes** |
| MagicDec (windowed) | `sink + window + k` | no |
| SPIRe (windowed) | `sink + window + k` | no |

So the slack fell entirely on **vanilla, which is the control**, while the cost
model charges vanilla for `B · L`. The measured baseline was inflated and the
methods under test were not — a bias that flatters SPIRe.

Sized from calibrated τ instead (`rounds * τ * 1.25`), the pin is 603: 14.1%
smaller, ~3% headroom rather than ~20%. Also added an overrun check, because an
out-of-bounds cache write does not raise — `dynamic_update_slice` clamps — so a
run that outgrew a tighter buffer would corrupt the cache silently and still
produce a plausible timing. It tests the longest row in the batch, not the mean.

Note this does NOT invalidate the k=1..7 ITM divergence already measured: `Klen`
was pinned to one value across every depth, so it cannot produce a trend in k.
It shifts the absolute ITM levels, and it matters for the three-way comparison
that has not been run yet.

### 10.4 SPIRe's memory-projection FLOPs are inert at every cell the paper plots

The appendix charges SPIRe's draft one extra term beyond an ordinary forward pass:

```python
N_draft_kv_params = 2 * layers * n_kv * d_model * d_head
FLOPs_draft += 2 * N_draft_kv_params * B   # memory vector -> a key and a value
```

It is correctly included, and it changes nothing the paper reports. At B=64,
L=512 the SPIRe draft is memory-bound by **8.4×** — 2.03e10 against 2.42e9 — so
`f = max(FLOPs, (N + KV)·HOI)` is decided by the cache and the weights, and the
FLOPs term is invisible. Dropping it entirely leaves every ITM in Figures 1, 4, 5
and 6 bit-identical. It becomes visible only in the compute-bound corner (B≥512
with contexts of a few tens of tokens), which is outside the plotted region.

Worth stating because of what it prices: this is the cost of the feedback-memory
mechanism, the component Figure 5 already ranks 5th of 6 at +0.049 τ. So feedback
memory is a component whose benefit is 1.5% and whose modelled cost is exactly
zero across the paper's entire operating region.

`tests/test_cost_model.py` asserts both halves — that the term is inert at the
headline cell, and that it still bites when compute-bound, so the first assertion
cannot pass by the term simply being absent.

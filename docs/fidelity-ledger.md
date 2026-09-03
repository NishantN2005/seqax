# Fidelity ledger

What this reproduction of SPIRe (Neelam et al., MatX, arXiv:2504.06419) matches,
where it knowingly departs, and what the paper leaves undefined. Every number we
publish should be traceable to a row here.

Last updated 2026-08-31, after the 67M target and the vanilla draft.

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
| Speculation config | k=4, sink=1, window = L/8. |

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

### 2.4 Context lengths {256, 512, 1024}, not {256, 512, 960}

`tau.py` requires `context_len` to divide `dataset_seqlen`, and 960 ∤ 4096. We
substitute 1024, which is also the training sequence length.

### 2.5 Sparse cache is a mask, not a compact ring buffer

Distributions and τ are exact, but the draft's memory traffic is unreduced. **Phase
4 draft-side timing is invalid until a real compact cache exists** — the whole
point of SPIRe's draft is the cache it does not carry.

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

### 3.6 SPIRe draft training window — RESOLVED: scale as L/8

The paper pins `window_factor = 8` and the L=512 value of 64, but not whether the
draft *trains* with a fixed window or one scaled to the evaluation L. Training at
window 64 and evaluating at 120 would put the draft off-distribution in a way that
directly suppresses τ — indistinguishable from a failed reproduction.

**Decided 2026-09-01: the window scales as L/8.** Rationale: `calculate_itm`
computes `kv_draft` with `window = L/8` at every L, so a fixed window would be
measuring a configuration the cost model does not describe — and validating that
model is objective #1.

A fixed-window arm is deferred to future work. It has the better asymptote
(`ITM → 1.0` rather than 1.125, since a constant cache against a growing target
cache drives `speculate_cost/target_cost → 0`), and the crossover is decidable:
at L=4096 fixed beats proportional iff it retains >90.5% of proportional's τ. But
it is a fourth contribution on a project that has not yet delivered its first.

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

### 4.3 τ is not constant in L — the cost model assumes it is

MagicDec at T=1.0 measures τ = 3.307 / 3.763 / 3.725 at L = 256 / 512 / 1024: a
**14% range**, rising then plateauing. `calculate_itm` takes the L=512 measurement
and reuses it unchanged out to L=8192. This is the first direct evidence against
that assumption and is the core of the long-context contribution.

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

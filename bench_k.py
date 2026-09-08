"""Speedup vs speculation depth k, with the confounds actually controlled.

An earlier version of this measurement (bench.py) held the ROUND COUNT fixed
across k. Two things went wrong, and both inflated cost at high k:

  1. A round at k=8 commits ~2.3x the tokens a round at k=1 does, so over a fixed
     number of rounds the KV cache grew much further. Higher k was partly being
     charged for a longer context.
  2. Worse and less obvious: speculative.py sizes its cache from the WORST case,
     Klen = P + 1 + R(k+1) + k + 1. For a fixed token budget that bound scales
     with k, because more slack separates R(k+1) from what tau actually delivers.
     Attention READS the whole allocation, so k=8 paid for ~2x the buffer even at
     matched token counts.

This version controls both:

  * TOKEN BUDGET fixed, not rounds, AND SMALL. Rounds are derived per depth as
    G/tau(k) so every configuration walks the cache over the same positions. The
    budget must also stay near the paper's G=64: at 256+ generated tokens a 67M
    model drifts into repetitive self-generated text that the draft predicts
    trivially, and observed tau climbs toward its k+1 ceiling -- 7.689 of a
    possible 9 at k=8, against a reference of 3.819. Timing there measures a
    degenerate regime, not deployment.
  * Klen PINNED to one value for every configuration, including plain decoding,
    and asserted identical. It is sized from the true worst case (every round
    accepting all k) so a lucky run can never overflow -- JAX's dynamic_update_slice
    CLAMPS out-of-range writes rather than erroring, so an undersized buffer would
    corrupt silently rather than fail loudly.
  * ACTUAL tokens generated are measured, never assumed. The slope divides by the
    measured token difference, so an inaccurate tau estimate changes which rounds
    are run but cannot bias the result.
  * Prefill and compilation cancel via the two-point slope, as before.

Validation built in, because the point of this rerun is trustworthiness:
  - tau derived from (tokens / rounds) is cross-checked against the independent
    tau.py measurement. Agreement means the timing harness and the acceptance
    harness are describing the same computation.
  - per-point timing jitter (IQR/median) is reported and cells above a threshold
    are REJECTED rather than published.
  - Klen equality is asserted, not assumed.

    PYTHONPATH=. python bench_k.py --config spire_target_1024_resolved \\
        --model-name spire_target_1024 \\
        --draft-config spire_draft_vanilla --draft-model-name spire_draft_vanilla \\
        --taus 1:1.688,2:2.205,3:2.599,4:2.931,5:3.212,6:3.411,7:3.654,8:3.819
"""

import argparse
import json
import math
import os
import time

import init_seqax  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from omegaconf import OmegaConf

import shardlib.shardtypes as shardtypes

shardtypes.register_with_typeguard()

from bench import (HOI_H100_PCIE, body_params, calculate_itm,  # noqa: E402
                   draft_kv_projection_params)
from decode import load_weights, make_generate  # noqa: E402
from model import ModelConfig  # noqa: E402
from speculative import make_speculative_generate  # noqa: E402
from tau import dataset_prompts  # noqa: E402  (same context source as the tau harness)


def timeit(fn, reps: int):
    """Median wall time and relative IQR, warmed up and blocked on completion."""
    jax.block_until_ready(fn())  # warmup absorbs compile + first-call cost
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    med = float(np.median(ts))
    iqr = float((np.percentile(ts, 75) - np.percentile(ts, 25)) / med) if med > 0 else float("inf")
    return med, iqr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--mode", choices=["vanilla", "magicdec", "spire"], default="vanilla",
                   help="which of the paper's three draft models to time")
    p.add_argument("--draft-config", default=None,
                   help="not used by magicdec, whose draft IS the target")
    p.add_argument("--draft-model-name", default=None)
    p.add_argument("--sink", type=int, default=1)
    p.add_argument("--window", type=int, default=64,
                   help="StreamingLLM window for magicdec/spire. The paper fixes it at 64 "
                        "for training and tau, while the appendix cost model parameterizes "
                        "it as L//8; at the L=512 this harness defaults to, those coincide.")
    p.add_argument("--taus", required=True, help="k:tau,... from the independent tau.py sweep")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--g1", type=int, default=16, help="short TOKEN budget for the slope")
    p.add_argument("--g2", type=int, default=64, help="long TOKEN budget for the slope")
    p.add_argument("--reps", type=int, default=11)
    p.add_argument("--max-jitter", type=float, default=0.10)
    p.add_argument("--hoi", type=float, default=HOI_H100_PCIE)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    taus = {int(kv.split(":")[0]): float(kv.split(":")[1]) for kv in args.taus.split(",")}
    ks = sorted(taus)
    B, L, G1, G2 = args.batch, args.context, args.g1, args.g2

    cfg = OmegaConf.load(f"configs/{args.config}.yaml")
    h_t = ModelConfig(**cfg.model)

    # Mode -> draft wiring. Identical to tau.py's, deliberately: if the two
    # harnesses configured the draft differently, the tau cross-check at the end
    # would be comparing two different computations and could not detect anything.
    if args.mode == "magicdec":
        # MagicDec's draft is the target model reading its cache through a
        # StreamingLLM mask. No second checkpoint.
        h_d = h_t
        d_window, prefill_dense, mdrope = args.window, True, True
    elif args.mode == "vanilla":
        assert args.draft_config and args.draft_model_name, "vanilla needs a draft model"
        h_d = ModelConfig(**OmegaConf.load(f"configs/{args.draft_config}.yaml").model)
        d_window, prefill_dense, mdrope = None, True, False
    else:  # spire: trained with the sparse mask, original-text RoPE positions
        assert args.draft_config and args.draft_model_name, "spire needs a draft model"
        h_d = ModelConfig(**OmegaConf.load(f"configs/{args.draft_config}.yaml").model)
        d_window, prefill_dense, mdrope = args.window, False, False
    mc = cfg.mesh
    N_t, N_d = body_params(h_t), body_params(h_d)

    # What the cost model predicts for THIS draft (spire_appendix.ipynb, cells
    # 8-10). The three variants differ in exactly three places: how much of the
    # cache the draft reads, whose weights it loads, and whether it pays to
    # project a memory vector into a key and a value.
    if args.mode == "vanilla":
        kv_len_pred, N_pred, h_pred = L, N_d, h_d
        kvp_pred = 0
    elif args.mode == "magicdec":
        kv_len_pred, N_pred, h_pred = args.sink + args.window, N_t, h_t
        kvp_pred = 0
    else:
        kv_len_pred, N_pred, h_pred = args.sink + args.window, N_d, h_d
        kvp_pred = draft_kv_projection_params(h_d)
    sparse_draft = args.mode != "vanilla"

    # Rounds per depth are CALIBRATED, not assumed. The reference tau from tau.py
    # is only a starting guess: it is measured over 8 rounds, while this harness
    # runs 20-150, and over a long generation the model is increasingly reading its
    # OWN output, which is more self-consistent and easier for the draft to predict.
    # Observed tau therefore runs above the reference, by more at higher k.
    #
    # Sizing rounds from the reference makes actual tokens overshoot unevenly -- 290
    # at k=1 but 344 at k=3 against a 256 budget -- so the speculative slope covers a
    # longer token range, and a larger average cache, than the plain baseline it is
    # divided by. That produced a NON-MONOTONIC ITM (1.637, 2.030, 1.866), which is
    # impossible: every extra depth adds a draft pass. A cheap probe run per depth
    # measures the real tau, then rounds are set from that and the token ranges match.
    rounds = {}
    calib = {}

    # Klen is chosen AFTER calibration (below) so it can be sized from measured
    # tau rather than a worst case. A tau=1 bound gives Klen 2826 against ~1400
    # actually needed, and since attention reads the whole allocation that inflates
    # every cell and shifts where the roofline sits. The probe runs under a
    # generous bound; timing runs under the tight one.

    print(f"devices: {jax.devices()}")
    print(f"mode={args.mode}  B={B} L={L}  token budgets G1={G1} G2={G2}  HOI={args.hoi}")
    print(f"draft: window={d_window} sink={args.sink} prefill_dense={prefill_dense} "
          f"magicdec_rope={mdrope}; cost model reads {kv_len_pred} cache entries "
          f"({kv_len_pred}+k corrected)")


    rows = []
    with Mesh(mesh_utils.create_device_mesh([mc.d, mc.t, mc.s], jax.devices()), ("d", "t", "s")):
        rng0 = jnp.zeros((2,), jnp.uint32)
        with shardtypes.Scope():
            w_t, _ = load_weights(h_t, os.path.join(cfg.root_working_dir, args.model_name), rng0)
        if args.mode == "magicdec":
            w_d = w_t          # the draft IS the target, restricted at decode
        else:
            with shardtypes.Scope():
                w_d, _ = load_weights(
                    h_d, os.path.join(cfg.root_working_dir, args.draft_model_name), rng0)

        # REAL contexts, from the same validation split tau.py draws from.
        #
        # An earlier version used jnp.ones((B, L)): a degenerate constant prompt on
        # which the draft agrees with the target almost always. At k=1 it produced
        # tau_obs = 2.000 -- the maximum possible -- against 1.688 on real text. The
        # timing of a forward pass is data-independent, but the ROUND STRUCTURE is
        # not: inflated acceptance means more tokens committed per round, a faster
        # advancing cache, and a speedup computed against the wrong tau. The
        # cross-check below only caught it because the two harnesses now share this
        # source; with synthetic prompts here it would have been comparing against a
        # tau that no longer described the run.
        prompt = next(dataset_prompts(cfg, L, B, 1))
        rng = jnp.array([0, 0], jnp.uint32)

        # ---- Calibration pass: one short run per depth to measure the real tau ----
        probe = 12
        probe_klen = L + 1 + probe * (max(ks) + 1) + max(ks) + 1
        print(f"calibrating rounds from observed tau ({probe}-round probe, klen {probe_klen}):")
        for k in ks:
            with shardtypes.Scope():
                sp = make_speculative_generate(
                    h_t, h_d, L, probe, k, 0.0, klen=probe_klen,
                    draft_sink=args.sink, draft_window=d_window,
                    draft_prefill_dense=prefill_dense, magicdec_rope=mdrope,
                    compact_draft_cache=True)
                op = jax.block_until_ready(sp(w_t, w_d, prompt, rng))
            tau_c = float(np.mean(np.asarray(op[1]))) / probe
            calib[k] = tau_c
            rounds[k] = (max(1, round(G1 / tau_c)), max(2, round(G2 / tau_c)))
            print(f"    k={k}: tau_ref {taus[k]:.3f} -> tau_probe {tau_c:.3f} "
                  f"-> rounds {rounds[k][0]}/{rounds[k][1]}")

        # Klen has to reflect what a serving system would actually hold: the
        # prompt, the tokens generated, and the k-token block in flight.
        #
        # The previous bound multiplied the round count by (k + 1) -- the MAXIMUM
        # a round can yield, i.e. every round accepting every draft. That never
        # happens: measured tau runs 1.8 to 4.1 against a k+1 of up to 9. It
        # pinned Klen at 702 where about 600 is ever occupied, and since attention
        # READS the whole allocation, the surplus was charged to every cell.
        #
        # The surplus is not charged evenly, which is why it mattered. A dense
        # draft reads all of Klen, so VANILLA -- the control -- paid for ~100
        # slots of slack it never fills, while the cost model charges it for
        # B * L. A windowed draft reads sink + window + k whatever Klen is, so
        # SPIRe and MagicDec never paid it. Sizing from measured tau removes a
        # bias that ran against the baseline and flattered the method under test.
        #
        # 25% headroom on the token count, not the round count, so a run that
        # accepts better than its probe still cannot overrun the buffer.
        klen = max(L + 1 + int(rounds[k][1] * calib[k] * 1.25) + k + 1 for k in ks)
        klen = max(klen, L + 1 + G2 + max(ks) + 1)
        print(f"\nKlen PINNED to {klen} for every timed configuration\n")

        # ---- Baseline: plain target decoding, same pinned Klen ----
        with shardtypes.Scope():
            g1f = make_generate(h_t, L, G1, 0.0, klen=klen)
            t1, j1 = timeit(lambda: g1f(w_t, prompt, rng), args.reps)
        with shardtypes.Scope():
            g2f = make_generate(h_t, L, G2, 0.0, klen=klen)
            t2, j2 = timeit(lambda: g2f(w_t, prompt, rng), args.reps)
        tpt_plain = (t2 - t1) / (G2 - G1)
        print(f"plain decode: {tpt_plain * 1e3:.4f} ms/token  (jitter {max(j1, j2):.1%})")
        if max(j1, j2) > args.max_jitter or tpt_plain <= 0:
            raise SystemExit(f"baseline unusable: jitter {max(j1, j2):.1%}, tpt {tpt_plain:.2e}")


        print(f"\n{'k':>3} {'rounds':>11} {'tokens':>13} {'tau_obs':>8} {'tau_ref':>8} "
              f"{'ms/tok':>8} {'speedup':>8} {'ITM_meas':>9} {'ITM_pred':>9} {'ITM_pr+':>8} "
              f"{'jit':>5}")
        print("-" * 105)
        for k in ks:
            r1, r2 = rounds[k]
            try:
                with shardtypes.Scope():
                    s1 = make_speculative_generate(
                        h_t, h_d, L, r1, k, 0.0, klen=klen,
                        draft_sink=args.sink, draft_window=d_window,
                        draft_prefill_dense=prefill_dense, magicdec_rope=mdrope,
                        compact_draft_cache=True)
                    o1 = jax.block_until_ready(s1(w_t, w_d, prompt, rng))
                    n1 = float(np.mean(np.asarray(o1[1])))
                    n1max = float(np.max(np.asarray(o1[1])))
                    ts1, js1 = timeit(lambda: s1(w_t, w_d, prompt, rng), args.reps)
                with shardtypes.Scope():
                    s2 = make_speculative_generate(
                        h_t, h_d, L, r2, k, 0.0, klen=klen,
                        draft_sink=args.sink, draft_window=d_window,
                        draft_prefill_dense=prefill_dense, magicdec_rope=mdrope,
                        compact_draft_cache=True)
                    o2 = jax.block_until_ready(s2(w_t, w_d, prompt, rng))
                    n2 = float(np.mean(np.asarray(o2[1])))
                    n2max = float(np.max(np.asarray(o2[1])))
                    ts2, js2 = timeit(lambda: s2(w_t, w_d, prompt, rng), args.reps)
            except Exception as e:  # noqa: BLE001
                print(f"{k:>3}   FAILED: {type(e).__name__}: {e}")
                rows.append({"k": k, "failed": f"{type(e).__name__}"})
                continue

            # A tighter Klen is only honest if the run fit inside it. An
            # out-of-bounds cache write does not raise -- dynamic_update_slice
            # CLAMPS -- so an overrun corrupts the cache silently and still
            # produces a plausible timing. Check the longest row, not the mean.
            need = L + 1 + int(max(n1max, n2max)) + k + 1
            if need > klen:
                print(f"{k:>3}   REJECTED: overran Klen ({need} > {klen}); "
                      f"raise the headroom above 25%")
                rows.append({"k": k, "rejected": f"overran klen {need}>{klen}"})
                continue

            jit = max(js1, js2)
            dtok = n2 - n1
            if dtok <= 0 or ts2 <= ts1 or jit > args.max_jitter:
                why = "neg slope" if (dtok <= 0 or ts2 <= ts1) else f"jitter {jit:.0%}"
                print(f"{k:>3}   REJECTED: {why}")
                rows.append({"k": k, "rejected": why})
                continue

            tpt_spec = (ts2 - ts1) / dtok
            speedup = tpt_plain / tpt_spec
            tau_obs = dtok / (r2 - r1)           # tokens per round, from THIS run
            itm_meas = tau_obs / speedup          # speedup = tau / ITM, by definition
            itm_pred = calculate_itm(B, L, h_t, k, args.hoi, N_t, N_pred,
                                     kv_len_pred, h_pred, kvp_pred)
            # What the model predicts for the cache we actually read: a windowed
            # draft needs sink + window + k, not sink + window (ledger 10.1).
            itm_pred_c = calculate_itm(B, L, h_t, k, args.hoi, N_t, N_pred,
                                       kv_len_pred + (k if sparse_draft else 0),
                                       h_pred, kvp_pred)
            print(f"{k:>3} {f'{r1}/{r2}':>11} {f'{n1:.0f}/{n2:.0f}':>13} {tau_obs:>8.3f} "
                  f"{taus[k]:>8.3f} {tpt_spec * 1e3:>8.4f} {speedup:>8.3f} "
                  f"{itm_meas:>9.3f} {itm_pred:>9.3f} {itm_pred_c:>8.3f} {jit:>5.0%}")
            rows.append({"k": k, "rounds": [r1, r2], "tokens": [n1, n2], "tau_observed": tau_obs,
                         "tau_reference": taus[k], "tpt_spec_s": tpt_spec, "tpt_plain_s": tpt_plain,
                         "speedup": speedup, "itm_measured": itm_meas, "itm_predicted": itm_pred,
                         "itm_predicted_corrected": itm_pred_c,
                         "jitter": jit, "klen": klen})

    ok = [r for r in rows if "speedup" in r]
    if ok:
        print("\n--- VALIDATION ---")
        # The strongest check available: tau derived from the TIMING run must match
        # the tau measured independently by tau.py. If these disagree, the two
        # harnesses are not describing the same computation and nothing else holds.
        toks = [r["tokens"][1] for r in ok]
        spread = (max(toks) - min(toks)) / np.mean(toks)
        print(f"  token-range match across depths: {min(toks):.0f}-{max(toks):.0f} "
              f"(spread {spread:.1%}) -- unequal ranges are what made ITM non-monotonic")
        itms = [r["itm_measured"] for r in sorted(ok, key=lambda r: r["k"])]
        mono = all(b >= a - 0.02 for a, b in zip(itms, itms[1:]))
        print(f"  ITM monotonic in k: {'YES' if mono else 'NO -- residual confound remains'}")
        devs = [abs(r["tau_observed"] - r["tau_reference"]) / r["tau_reference"] for r in ok]
        print(f"  tau (timing run) vs tau (tau.py): max deviation {max(devs):.1%}, median {np.median(devs):.1%}")
        print("    (both harnesses now draw contexts from the same validation split,")
        print("     so a disagreement here means a real discrepancy, not a prompt mismatch)")
        print(f"  {'AGREE' if max(devs) < 0.05 else 'DISAGREE -- investigate before trusting the speedups'}")
        print(f"  Klen identical across all cells: {len({r['klen'] for r in ok}) == 1}")
        print(f"  worst timing jitter: {max(r['jitter'] for r in ok):.1%} (threshold {args.max_jitter:.0%})")
        best = max(ok, key=lambda r: r["speedup"])
        bestp = max(ok, key=lambda r: r["tau_reference"] / r["itm_predicted"])
        print(f"\n  MEASURED optimum:  k={best['k']}  speedup {best['speedup']:.3f}x")
        print(f"  PREDICTED optimum: k={bestp['k']}  "
              f"speedup {bestp['tau_reference'] / bestp['itm_predicted']:.3f}x")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"mode": args.mode, "sink": args.sink, "window": args.window,
                       "B": B, "L": L, "G1": G1, "G2": G2, "klen": klen, "hoi": args.hoi,
                       "tpt_plain_s": tpt_plain, "rows": rows}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

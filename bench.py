"""Measure the iteration-time multiplier (ITM) and compare it to the cost model.

This is objective #1 of the reproduction: the paper's speedups come from an
analytical roofline model, never a stopwatch. `calculate_itm` predicts

    ITM = (speculate_cost + verify_cost) / target_cost

the cost of one speculative round relative to one plain target decode step, with
end-to-end speedup = tau / ITM. Here we measure that ratio directly.

METHOD. Timing a single call would fold in prefill and JIT compilation. Instead we
time each configuration at two generation lengths and take the SLOPE:

    per_step = (t(G2) - t(G1)) / (G2 - G1)

which cancels prefill, compilation, and dispatch overhead exactly, because both
runs pay them once. The same is done for speculative rounds. Every timed call is
preceded by a warmup call and followed by block_until_ready, so we are measuring
device work rather than async dispatch.

SCOPE. Only `--mode vanilla` gives a meaningful draft-side number today. The
sparse draft's cache is a MASK, not a compact ring buffer, so a SPIRe or MagicDec
draft still reads the full Klen from HBM: its distributions and tau are exact, but
its memory traffic -- the very thing the cost model prices -- is unreduced. Timing
those would measure our implementation rather than the method (fidelity-ledger
2.5). The target and verify terms are unaffected and are measured for all modes.

    PYTHONPATH=. python bench.py --config spire_target_1024_resolved \
        --model-name spire_target_1024 --mode vanilla \
        --draft-config spire_draft_vanilla --draft-model-name spire_draft_vanilla \
        --batches 1,4,16,64 --contexts 256,512,1024 --k 4
"""

import argparse
import json
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

from decode import load_weights, make_generate  # noqa: E402
from model import ModelConfig  # noqa: E402
from speculative import make_speculative_generate  # noqa: E402

# H100 PCIe: 1513 TFLOPS FP8 dense / 2.0 TB/s HBM2e. Dtype-invariant on H100 --
# bf16 gives 2 * 756.5/2.0 = the same 756 -- because FP8 peak is exactly 2x bf16,
# so the doubled bytes cancel the halved throughput. SXM would be 600.
HOI_H100_PCIE = 756.0


def kv_elements_per_token(h: ModelConfig) -> int:
    """K and V, per layer, per KV head."""
    return 2 * h.layers * h.n_kv * h.d_head


def body_params(h: ModelConfig) -> int:
    per_layer = (
        2 * h.d_model  # the two layernorms
        + 2 * h.d_model * h.n_q_per_kv * h.n_kv * h.d_head  # w_q, w_o
        + 2 * h.d_model * h.n_kv * h.d_head  # w_k, w_v
        + 3 * h.d_model * h.d_ff  # SwiGLU gate/up/down
    )
    return h.layers * per_layer


def calculate_itm(B, L, h_t, k, HOI, N_target, N_draft, kv_draft_len, h_d) -> float:
    """The paper's cost model, transcribed from spire_appendix.ipynb.

    Counts ELEMENTS, not bytes; f = max is the roofline with perfect overlap
    assumed. HOI converts an element count into FLOP-equivalent time."""
    f = max
    kv_target = B * L * kv_elements_per_token(h_t)
    kv_draft = B * kv_draft_len * kv_elements_per_token(h_d)
    FLOPs_target = 2 * N_target * B
    FLOPs_draft = 2 * N_draft * B
    FLOPs_verify = FLOPs_target * (k + 1)
    speculate = k * f(FLOPs_draft, (N_draft + kv_draft) * HOI)
    verify = f(FLOPs_verify, (N_target + kv_target) * HOI)
    target = f(FLOPs_target, (N_target + kv_target) * HOI)
    return (speculate + verify) / target


def timeit(fn, *args, reps: int = 7) -> tuple:
    """Median wall time of `fn(*args)`, warmed up and blocked on completion.

    Returns (median, relative_spread). The spread is the interquartile range over
    the median: a slope built from two noisy points can go NEGATIVE, which is how
    the first version of this benchmark reported -40ms per round. Carrying the
    spread lets the caller reject a measurement instead of publishing it."""
    jax.block_until_ready(fn(*args))  # warmup: compile + first-call overhead
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    med = float(np.median(ts))
    spread = float((np.percentile(ts, 75) - np.percentile(ts, 25)) / med) if med > 0 else float("inf")
    return med, spread


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--mode", choices=["vanilla", "self"], default="vanilla")
    p.add_argument("--draft-config", default=None)
    p.add_argument("--draft-model-name", default=None)
    p.add_argument("--batches", default="1,4,16,64")
    p.add_argument("--contexts", default="256,512,1024")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--hoi", type=float, default=HOI_H100_PCIE)
    p.add_argument("--g1", type=int, default=32, help="short generation length for the slope")
    p.add_argument("--g2", type=int, default=160, help="long generation length for the slope")
    p.add_argument("--reps", type=int, default=7, help="timed repetitions per point")
    p.add_argument("--out", default=None, help="write results as JSON here")
    args = p.parse_args()

    cfg = OmegaConf.load(f"configs/{args.config}.yaml")
    h_t = ModelConfig(**cfg.model)
    h_d = ModelConfig(**OmegaConf.load(f"configs/{args.draft_config}.yaml").model) if args.draft_config else h_t
    mc = cfg.mesh
    N_t, N_d = body_params(h_t), body_params(h_d)

    print(f"devices: {jax.devices()}")
    print(f"HOI = {args.hoi}   N_target = {N_t:,}   N_draft = {N_d:,}   k = {args.k}")
    print(f"slope from G={args.g1} to G={args.g2}\n")

    rows = []
    with Mesh(mesh_utils.create_device_mesh([mc.d, mc.t, mc.s], jax.devices()), ("d", "t", "s")):
        rng0 = jnp.zeros((2,), jnp.uint32)
        with shardtypes.Scope():
            w_t, _ = load_weights(h_t, os.path.join(cfg.root_working_dir, args.model_name), rng0)
        if args.mode == "vanilla":
            assert args.draft_model_name, "--draft-model-name required for vanilla"
            with shardtypes.Scope():
                w_d, _ = load_weights(h_d, os.path.join(cfg.root_working_dir, args.draft_model_name), rng0)
        else:
            w_d, h_d, N_d = w_t, h_t, N_t

        print(f"{'B':>5} {'L':>6} {'t_step(ms)':>11} {'t_round(ms)':>12} {'ITM_meas':>9} {'ITM_pred':>9} {'err':>8}")
        print("-" * 66)
        for L in [int(x) for x in args.contexts.split(",")]:
            for B in [int(x) for x in args.batches.split(",")]:
                prompt = jnp.ones((B, L), jnp.uint32)
                rng = jnp.array([0, 0], jnp.uint32)
                try:
                    # --- plain target decode: per-token cost ---
                    with shardtypes.Scope():
                        g1 = make_generate(h_t, L, args.g1, 0.0)
                        t1, sp1 = timeit(lambda: g1(w_t, prompt, rng), reps=args.reps)
                    with shardtypes.Scope():
                        g2 = make_generate(h_t, L, args.g2, 0.0)
                        t2, sp2 = timeit(lambda: g2(w_t, prompt, rng), reps=args.reps)
                    t_step = (t2 - t1) / (args.g2 - args.g1)

                    # --- speculative: per-round cost ---
                    with shardtypes.Scope():
                        s1 = make_speculative_generate(h_t, h_d, L, args.g1, args.k, 0.0)
                        r1, sp3 = timeit(lambda: s1(w_t, w_d, prompt, rng), reps=args.reps)
                    with shardtypes.Scope():
                        s2 = make_speculative_generate(h_t, h_d, L, args.g2, args.k, 0.0)
                        r2, sp4 = timeit(lambda: s2(w_t, w_d, prompt, rng), reps=args.reps)
                    t_round = (r2 - r1) / (args.g2 - args.g1)
                except Exception as e:  # OOM at large B*L is expected; record and continue
                    print(f"{B:>5} {L:>6}   skipped: {type(e).__name__}")
                    rows.append({"B": B, "L": L, "skipped": type(e).__name__})
                    continue

                # Reject rather than publish. A slope from two noisy points is only
                # meaningful when the difference dominates the jitter in each point.
                worst_spread = max(sp1, sp2, sp3, sp4)
                if t_step <= 0 or t_round <= 0 or worst_spread > 0.10:
                    reason = "negative slope" if (t_step <= 0 or t_round <= 0) else f"jitter {worst_spread:.0%}"
                    print(f"{B:>5} {L:>6}   rejected: {reason}")
                    rows.append({"B": B, "L": L, "rejected": reason})
                    continue
                itm_meas = t_round / t_step
                itm_pred = calculate_itm(B, L, h_t, args.k, args.hoi, N_t, N_d, L, h_d)
                err = (itm_meas - itm_pred) / itm_pred
                print(
                    f"{B:>5} {L:>6} {t_step * 1e3:>11.3f} {t_round * 1e3:>12.3f} "
                    f"{itm_meas:>9.3f} {itm_pred:>9.3f} {err:>+7.1%}"
                )
                rows.append(
                    {"B": B, "L": L, "t_step_s": t_step, "t_round_s": t_round,
                     "itm_measured": itm_meas, "itm_predicted": itm_pred, "rel_err": err}
                )

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"hoi": args.hoi, "k": args.k, "mode": args.mode,
                       "N_target": N_t, "N_draft": N_d, "rows": rows}, f, indent=2)
        print(f"\nwrote {args.out}")

    ok = [r for r in rows if "rel_err" in r]
    if ok:
        errs = np.array([r["rel_err"] for r in ok])
        print(f"\n{len(ok)} cells measured. median |err| = {np.median(np.abs(errs)):.1%}, "
              f"worst = {np.max(np.abs(errs)):.1%}")
        bad = [r for r in ok if abs(r["rel_err"]) > 0.15]
        print(f"{len(bad)} cell(s) diverge >15% -- the profiling targets.")


if __name__ == "__main__":
    main()

"""Integrity checks for a flat-tokens dataset before spending GPU-hours on it.

Catches the failure modes that are silent rather than loud:
  * degenerate documents -- a run of one repeated token, which is what an absent
    zarr chunk (fill_value 0) turns into after remap_zero_token. These load fine
    and train fine; they just quietly waste budget and corrupt eval contexts.
  * token ids outside [1, vocab), including any token 0 that escaped the remap.
  * seq_starts that disagree with the encoded low-bit start markers.

Example:
    PYTHONPATH=. python tools/verify_flat_tokens.py \
        --zarr ~/data/longcrawl64-flat-tokens.zarr --vocab 50304
"""

import argparse

import numpy as np
import zarr


def check_split(group, split: str, vocab: int, sample_docs: int) -> list[str]:
    problems = []
    g = group[split]
    enc = g["encoded_tokens"]
    starts = g["seq_starts"][:]
    n_seq = len(starts) - 1
    total = int(enc.shape[0])

    print(f"\n--- {split} ---")
    print(f"  sequences: {n_seq:,}   tokens: {total:,}   max_token_id attr: {g.attrs['max_token_id']}")

    if int(starts[-1]) != total:
        problems.append(f"{split}: seq_starts[-1]={int(starts[-1])} != total tokens {total}")

    lengths = np.diff(starts.astype(np.int64))
    if lengths.min() <= 0:
        problems.append(f"{split}: non-increasing seq_starts (min length {lengths.min()})")
    print(f"  document length: min {lengths.min():,} max {lengths.max():,}")

    # Sample documents spread across the whole array, not just the head.
    idx = np.unique(np.linspace(0, n_seq - 1, num=min(sample_docs, n_seq)).astype(np.int64))
    degenerate, oor, zero_tok = 0, 0, 0
    for i in idx:
        lo, hi = int(starts[i]), int(starts[i + 1])
        e = enc[lo:hi]
        tokens = (e >> 1).astype(np.int64)
        is_start = (e & 1).astype(bool)
        if len(np.unique(tokens)) == 1:
            degenerate += 1
        if tokens.min() < 0 or tokens.max() >= vocab:
            oor += 1
        if (tokens == 0).any():
            zero_tok += 1
        if is_start[0] != True or is_start[1:].any():  # noqa: E712
            problems.append(f"{split}: doc {i} start-bit pattern wrong")

    print(f"  sampled {len(idx):,} docs: degenerate={degenerate} out_of_range={oor} contains_token_0={zero_tok}")
    if degenerate:
        problems.append(f"{split}: {degenerate}/{len(idx)} sampled documents are a single repeated token")
    if oor:
        problems.append(f"{split}: {oor}/{len(idx)} sampled documents have ids outside [0,{vocab})")
    if zero_tok:
        problems.append(f"{split}: {zero_tok}/{len(idx)} sampled documents still contain token 0")
    return problems


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr", required=True)
    p.add_argument("--vocab", type=int, default=50304)
    p.add_argument("--sample-docs", type=int, default=2000)
    args = p.parse_args()

    group = zarr.open_group(args.zarr, mode="r")
    problems = []
    for split in ("train", "validation"):
        problems += check_split(group, split, args.vocab, args.sample_docs)

    print()
    if problems:
        print("FAILED:")
        for x in problems:
            print(f"  - {x}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

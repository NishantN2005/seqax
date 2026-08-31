"""Fetch a subset of LongCrawl64 from the HuggingFace mirror `clankur/longcrawl64`.

The source named in the seqax README (`gs://longcrawl64`) no longer exists --
the JSON API, the XML API, and the virtual-host URL all report the bucket is
gone. This mirror is the remaining route. Its train + heldout document counts
sum to exactly the 6,661,465 Manifest AI publishes, and sampled documents decode
to coherent text under GPT-2's tokenizer, so the content is trustworthy even
though the container is not the original.

Two things about the mirror force the design here:

  1. It is PARTIAL. train.zarr holds ~313 of the source's 3,228 row-chunks
     (~641k of 6.6M documents, ~10%). heldout.zarr is complete (26 row-chunks).
  2. Its row-chunks are NON-CONTIGUOUS -- present indices are scattered over
     0..1279. Requesting rows 0..N-1 mostly 404s.

So this script asks the Hub which chunks actually exist, keeps the rows that
have every column-chunk we want, and REINDEXES them densely on download: source
chunk `r.c` is stored as `i.c` for dense i. Chunk files are independent, and
training shuffles documents anyway, so renaming only changes which row a
document lands on, never its contents.

The rewritten `.zarray` is what makes a partial download safe. Zarr fills absent
chunks with `fill_value` (0) and raises nothing, so a partial download under the
original metadata would silently yield all-zero documents that look fine.

Note on `--col-chunks`: seqax's loader keeps the first `dataset_seqlen` tokens of
each document and splits them into `dataset_seqlen / seqlen` training sequences
(input_loader.py:108,248). So column-chunks bound the usable dataset_seqlen, and
2 col-chunks (4096 tokens) is what the mirror's document count needs to reach a
Chinchilla budget for the 67M target.

Example:
    python fetch_longcrawl64_subset.py --out ~/data/longcrawl64 --col-chunks 2
"""

import argparse
import json
import re
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO = "clankur/longcrawl64"
CHUNK_ROWS = 2048
CHUNK_COLS = 2048
SPLITS = ("train.zarr", "heldout.zarr")
# True document counts of the source arrays, from their published .zarray shapes.
# Needed because the final row-chunk of a split is ragged: heldout's row 25 holds
# 931 real documents, not 2048, and zarr pads the rest with fill_value=0. Counting
# it as full would invent all-zero documents that survive as runs of token 50256.
SPLIT_DOCS = {"train.zarr": 6_609_334, "heldout.zarr": 52_131}


def list_available_chunks(split: str) -> dict[int, set[int]]:
    """Ask the Hub which `row.col` chunk files exist for a split."""
    rows: dict[int, set[int]] = defaultdict(set)
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{split}?limit=1000"
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "seqax-fetch"})
        with urllib.request.urlopen(req, timeout=60) as r:
            entries = json.load(r)
            link = r.headers.get("Link", "")
        for e in entries:
            name = e["path"].split("/")[-1]
            if re.fullmatch(r"\d+\.\d+", name):
                row, col = map(int, name.split("."))
                rows[row].add(col)
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return rows


def fetch_split(split: str, out_dir: Path, col_chunks: int, max_rows: int | None, workers: int) -> tuple[int, int]:
    available = list_available_chunks(split)
    wanted_cols = set(range(col_chunks))
    usable = sorted(r for r, cols in available.items() if wanted_cols <= cols)
    if not usable:
        raise SystemExit(f"{split}: no row-chunk has all of columns {sorted(wanted_cols)}")
    if max_rows is not None:
        usable = usable[:max_rows]

    # A source row-chunk holds 2048 documents except the array's final one, which
    # is ragged. Selected rows are sorted ascending, so a ragged row can only be
    # last, which is exactly where a truncated shape needs it.
    n_docs = sum(min(CHUNK_ROWS, SPLIT_DOCS[split] - r * CHUNK_ROWS) for r in usable)
    print(
        f"{split}: {len(available)} row-chunks on mirror, {len(usable)} have cols 0..{col_chunks - 1}"
        f" -> {n_docs:,} docs x {col_chunks * CHUNK_COLS:,} tokens"
    )

    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # (source name, dense destination name)
    jobs = [(f"{split}/{r}.{c}", f"{i}.{c}") for i, r in enumerate(usable) for c in range(col_chunks)]

    def get(job):
        src, dst = job
        dst_path = split_dir / dst
        if dst_path.exists():
            return
        # Download into a scratch dir under the split, then rename to the dense index.
        local = hf_hub_download(REPO, src, repo_type="dataset", local_dir=str(out_dir / ".raw"))
        Path(local).replace(dst_path)

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(get, j): j for j in jobs}
        for fut in tqdm(as_completed(futures), total=len(jobs), desc=split, unit="chunk"):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                failures.append((futures[fut][0], repr(e)))

    if failures:
        print(f"\n{len(failures)} chunk(s) failed:", file=sys.stderr)
        for rel, err in failures[:5]:
            print(f"  {rel}: {err}", file=sys.stderr)
        raise SystemExit(f"{split}: incomplete download; rerun to resume")

    zarray = {
        "chunks": [CHUNK_ROWS, CHUNK_COLS],
        "compressor": {"blocksize": 0, "clevel": 5, "cname": "lz4", "id": "blosc", "shuffle": 1},
        "dtype": "<u2",
        "fill_value": 0,
        "filters": None,
        "order": "C",
        "shape": [n_docs, col_chunks * CHUNK_COLS],
        "zarr_format": 2,
    }
    (split_dir / ".zarray").write_text(json.dumps(zarray, indent=4))
    (split_dir / ".zattrs").write_text("{}")
    print(f"{split}: wrote .zarray shape {zarray['shape']}")
    return n_docs, col_chunks * CHUNK_COLS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--col-chunks", type=int, default=2, help="column-chunks (2048 tokens each) per document")
    p.add_argument("--max-train-rows", type=int, default=None, help="cap train row-chunks (2048 docs each)")
    p.add_argument("--workers", type=int, default=24)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for split in SPLITS:
        results[split] = fetch_split(split, args.out, args.col_chunks, args.max_train_rows, args.workers)
        if split == "train.zarr":
            args.max_train_rows = None  # cap applies to train only

    raw = args.out / ".raw"
    if raw.exists():
        import shutil

        shutil.rmtree(raw, ignore_errors=True)

    print(f"\nDone -> {args.out}")
    train_docs, width = results["train.zarr"]
    held_docs, _ = results["heldout.zarr"]
    print(f"  train:   {train_docs:,} docs")
    print(f"  heldout: {held_docs:,} docs")
    print("\nTraining tokens available, by dataset_seqlen (loader keeps the first")
    print("dataset_seqlen tokens of each document, split into seqlen-sized sequences):")
    for sl in (1024, 2048, 4096, 8192):
        if sl <= width:
            print(f"  dataset_seqlen={sl:5d} -> {train_docs * sl / 1e9:.2f}B tokens")


if __name__ == "__main__":
    main()

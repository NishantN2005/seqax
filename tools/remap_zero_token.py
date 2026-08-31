"""Remap zeros in a flat-tokens dataset in-place.

LongCrawl64 is tokenized with GPT-2, where token 0 is '!'. seqax reserves token
0, so every occurrence is moved to 50256. Values here are flat-tokens ENCODED
(token_id * 2, +1 at a sequence start), so encoded 0 and 1 are the two forms of
token 0 and both must move, preserving the start bit.

Not idempotent: a second run trips the max_token_id assert, which is the
intended guard rather than a bug.

Example:
    python remap_zero_token.py --zarr ~/data/longcrawl64-flat-tokens.zarr
"""

import argparse

import numpy as np
import zarr
from tqdm import tqdm


def remap_zeros_in_flat_tokens_dataset(zarr_dir, REMAP_TO=50256):
    """
    Remap zeros in a flat-tokens format zarr array in-place.

    Args:
        zarr_dir: Path to zarr group in flat tokens format
    """
    group = zarr.open_group(zarr_dir, mode="r+")

    for split in ["train", "validation"]:
        print(f"\nProcessing {split}...")

        encoded_tokens = group[f"{split}/encoded_tokens"]

        max_token_id = group[split].attrs["max_token_id"]
        assert max_token_id < REMAP_TO, "REMAP_TO must be greater than max_token_id"
        group[split].attrs["max_token_id"] = max(max_token_id, REMAP_TO)

        chunk_size = encoded_tokens.chunks[0]
        total_tokens = encoded_tokens.shape[0]

        remapped = 0
        for start_idx in tqdm(range(0, total_tokens, chunk_size)):
            end_idx = min(start_idx + chunk_size, total_tokens)
            chunk = encoded_tokens[start_idx:end_idx]
            remapped += int(((chunk == 0) | (chunk == 1)).sum())
            # Modify chunk in-place
            new_chunk = np.where(chunk == 0, REMAP_TO * 2, np.where(chunk == 1, REMAP_TO * 2 + 1, chunk))
            encoded_tokens[start_idx:end_idx] = new_chunk
        print(f"{split}: remapped {remapped:,} of {total_tokens:,} tokens ({100 * remapped / total_tokens:.4f}%)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr", required=True, help="flat-tokens zarr group to modify in place")
    p.add_argument("--remap-to", type=int, default=50256)
    args = p.parse_args()
    remap_zeros_in_flat_tokens_dataset(args.zarr, args.remap_to)


if __name__ == "__main__":
    main()

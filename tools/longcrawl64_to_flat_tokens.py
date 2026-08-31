"""Convert LongCrawl64 (https://manifestai.com/articles/longcrawl64/) to flat-tokens format.

The dataset is pre-tokenized with GPT-2's tokenizer, where token 0 is '!'. In
seqax token 0 is reserved, so run tools/remap_zero_token.py on the OUTPUT of
this script to move it to 50256. See docs/flat-tokens.md for the format.

Source: `gs://longcrawl64` no longer exists (all GCS endpoints 404). Use
tools/fetch_longcrawl64_subset.py to pull a subset from the HuggingFace mirror
`clankur/longcrawl64` instead.

Two changes from the original upstream version:
  * Source chunk deletion is now opt-in (--delete-source). It used to be
    unconditional, which made a failed conversion destroy the download.
  * --seq-width truncates each document. seqax's loader keeps only the first
    `dataset_seqlen` tokens of a document (input_loader.py:248), so storing
    65,536-token documents to train at 1024 costs 64x the disk and 64x the
    read bandwidth per step for data that is never looked at.

Example:
    python longcrawl64_to_flat_tokens.py --input ~/data/longcrawl64 \
        --output ~/data/longcrawl64-flat-tokens.zarr --seq-width 1024
"""

import argparse
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm


def inspect_zarr_array(path):
    """Inspect and print details about a zarr array."""
    z = zarr.open(path, mode="r")
    print(f"\nInspecting {path}:")
    print(f"Shape: {z.shape}")
    print(f"Dtype: {z.dtype}")
    print(f"Chunks: {z.chunks}")
    print(f"Compressor: {z.compressor}")
    num_bytes = np.prod(z.shape) * z.dtype.itemsize
    print(f"Theoretical uncompressed size: {num_bytes / (1024**3):.2f} GB")
    print(f"Size on disk: {z.nbytes_stored / (1024**3):.2f} GB")
    return z


def create_flat_tokens_dataset(input_dir, output_dir, seq_width=None, delete_source=False):
    """
    Convert 2D uint16 zarr arrays to flat-tokens format.

    Args:
        input_dir: Path to directory containing train.zarr and heldout.zarr
        output_dir: Path where the new zarr group should be created
        seq_width: If set, keep only the first `seq_width` tokens of each document.
        delete_source: Delete source chunk files as they are consumed (saves disk,
            destroys the input). Off by default.
    """
    root = zarr.open_group(output_dir, mode="w")

    split_mapping = {"train": "train.zarr", "validation": "heldout.zarr"}

    for split_out, split_file in split_mapping.items():
        print(f"\nProcessing {split_file} -> {split_out}...")

        input_path = Path(input_dir) / split_file
        input_array = inspect_zarr_array(input_path)

        num_sequences = input_array.shape[0]
        source_width = input_array.shape[1]
        chunk_rows = input_array.chunks[0]
        chunk_cols = input_array.chunks[1]

        seq_len = source_width if seq_width is None else min(seq_width, source_width)
        if seq_width is not None and seq_width > source_width:
            raise ValueError(f"--seq-width {seq_width} exceeds source width {source_width} for {split_file}")
        print(f"Emitting {seq_len:,} tokens/document (source has {source_width:,})")

        split_group = root.create_group(split_out)
        total_tokens = num_sequences * seq_len

        # One output chunk per source row-chunk, so writes stay chunk-aligned.
        encoded_tokens = split_group.create_dataset(
            "encoded_tokens", shape=(total_tokens,), chunks=(chunk_rows * seq_len,), dtype=np.uint32
        )
        seq_starts = split_group.create_dataset("seq_starts", shape=(num_sequences + 1,), dtype=np.uint64)

        max_token_id = 0

        for i in tqdm(range(0, num_sequences, chunk_rows), desc=f"Converting {split_file}", unit="rowchunk"):
            end_idx = min(i + chunk_rows, num_sequences)

            chunk = input_array[i:end_idx, :seq_len]
            max_token_id = max(max_token_id, int(chunk.max()))

            flat_chunk = chunk.reshape(-1)

            # flat-tokens encoding: token_id * 2 + 1 at a sequence start, else token_id * 2.
            starts_mask = np.zeros_like(flat_chunk, dtype=bool)
            starts_mask[::seq_len] = True
            encoded = flat_chunk.astype(np.uint32) * 2 + starts_mask

            start_pos = i * seq_len
            end_pos = end_idx * seq_len
            encoded_tokens[start_pos:end_pos] = encoded
            seq_starts[i:end_idx] = np.arange(start_pos, end_pos, seq_len)

            if delete_source and end_idx - i == chunk_rows:
                chunk_row = i // chunk_rows
                for col in range(source_width // chunk_cols):
                    chunk_path = Path(input_array.store.path) / f"{chunk_row}.{col}"
                    if chunk_path.exists():
                        chunk_path.unlink()

        seq_starts[-1] = total_tokens
        split_group.attrs["max_token_id"] = max_token_id

        print(f"Finished {split_out}: {num_sequences:,} sequences, {total_tokens:,} tokens, max_token_id {max_token_id}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="dir containing train.zarr and heldout.zarr")
    p.add_argument("--output", required=True, help="destination flat-tokens zarr group")
    p.add_argument("--seq-width", type=int, default=None, help="keep only the first N tokens of each document")
    p.add_argument(
        "--delete-source", action="store_true", help="delete source chunks as consumed (destroys the input)"
    )
    args = p.parse_args()
    create_flat_tokens_dataset(args.input, args.output, args.seq_width, args.delete_source)


if __name__ == "__main__":
    main()

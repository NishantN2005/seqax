"""Convert LongCrawl64 (https://manifestai.com/articles/longcrawl64/) to flat-tokens format.

Assumes that LongCrawl64 was downloaded using the following command:
```
GSUTIL_PARALLEL_THREAD_COUNT=5 GSUTIL_PARALLEL_PROCESS_COUNT=5 gsutil -m cp -r gs://longcrawl64/*.zarr /your/path/to/longcrawl64/
```
See docs/flat-tokens.md for details on the format.

Note that the LongCrawl64 is pre-tokenized using GPT-2's tokenizer, where the zero token corresponds to '!'.
In seqax the zero token is a special token, so we map every instance of 0 to 50256 using remap_zero_token.py.
"""

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
    # Print size information
    num_bytes = np.prod(z.shape) * z.dtype.itemsize
    print(f"Theoretical uncompressed size: {num_bytes / (1024**3):.2f} GB")
    print(f"Size on disk: {z.nbytes_stored / (1024**3):.2f} GB")
    return z


def create_flat_tokens_dataset(input_dir, output_dir):
    """
    Convert 2D uint16 zarr arrays to flat-tokens format.

    Args:
        input_dir: Path to directory containing train.zarr and heldout.zarr
        output_dir: Path where the new zarr group should be created
    """
    # Create the root group
    root = zarr.open_group(output_dir, mode="w")

    # Process train and heldout (validation) sets
    split_mapping = {"train": "train.zarr", "validation": "heldout.zarr"}

    for split_out, split_file in split_mapping.items():
        print(f"\nProcessing {split_file} -> {split_out}...")

        # Open input array
        input_path = Path(input_dir) / split_file
        input_array = inspect_zarr_array(input_path)

        # Get array details
        num_sequences = input_array.shape[0]
        seq_len = input_array.shape[1]  # 65536
        chunk_rows = input_array.chunks[0]  # 2048
        chunk_cols = input_array.chunks[1]  # 2048

        # Create the split group
        split_group = root.create_group(split_out)

        # Calculate dimensions for the new format
        total_tokens = num_sequences * seq_len

        # Create encoded_tokens array (uint32)
        # Use chunk size that aligns with original 2048x2048 chunks
        encoded_tokens = split_group.create_dataset(
            "encoded_tokens", shape=(total_tokens,), chunks=(chunk_rows * chunk_cols,), dtype=np.uint32
        )

        # Create seq_starts array (uint64)
        seq_starts = split_group.create_dataset("seq_starts", shape=(num_sequences + 1,), dtype=np.uint64)

        # Track max_token_id across all chunks
        max_token_id = 0

        # Process data in chunks matching the source chunking
        for i in tqdm(range(0, num_sequences, chunk_rows), desc=f"Processing {split_file}"):
            # Process rows in chunks of 2048
            end_idx = min(i + chunk_rows, num_sequences)

            # Read full rows for this chunk
            chunk = input_array[i:end_idx, :]

            # Update max_token_id
            chunk_max = int(chunk.max())
            max_token_id = max(max_token_id, chunk_max)

            # Reshape chunk to 1D array
            flat_chunk = chunk.reshape(-1)

            # Create array indicating which positions are sequence starts
            starts_mask = np.zeros_like(flat_chunk, dtype=bool)
            starts_mask[::seq_len] = True

            # Encode tokens according to flat-tokens format:
            # token_id * 2 + 1 for sequence starts
            # token_id * 2 for other positions
            encoded = flat_chunk.astype(np.uint32) * 2 + starts_mask

            # Write to encoded_tokens
            start_pos = i * seq_len
            end_pos = end_idx * seq_len
            encoded_tokens[start_pos:end_pos] = encoded

            # Update seq_starts
            seq_starts[i:end_idx] = np.arange(start_pos, end_pos, seq_len)

            # Delete processed chunk files if we processed a full chunk
            if end_idx - i == chunk_rows:  # Only delete if we processed a full chunk
                # Calculate which chunk files to delete
                chunk_row = i // chunk_rows  # Integer division to get chunk row number
                # Delete all column chunks for this row
                num_col_chunks = seq_len // chunk_cols  # 65536 // 2048 = 32 columns
                for col in range(num_col_chunks):
                    chunk_file = f"{chunk_row}.{col}"
                    chunk_path = Path(input_array.store.path) / chunk_file
                    if chunk_path.exists():
                        chunk_path.unlink()

            print(f"Processed sequences {i:,} to {end_idx:,}, max_token_id so far: {max_token_id}")

        # Set the final seq_start
        seq_starts[-1] = total_tokens

        # Set max_token_id attribute
        split_group.attrs["max_token_id"] = max_token_id

        print(f"Finished {split_out}")
        print(f"Total sequences: {num_sequences:,}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Final max_token_id: {max_token_id}")


if __name__ == "__main__":
    INPUT_DIR = "/your/path/to/longcrawl64"
    OUTPUT_DIR = "/your/path/to/longcrawl64-flat-tokens.zarr"
    create_flat_tokens_dataset(INPUT_DIR, OUTPUT_DIR)

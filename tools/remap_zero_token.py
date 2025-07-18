"""Remap zeros in a flat-tokens dataset in-place."""

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

        for start_idx in tqdm(range(0, total_tokens, chunk_size)):
            end_idx = min(start_idx + chunk_size, total_tokens)
            chunk = encoded_tokens[start_idx:end_idx]
            # Modify chunk in-place
            new_chunk = np.where(chunk == 0, REMAP_TO * 2, np.where(chunk == 1, REMAP_TO * 2 + 1, chunk))
            encoded_tokens[start_idx:end_idx] = new_chunk


if __name__ == "__main__":
    ZARR_DIR = "/your/path/to/longcrawl64-flat-tokens.zarr"
    remap_zeros_in_flat_tokens_dataset(ZARR_DIR)

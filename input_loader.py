"""Input data loading from `flat-tokens` data format.

See `docs/flat-tokens.md` for details on the format.
See `docs/shuffling_loader.md` for a diagram explaining the shuffling algorithm.

We support shuffling of the input data, by the following algorithm:
* there are N independent "streams" of data, each of which has disjoint data and is
  shuffled independently.
* within each stream, we fetch a "shuffle buffer" consisting of many "read blocks" of
  data. We shuffle the entire buffer in memory.
* the "read blocks" attached to each shuffle buffer are themselves selected randomly.
* the shuffle buffer operates on sequences of a specified sequence length
  `params.dataset_seqlen`, keeping those contiguous sequences of tokens together
  and unshuffled. The `dataset_seqlen` may be larger than the `seqlen` that is used
  for training: the data loader performs the necessary reshaping to "re-pack" from
  a long `dataset_seqlen` into a shorter `seqlen`. Thus the order in which training
  visits the data depends only on `dataset_seqlen` and not on `seqlen`, allowing you
  to sweep over different context lengths while still visiting the data in the same
  random order.

This is the standard shuffling used by e.g. Huggingface Datasets. Unlike them, we run
this algorithm _after_ tokenization, so we know exactly at which step number each new
shuffle buffer starts at, allowing us to do instant resumes after job restarts. In our
default recommended configuration, we also recommend a much larger shuffle buffer size
than Huggingface Datasets, which allows for more thorough shuffling, taking advantage
of the fact that a single sequence of tokens uses very little memory compared to e.g.
a single image.

Mosaic's StreamingDatasets library uses a similar algorithm as us, which they call py1b:
https://docs.mosaicml.com/projects/streaming/en/stable/fundamentals/shuffling.html.
"""

import datetime
import functools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Tuple

import jax
import numpy as np
import numpy.typing as npt
import zarr
from typeguard import typechecked

import shardlib.shardtypes as shardtypes
from shardlib.shardtypes import bool_, pytree_dataclass, u32


@dataclass(frozen=True)
class TokenBatchParams:
    """The shape of a token batch."""

    len: int
    batch: int


@pytree_dataclass
class TokenBatch:
    """A batch of tokens, which are typically the input to training."""

    targets: u32[b"batch/d len/s"]
    is_seq_start: bool_[b"batch/d len/s"]


@dataclass(frozen=True)
class FlatTokensParams:
    filespec: str

    # A "stream" is what's attached to one independent shuffle buffer. There may be multiple
    # independent shuffle buffers, allowing parallelism.
    #
    # A "minipoch" (mini-epoch) is the set of sequences visited by one global refill of shuffle
    # buffers. The last minipoch may be shorter than others, but each stream in the last minipoch
    # must have the same number of read blocks, which must also be an integer.
    #
    # (To minimize discarded data on very small training sets, set streams=1 and make
    # sequences_per_read_block small.)
    #
    # Shuffling transforms the uint32[num_tokens] into uint32[streams, sequences, len], the
    # "shuffled tokens". We then form batches by a transformation on [streams, sequences].

    streams: int  # Recommended: maximum number of hosts you expect to use.
    read_blocks_per_shuffle_buffer: int  # Recommended: 1 << 10. 4GiB (uncompressed) shuffle buffer.
    sequences_per_read_block: int  # Recommended: (1 << 20) / len. 1MiB (compressed) read block.
    dataset_seqlen: int
    seed: int
    sequence_packing: bool
    eval_tokens: int


@dataclass
class _ShuffleBuffer:
    minipoch: int
    buffer: u32[b"Buflen dlen"]


class ShufflingLoader:
    def __init__(self, split: str, params: FlatTokensParams, training_batch_params: TokenBatchParams):
        # We run the data loader with a particular target seqlen, given by params.dataset_seqlen.
        # This is allowed to be larger than the training_batch_params.len, to allow users to sweep
        # over different seqlens while still visiting the training data in the same random order.
        #
        # We accommodate that by running with an adjusted TokenBatchParams (using the dataset_seqlen),
        # and at the very last moment (after reading from the shuffle buffer) reshape back to the
        # training batch params.
        self.params = params
        self.training_batch_params = training_batch_params
        self.training_sequences_per_dataset_sequence = _div_exact(params.dataset_seqlen, training_batch_params.len)
        dataset_batch_params = TokenBatchParams(
            len=params.dataset_seqlen,
            batch=_div_exact(training_batch_params.batch, self.training_sequences_per_dataset_sequence),
        )
        self.dataset_batch_params = dataset_batch_params
        self.root = zarr.open_group(params.filespec, mode="r")
        assert split in ["train", "validation"], "Invalid split"
        self.encoded_tokens = self.root[split]["encoded_tokens"]
        self.encoded_tokens_fetcher = UncompressedDataZarrFetcher(self.encoded_tokens)
        self.seq_starts = self.root[split]["seq_starts"]
        self.max_token_id = self.root[split].attrs["max_token_id"]
        assert len(self.encoded_tokens.shape) == 1, "Expected 1D zarr"
        assert self.encoded_tokens.dtype == np.uint32, "Expected uint32 zarr"
        assert len(self.seq_starts.shape) == 1, "Expected 1D zarr"
        assert self.seq_starts.dtype == np.uint64, "Expected uint64 zarr"

        token_count = self.encoded_tokens.shape[0]
        if params.sequence_packing:
            self.seq_count = token_count // dataset_batch_params.len
        else:
            self.seq_count = self.seq_starts.shape[0] - 1

        # Count read blocks. Round it down to a multiple of streams
        read_block_count = self.seq_count // params.sequences_per_read_block
        read_block_count = (read_block_count // params.streams) * params.streams
        self.read_block_count = read_block_count
        assert read_block_count > 0, (
            "Must have at least one read block per stream. Try shrinking streams and sequences_per_read_block."
        )
        self.step_count = (read_block_count * params.sequences_per_read_block) // dataset_batch_params.batch
        # Count minipochs
        self.minipoch_count = _div_up(read_block_count, params.streams * params.read_blocks_per_shuffle_buffer)
        self.seq_indices_per_shuffle_buffer = params.read_blocks_per_shuffle_buffer * params.sequences_per_read_block
        # Calculate batch->stream mapping.
        self.batch_indices_per_stream = _div_exact(dataset_batch_params.batch, params.streams)
        # Calculate which streams and which batch indices this host is responsible for, based on the sharding.
        self.sharding = shardtypes.make_shardings(TokenBatch).targets
        streams = set()
        batch_indices = set()
        for batch_slices, _ in self.sharding.addressable_devices_indices_map(
            (dataset_batch_params.batch, dataset_batch_params.len)
        ).values():
            batch_lo, batch_hi, batch_step = batch_slices.indices(dataset_batch_params.batch)
            for b in range(batch_lo, batch_hi, batch_step):
                batch_indices.add(b)
                streams.add(b // self.batch_indices_per_stream)
        self.shuffle_buffers_by_stream = {stream_index: None for stream_index in streams}
        self.batch_indices = sorted(batch_indices)
        # Shuffle read blocks
        assert read_block_count < 1 << 32, (
            f"Too many read blocks. Try growing sequences_per_read_block: {read_block_count}"
        )
        self.read_block_ordering = _random_permutation(params.seed, read_block_count)

    def load(self, step: int) -> TokenBatch:
        assert step < self.step_count, (
            f"Requested step {step} but dataset only supports {self.step_count} steps at batch size {self.dataset_batch_params.batch}."
        )
        # Conceptually, we remap IDs as follows:
        # 1. (step, batch_index) -> (stream, seq_index_in_stream)
        # 2. seq_index_in_stream -> (minipoch, seq_index_in_shuffle_buffer)
        #
        # We visit all batch_indices in increasing order. Since the map batch_index->(stream, minipoch)
        # is monotonic (non-decreasing), we can reload the shuffle buffer for a stream whenever
        # we cross to a new minipoch without thrashing back and forth between adjacent minipochs.
        seq_by_batch_index = {}
        for batch_index in self.batch_indices:
            # 1. (step, batch_index) -> (stream, seq_index_in_stream)
            stream = batch_index // self.batch_indices_per_stream
            seq_index_in_stream = step * self.batch_indices_per_stream + (batch_index % self.batch_indices_per_stream)
            # 2. seq_index_in_stream -> (minipoch, seq_index_in_shuffle_buffer)
            minipoch = seq_index_in_stream // self.seq_indices_per_shuffle_buffer
            seq_index_in_shuffle_buffer = seq_index_in_stream % self.seq_indices_per_shuffle_buffer
            shuffle_buffer = self._get_shuffle_buffer(stream, minipoch)
            seq_by_batch_index[batch_index] = shuffle_buffer[seq_index_in_shuffle_buffer]

        def get_shard(indexing: Tuple[slice, ...]) -> npt.NDArray:
            seqlen_slice = indexing[1]
            examples = []
            # Here we reindex from training_batch_params to dataset_batch_params.
            for training_batch_index in range(*indexing[0].indices(self.training_batch_params.batch)):
                dataset_batch_index = training_batch_index // self.training_sequences_per_dataset_sequence
                seqlen_slice_offset = (
                    training_batch_index % self.training_sequences_per_dataset_sequence
                ) * self.training_batch_params.len
                dataset_seq = seq_by_batch_index[dataset_batch_index]
                training_seq = dataset_seq[seqlen_slice_offset : seqlen_slice_offset + self.training_batch_params.len]
                examples.append(training_seq[seqlen_slice])
            return np.stack(examples)

        shape = (self.training_batch_params.batch, self.training_batch_params.len)
        encoded_tokens = jax.make_array_from_callback(shape, self.sharding, get_shard)
        return _decode(encoded_tokens)

    def _get_shuffle_buffer(self, stream: int, minipoch: int) -> u32[b"Buflen dlen"]:
        if (
            self.shuffle_buffers_by_stream[stream] is None
            or self.shuffle_buffers_by_stream[stream].minipoch != minipoch
        ):
            self.shuffle_buffers_by_stream[stream] = None  # Free the underlying memory
            blocks_in_shuffle_buffer = self.params.read_blocks_per_shuffle_buffer
            if minipoch == self.minipoch_count - 1:
                blocks_in_shuffle_buffer = (
                    self.read_block_count // self.params.streams
                ) - self.params.read_blocks_per_shuffle_buffer * minipoch
            # We form a mapping:
            #   (stream, minipoch, read_block_in_minipoch) -> sequential_read_block
            # then we map
            #   sequential_read_block -> shuffled_read_block
            # using self.shuffled_read_blocks.
            shuffled_read_block_indices = []
            for read_block_in_minipoch in range(blocks_in_shuffle_buffer):
                sequential_read_block = (
                    minipoch * self.params.read_blocks_per_shuffle_buffer + read_block_in_minipoch
                ) * self.params.streams + stream
                shuffled_read_block = self.read_block_ordering[sequential_read_block].item()
                shuffled_read_block_indices.append(shuffled_read_block)

            # Now load all of the read blocks in parallel.
            def load_read_block(read_block_index: int) -> u32[b"Blocklen dlen"]:
                start_seq = read_block_index * self.params.sequences_per_read_block
                end_seq = start_seq + self.params.sequences_per_read_block
                block_shape = (self.params.sequences_per_read_block, self.dataset_batch_params.len)
                if self.params.sequence_packing:
                    flat_tokens = self.encoded_tokens_fetcher.fetch(
                        start_seq * self.dataset_batch_params.len, end_seq * self.dataset_batch_params.len
                    )
                    # flat_tokens = self.encoded_tokens[start_seq * self.dataset_batch_params.len : end_seq * self.dataset_batch_params.len]
                    return flat_tokens.reshape(block_shape)
                else:
                    seq_starts = self.seq_starts[start_seq : end_seq + 1]
                    flat_tokens = self.encoded_tokens_fetcher.fetch(int(seq_starts[0]), int(seq_starts[-1]))
                    seq_starts -= seq_starts[0]  # Map indices for self.encoded_tokens to indices for flat_tokens.
                    # Read the ragged array into a (padded) dense array.
                    #
                    # We pad on the left with 0s, which decode to (0, new_sequence=false).
                    result = np.zeros(block_shape, dtype=np.uint32)
                    for i in range(self.params.sequences_per_read_block):
                        start = seq_starts[i]
                        end = min(seq_starts[i + 1], start + np.uint64(self.dataset_batch_params.len))
                        result[i, np.uint64(block_shape[1]) - (end - start) :] = flat_tokens[start:end]
                    return result

            print(f"[{datetime.datetime.now()}] Loading shuffle buffer")
            import time

            start = time.time()
            # Loading a read block is IO-dominated work, with very little CPU time involved, so we can afford
            # to run a huge number of these in parallel with little concern about thrashing the CPU by having
            # excessively many threads doing CPU-intensive work. At the recommended read block sizing of 1MiB,
            # the memory footprint of a read block is typically bigger than the memory footprint of a CPU thread,
            # so we're also unlikely to waste a significant fraction of memory by having too many threads. In
            # net, allow a lot of threads, potentially way more than we have CPUs! Other overheads will
            # bite us before thread overheads do.
            with ThreadPoolExecutor(max_workers=len(shuffled_read_block_indices)) as executor:
                shuffled_read_blocks = list(executor.map(load_read_block, shuffled_read_block_indices))
            shuffle_buffer = np.concatenate(shuffled_read_blocks, axis=0)
            print(
                f"[{datetime.datetime.now()}] Finished loading shuffle buffer, {shuffle_buffer.size * 4:_} bytes, {time.time() - start:.1f}s"
            )

            # Actually shuffle it.
            sequences_in_shuffle_buffer = blocks_in_shuffle_buffer * self.params.sequences_per_read_block
            assert shuffle_buffer.shape == (sequences_in_shuffle_buffer, self.dataset_batch_params.len)
            shuffle_seed = self.params.seed + 1 + minipoch * self.params.streams + stream
            permutation = _random_permutation(shuffle_seed, sequences_in_shuffle_buffer)
            shuffle_buffer = shuffle_buffer[permutation, :]
            self.shuffle_buffers_by_stream[stream] = _ShuffleBuffer(minipoch, shuffle_buffer)

        return self.shuffle_buffers_by_stream[stream].buffer


@dataclass
class UncompressedDataZarrFetcher:
    def __init__(self, z: zarr.Array):
        # TODO: This could be generalized to all FSStore instances, not just GCS. fsspec
        # supports efficient partial reads. We could include the DirectoryStore case under
        # the fsspec abstraction too.
        self.is_fast_path = (
            z.compressor is None
            and z.filters is None
            and z.ndim == 1
            and (
                isinstance(z.store, zarr.storage.DirectoryStore)
                or (isinstance(z.store, zarr.storage.FSStore) and "gcs" in z.store.fs.protocol)
            )
        )
        self.z = z
        if not self.is_fast_path:
            print("Using default (slow path) zarr loader`")
            return
        if isinstance(z.store, zarr.storage.DirectoryStore):
            print("Using fast path (uncompressed local) zarr loader")
            zgroup_root_path = z.store.path
            max_workers = 64  # Seems fine on my MacBook with an SSD. Probably similar on other SSD machines.
        else:
            print("Using fast path (uncompressed GCS) zarr loader")
            from google.cloud import storage

            client = storage.Client()
            bucket_name, zgroup_root_path = z.store.path.split("/", maxsplit=1)
            self.bucket = client.bucket(bucket_name)
            # GCS client library limits us to 10 concurrent HTTP connections.
            # TODO: Perhaps direct HTTP2 access to the REST API would bypass this limit?
            max_workers = 10

        zarr_path = zgroup_root_path + "/" + z.path
        self.path = zarr_path
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def fetch(self, start: int, end: int) -> np.ndarray:
        if not self.is_fast_path:
            return self.z[start:end]

        element_bytes = self.z.dtype.itemsize
        start_bytes = start * element_bytes
        end_bytes = end * element_bytes
        chunk_len_bytes = self.z.chunks[0] * element_bytes

        def _fetch(chunk_i):
            path = self.path + "/" + str(chunk_i)
            start = max(0, start_bytes - chunk_i * chunk_len_bytes)
            end = min(chunk_len_bytes, end_bytes - chunk_i * chunk_len_bytes)
            if isinstance(self.z.store, zarr.storage.DirectoryStore):
                with open(path, "rb") as f:
                    f.seek(start)
                    bytes = f.read(end - start)
            else:
                bytes = self.bucket.blob(path).download_as_bytes(
                    start=start,
                    end=end - 1,
                    checksum=None,  # It's not possible checksum for partial downloads. Suppress warning.
                )
            return np.frombuffer(bytes, dtype=self.z.dtype)

        chunks = list(
            self.executor.map(_fetch, range(start_bytes // chunk_len_bytes, _div_up(end_bytes, chunk_len_bytes)))
        )
        return np.concatenate(chunks)


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def _div_exact(a: int, b: int) -> int:
    assert a % b == 0
    return a // b


@functools.partial(jax.jit, donate_argnums=(0,))
@typechecked
def _decode(encoded_tokens: u32[b"batch/d len/s"]) -> TokenBatch:
    # encoded_tokens encoding:
    #  2*id+1 for the first token in a sequence
    #  2*id for other tokens in the sequence
    targets = encoded_tokens >> 1
    is_seq_start = (encoded_tokens & 1) == 1
    return TokenBatch(targets, is_seq_start)


def _random_permutation(seed: int, n: int) -> npt.NDArray[np.uint32]:
    """Same as `np.random.Generator.permutation`, but with a guarantee that it will always produce the same results for a given seed."""
    assert n < 1 << 32
    # We do a Fisher-Yates shuffle using the Philox BitGenerator. Unlike the rest of np.random,
    # which is documented as potentially changing between numpy versions or even platforms on
    # the same version, the Philox BitGenerator is documented as stable. Likewise, we also promise
    # not to change the following implementation of the Fisher-Yates shuffle.
    #
    # We calculate the random numbers using `random_uint64() % n` rather than using rejection
    # sampling to generate numbers in range `[0, n)`. (Rejection sampling is more complicated,
    # because we don't know up front how many random numbers we'll need.) Our approach
    # introduces some bias, but it's small: since n<2^32, the bias is at most 2^-32 for each
    # random number generated. We're fine with this.
    randoms = np.random.Philox(seed).random_raw(n) % (np.arange(n, dtype=np.uint64) + 1)
    result = np.arange(n, dtype=np.uint32)
    for i in reversed(range(n)):
        j = randoms[i]
        tmp = result[i]
        result[i] = result[j]
        result[j] = tmp
    return result

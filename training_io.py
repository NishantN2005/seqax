"""Provides IO support for training:
* checkpoint save and load
* metrics logging
* profiling of XLA computations
* reporting FLOPs per device
"""

import concurrent
import dataclasses
import datetime
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, fields, is_dataclass
from threading import Thread
from typing import Any, List, Optional, OrderedDict, Tuple

import fsspec
import gcsfs
import jax
import jax.numpy as jnp
import jax.profiler
import numpy as np
import zarr
import zarr.errors
from google.cloud import firestore
from jax.experimental import multihost_utils
from jax.lib import xla_extension
from numcodecs import blosc

PyTree = Any


def clear_tpu_locks():
    """
    If the previous job was not cleaned up properly, we need to kill the previous
    processes that were using the TPU and remove the lock file.

    The approach is based on logic found in:
    https://github.com/google/maxtext/blob/fe0dee10de708631088a4d51a35969bd70708c4e/multihost_runner.py#L161-L182
    """
    try:
        raw_pids = subprocess.run(["lsof", "-w", "/dev/accel0"], capture_output=True, text=True).stdout
        pids = set()
        for line in raw_pids.splitlines()[1:]:
            parts = line.split()
            if len(parts) > 1:
                pids.add(parts[1])
        for pid in pids:
            os.kill(int(pid), signal.SIGTERM)
        if pids:
            os.remove("/tmp/libtpu_lockfile")
    except Exception as e:
        print(f"Error clearing TPU locks: {e}")
        pass


@dataclass(frozen=True)
class IOConfig:
    """Configuration for IO operations.

    Args:
        max_io_threads: Maximum threads for IO tasks like checkpointing.
            1MiB/thread typical, 1024 threads reasonable for 1GiB overhead.
            Can exceed CPU cores since IO-bound.
    """

    max_io_threads: int


def print_attributes(attributes: dict[str, Any]):
    """Prints attributes to stdout."""
    if jax.process_index() == 0:
        for k, v in attributes.items():
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] Logging single value {k}: {v}")


def log_attributes(zarr_logger: "ZarrLog", firestore_logger: "FirestoreLog", attributes: dict[str, Any]):
    """Logs attributes to all loggers. If a logger is None, it is skipped."""
    print_attributes(attributes)
    if zarr_logger:
        zarr_logger.add_attributes(attributes)
    if firestore_logger:
        firestore_logger.add_attributes(attributes)


def print_training_step(step: int, output: PyTree):
    """Prints training step to stdout."""
    if jax.process_index() == 0:
        metrics_dict = {}
        output_no_stats = dataclasses.replace(output, tensor_stats=None)
        for path, arr in jax.tree_util.tree_leaves_with_path(output_no_stats):
            path = jax.tree_util.keystr(path)
            arr = jax.device_get(arr)
            if arr.shape == () and arr.dtype == jnp.float32:
                metrics_dict[path] = float(arr)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Step {step}: {metrics_dict}")


def log_training_step(zarr_logger: "ZarrLog", firestore_logger: "FirestoreLog", step: int, output: PyTree):
    """Logs training step to all loggers. If a logger is None, it is skipped."""
    print_training_step(step, output)
    if zarr_logger:
        zarr_logger.log_training_step(step, output)
    if firestore_logger:
        firestore_logger.log_training_step(step)


def log_finalize(zarr_logger: "ZarrLog", firestore_logger: "FirestoreLog"):
    """Finalizes logging for all loggers. If a logger is None, it is skipped."""
    if zarr_logger:
        zarr_logger.finalize()
    if firestore_logger:
        firestore_logger.finalize()


def _convert_dataclass_to_flat_dict(config):
    """Convert a dataclass to a flat dictionary with dot-separated keys."""
    dic = {}

    def recurse(x, path, dic):
        if is_dataclass(x):
            for i in fields(x):
                recurse(getattr(x, i.name), path + "." + i.name, dic)
        else:
            dic[path[1:]] = x  # remove leading '.'

    recurse(config, "", dic)
    return dic


@dataclass
class Loggers:
    """Container for different logging systems."""

    firestore: "FirestoreLog"
    zarr: "ZarrLog"


class FirestoreLog:
    use_firestore: bool
    project_name: str
    db: firestore.Client
    project_ref: firestore.CollectionReference
    document_ref: firestore.DocumentReference
    time_stamp_last_reported: int
    max_steps: int

    def __init__(self, project_name: str, config, create: bool, use_firestore: bool, gcp_project: str):
        """After restarting from a checkpoint, use create=False to _not_ write (which would alter time_started)."""
        self.use_firestore = use_firestore
        if jax.process_index() == 0 and self.use_firestore:
            self.project_name = project_name
            self.db = firestore.Client(project=gcp_project)
            self.project_ref = self.db.collection("projects")
            self.document_ref = self.project_ref.document(self.project_name)
            now = time.time_ns()
            self.time_stamp_last_reported = now
            self.max_steps = config.optimizer.steps
            if create:
                dic = _convert_dataclass_to_flat_dict(config)
                self.add_attributes(
                    {
                        "config": dic,
                        "time_started": now,
                        "time_last_updated": now,
                        "time_ended": None,
                        "last_valid_index": None,
                    }
                )

    def add_attributes(self, dic: dict[str, Any]):
        """Merge the dictionary dic into the Firestore document."""
        if jax.process_index() == 0 and self.use_firestore:
            # Always merge dic into Firestore document, because it might already exist (if we restarted from checkpoint).
            self.document_ref.set(dic, merge=True)

    def log_training_step(self, step: int):
        """Update the step number in Firestore, not more often than all 10 seconds."""
        if jax.process_index() == 0 and self.use_firestore:
            now = time.time_ns()
            if self.time_stamp_last_reported + 10e9 <= now:  # write not more often than once per 10 sec (1 s = 1e9 ns)
                self.time_stamp_last_reported = now
                self.add_attributes({"last_valid_index": step, "time_last_updated": now})

    def finalize(self):
        """Write the end time."""
        if jax.process_index() == 0 and self.use_firestore:
            now = time.time_ns()
            self.add_attributes({"last_valid_index": self.max_steps - 1, "time_last_updated": now, "time_ended": now})


class ZarrLog:
    """Create and update a log file as outlined in the specification in docs/zarr_log.md."""

    @dataclass
    class _WriteBufferHelper:
        values: np.array
        group: Optional[zarr.Group]
        names: Optional[List[str]] = None

    root: zarr.Group
    training_group: zarr.Group
    write_buffer: OrderedDict[str, _WriteBufferHelper]

    last_valid_index: int
    last_written_valid_index: int

    frequencies: List[int]
    flush_frequency: int
    max_steps: int
    groups_initialized: bool
    max_io_threads: int
    zarr_log_path: str

    flush_thread: Optional[Thread]

    def __init__(self, zarr_log_path: str, config, cloning: bool = False):
        """Write the contents of config as key-value-pairs to the newly created zarr file zarr_log_path.

        If zarr_log_path exists, load it and continue adding to it (e.g., when restarting from a checkpoint).
        """
        if jax.process_index() == 0:
            if config.clone_from and not cloning:
                source_zarr_log = ZarrLog(
                    os.path.join(config.root_working_dir, config.clone_from, "log.zarr"), config, True
                )
                assert source_zarr_log.write_buffer, "Cannot clone from an empty Zarr log"
            self.max_steps = config.optimizer.steps
            self.flush_frequency = config.zarr_log_flush_frequency
            self.max_io_threads = config.io.max_io_threads

            log4_steps = np.log(self.max_steps) / np.log(4)
            self.frequencies = tuple(4**i for i in range(int(log4_steps) + 3) if 4**i <= self.max_steps)
            self.last_valid_index = -1
            self.last_written_valid_index = -1
            self.write_buffer = OrderedDict()
            self.groups_initialized = False
            self.zarr_log_path = zarr_log_path
            self.flush_thread = None
            try:  # open and read assuming it exists with present training data (to allow restarting from a checkpoint)
                self.root = zarr.open_group(zarr_log_path, mode="r+")
                if "eval_loss" in self.root.attrs.asdict() and not cloning:
                    raise RuntimeError("Training is already finished, aborting.")
                self.training_group = self.root["training"]
                self.last_written_valid_index = self.training_group.attrs.get("last_valid_index", -1)
                for groupname, group in self.training_group.groups():
                    # [:] converts to local numpy array
                    self.write_buffer[groupname] = self._WriteBufferHelper(
                        values=group["1"][:], group=group, names=group.attrs["names"]
                    )
                self.write_buffer["timestamp"] = self._WriteBufferHelper(
                    self.training_group["timestamp"][:], self.training_group
                )
                self.groups_initialized = True
            except FileNotFoundError:
                self.root = zarr.open_group(zarr_log_path, mode="w")  # maybe overwrite
                self.training_group = self.root.create_group("training")
                config_group = self.root.create_group("config")
                dic = _convert_dataclass_to_flat_dict(config)
                config_group.attrs.put(dic)

                if config.clone_from and not cloning:
                    self._copy_write_buffer_contents(source_zarr_log)

    def add_attributes(self, attr: dict):
        """Add root level attributes from a dictionary."""
        if jax.process_index() == 0:
            self.root.attrs.update(attr)

    def log_training_step(self, step: int, output: PyTree):
        """Logs the output of a training step. The output must be a PyTree of f32 arrays.

        This write is buffered and a call to .flush() (which in turn calls ._flush_function()) will perform the actual write, this happens at most all zarr_log_flush_frequency steps and in .finalize().
        """
        if jax.process_index() == 0:
            if "metrics" not in self.write_buffer:
                num_metrics = 0
                names = []
                for path, arr in jax.tree_util.tree_leaves_with_path(output):
                    # Some names in the path were DictKey, some strings
                    metric = ".".join([p.key if isinstance(p, jax.tree_util.DictKey) else p for p in path])
                    arr = jax.device_get(arr)
                    if arr.shape == () and arr.dtype == jnp.float32:
                        num_metrics += 1
                        names.append(metric)
                    elif len(arr.shape) == 1 and arr.dtype == jnp.float32:
                        num_metrics += len(arr)
                        for j in range(len(arr)):
                            names.append(f"{metric}.{j}")
                    elif arr.dtype == jnp.float32:
                        print(f"WARNING: unimplemented zarr storing of {metric}: {arr}")
                    else:
                        raise ValueError(f"Output {metric} has unsupported shape {arr.shape} and dtype {arr.dtype}.")
                values = np.full((num_metrics, self.max_steps), np.nan, np.float32)
                self.write_buffer["metrics"] = self._WriteBufferHelper(values, group=None, names=names)
            i = 0
            for path, arr in jax.tree_util.tree_leaves_with_path(output):
                arr = jax.device_get(arr)
                if arr.shape == () and arr.dtype == jnp.float32:
                    self.write_buffer["metrics"].values[i, step] = float(arr)
                    i += 1
                elif len(arr.shape) == 1 and arr.dtype == jnp.float32:
                    for j in range(len(arr)):
                        self.write_buffer["metrics"].values[i, step] = float(arr[j])
                        i += 1
                elif arr.dtype == jnp.float32:
                    print(f"WARNING: unimplemented zarr storing of {metric}: {arr}")
                else:
                    raise ValueError(f"Output {metric} has unsupported shape {arr.shape} and dtype {arr.dtype}.")
            if "timestamp" not in self.write_buffer:
                values = np.zeros(self.max_steps, np.int64)
                self.write_buffer["timestamp"] = self._WriteBufferHelper(values, group=self.training_group)
            self.write_buffer["timestamp"].values[step] = time.time_ns()
            self.last_valid_index = step
            if step <= self.last_written_valid_index:
                # this can only happen if we restarted from a checkpoint
                # but the flush_frequency and the checkpoint_interval are such that we have old log file contents
                # which got invalidated by continuing the calculation from an earlier state
                self.last_written_valid_index = step - 1
            if step - self.last_written_valid_index >= self.flush_frequency:
                self.flush()

    def _initialize_training_group(self):
        """The first .flush_function() creates the zarr training groups and adds them to the write_buffer."""
        if jax.process_index() == 0 and not self.groups_initialized:
            if "timestamp" in self.write_buffer:
                self.write_buffer["timestamp"].group = self.training_group
                self.write_buffer["timestamp"].group.zeros(name="timestamp", shape=self.max_steps, dtype=np.int64)

            if "metrics" in self.write_buffer:
                names = self.write_buffer["metrics"].names
                group = self.training_group.create_group("metrics")
                group.attrs.put({"names": names})
                for f in self.frequencies:
                    group.full(
                        name=str(f), shape=(len(names), self.max_steps // f), fill_value=np.nan, dtype=np.float32
                    )
            self.groups_initialized = True

    def _copy_write_buffer_contents(self, source_zarr_log: "ZarrLog"):
        for key in source_zarr_log.write_buffer:
            values = source_zarr_log.write_buffer[key].values
            names = source_zarr_log.write_buffer[key].names
            new_shape = list(values.shape[:-1]) + [self.max_steps]
            new_values = np.zeros(new_shape, dtype=values.dtype)
            new_values[..., : min(self.max_steps, values.shape[-1])] = values[
                ..., : min(self.max_steps, values.shape[-1])
            ]
            self.write_buffer[key] = self._WriteBufferHelper(values=new_values, group=None, names=names)

    def log_eval_losses(self, eval_losses: np.array):
        """Log the eval losses."""
        if jax.process_index() == 0:
            self.root.create_dataset("eval_losses", data=eval_losses, shape=eval_losses.shape)

    # Note that we have
    # mean(a01,a02,a03,a04,a05,a06,a07,a08,a09,a10,a11,a12,a13,a14,a15,a16)
    # =
    # mean(mean(a01,a02,a03,a04),mean(a05,a06,a07,a08),mean(a09,a10,a11,a12),mean(a13,a14,a15,a16))
    # i.e., we could refine stepwise (from 1 to 4 and from 4 to 16 ...) but here, each thread shall be independent of each other.

    def flush(self, wait_for_previous_flush: bool = False, wait_for_this_flush: bool = False):
        """Flush the write buffer to the persistent storage.

        If wait_for_previous_flush is True, the function waits for the previous flush to finish before starting a new one.
        If wait_for_this_flush is True, the function waits for the current flush to finish before returning.
        """
        if jax.process_index() == 0:
            if self.flush_thread and self.flush_thread.is_alive():
                if wait_for_previous_flush:
                    self.flush_thread.join()
                else:
                    return
            from_index = self.last_written_valid_index + 1
            to_index = self.last_valid_index
            self.flush_thread = Thread(
                target=self._flush_function, kwargs=dict(from_index=from_index, to_index=to_index)
            )
            self.flush_thread.start()
            self.last_written_valid_index = self.last_valid_index
            if wait_for_this_flush:
                self.flush_thread.join()

    def _flush_function(self, from_index: int, to_index: int):
        """Write the values buffered by log_training_step() to the persistent zarr file.

        Here the frequency is taken into account to average 1, 4, 16, ... consecutive values.
        """
        if jax.process_index() == 0:
            self._initialize_training_group()

            def _flush_metrics(self, findex: int):
                values = self.write_buffer["metrics"].values
                thisgroup = self.training_group["metrics"]
                f = self.frequencies[findex]
                freq_next_index_to_write = from_index // f
                # the last index to write considers the largest possible multiple of f plus one iff f-1 entries afterwards are also valid
                freq_last_valid_index = (to_index - f + 1) // f
                if freq_next_index_to_write <= freq_last_valid_index:
                    to_write = values[:, f * freq_next_index_to_write : f * (freq_last_valid_index + 1)]
                    to_write = np.reshape(to_write, (len(values), -1, f))
                    to_write = np.mean(to_write, axis=2)
                    thisgroup[str(f)][:, freq_next_index_to_write : freq_last_valid_index + 1] = to_write

            def _flush_timestamp(self):
                values = self.write_buffer["timestamp"].values
                thisgroup = self.write_buffer["timestamp"].group
                if from_index <= to_index:
                    thisgroup["timestamp"][from_index : to_index + 1] = values[from_index : to_index + 1]

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_io_threads) as executor:
                futures = []
                futures.append(executor.submit(_flush_timestamp, self))
                for findex in range(len(self.frequencies)):
                    futures.append(executor.submit(_flush_metrics, self, findex))
                for future in futures:
                    future.result()

            self.training_group.attrs["last_valid_index"] = to_index

    def finalize(self):
        """Flush the write buffer to the persistent storage and consolidate.

        Calling as __del__ leads to "RuntimeError: cannot schedule new futures after interpreter shutdown".
        """
        if jax.process_index() == 0:
            self.flush(wait_for_previous_flush=True, wait_for_this_flush=True)
            del self.training_group.attrs["last_valid_index"]
            # with zarr==3.0.6:
            # UserWarning: Consolidated metadata is currently not part in the Zarr format 3 specification. It may not be supported by other zarr implementations and may change in the future.
            # see also: https://zarr.readthedocs.io/en/stable/user-guide/consolidated_metadata.html
            # zarr.consolidate_metadata(self.zarr_log_path)


def _split_fsspec(path: str) -> Tuple[str, str]:
    """Splits a fsspec path into the filesystem and the path within the filesystem."""
    f = fsspec.open(path)
    fs = f.fs
    path = f.path
    return fs, path


def load_checkpoint_if_it_exists(
    checkpoint_dir: str, state: PyTree, config: IOConfig, index: int = -1
) -> Tuple[PyTree, int]:
    """Loads the latest checkpoint if it exists, otherwise return the initial state.

    In either case, uses the sharding and PyTree structure of `state` to produce the output.

    Since the state may occupy a large amount of memory, this function makes sure to delete `state`
    before loading the checkpoint. To facilitate this, callers should ensure not to hold on to any
    additional references to `state` when calling this function.

    Returns state and step number. Step 0 is the initial state, which may or may not have been loaded
    from a checkpoint.
    """
    blosc.use_threads = False  # Blindly following recommendation from https://zarr.readthedocs.io/en/stable/tutorial.html#parallel-computing-and-synchronization
    fs, checkpoint_dir_path = _split_fsspec(checkpoint_dir)

    def write_completed(checkpoint: str) -> bool:
        try:
            if isinstance(fs, gcsfs.core.GCSFileSystem):
                checkpoint = f"gs://{checkpoint}"
            root = zarr.open(checkpoint, mode="r+")
            return "write_completed" in root.attrs
        except Exception:
            return False

    # Check working_dir for checkpoint files.
    # Process index 0 selects the checkpoint, then broadcasts it to everyone else.
    selected_checkpoint = -1
    if jax.process_index() == 0:
        if index >= 0:
            if write_completed(os.path.join(checkpoint_dir_path, step_to_str(index))):
                selected_checkpoint = index
            else:
                raise ValueError(f"No completed checkpoint found at index {index} under {checkpoint_dir_path}.")
        elif fs.exists(checkpoint_dir_path):
            checkpoint_dirs = fs.ls(checkpoint_dir_path)
            checkpoint_dirs = [(int(os.path.basename(c)), c) for c in checkpoint_dirs if os.path.basename(c).isdigit()]
            for checkpoint_number, c in reversed(sorted(checkpoint_dirs)):
                if not write_completed(c):
                    print(f"zarr 'write_completed' marker is missing in checkpoint {c}; skipping.")
                    continue
                selected_checkpoint = checkpoint_number
                break
    selected_checkpoint = multihost_utils.broadcast_one_to_all(jnp.int32(selected_checkpoint))

    if selected_checkpoint == -1:
        print(f"No checkpoints found in {checkpoint_dir_path}, starting from initial state.")
        return state, 0

    print(f"Found checkpoint {selected_checkpoint} in {checkpoint_dir_path}, starting from there.")
    return load_zarr(os.path.join(checkpoint_dir, step_to_str(selected_checkpoint)), state, config), int(
        selected_checkpoint
    )


def save_checkpoint(checkpoint_dir: str, step: int, state: PyTree, config: IOConfig):
    """Saves a checkpoint for the specified step number.

    See docs/pytree-zarr-checkpoint.md for the checkpoint format.
    """
    blosc.use_threads = False
    checkpoint_file = os.path.join(checkpoint_dir, step_to_str(step))
    if jax.process_index() == 0:
        # If there's already a checkpoint at this step, delete it. It might have been a partially
        # written checkpoint from a previous run.
        f = fsspec.open(checkpoint_dir)
        checkpoint_path = os.path.join(f.path, step_to_str(step))
        if f.fs.exists(checkpoint_path):
            f.fs.rm(checkpoint_path, recursive=True)

    print(f"[{datetime.datetime.now()}] Saving checkpoint {step} to {checkpoint_file}.")
    save_zarr(checkpoint_file, state, config)
    print(f"[{datetime.datetime.now()}] Finished saving checkpoint {step} to {checkpoint_file}.")


def load_zarr(filename: str, state: PyTree, config: IOConfig, path_prefix: str = "") -> PyTree:
    """Loads a zarr checkpoint from disk.

    See docs/pytree-zarr-checkpoint.md for the checkpoint format.
    To load the model and optimizer state, use the default `path_prefix=""`.
    To load the model state only, use `path_prefix="weights"`. In this case, `state` has to be the PyTree coming from the Model object, and not from the full State object.
    """
    root = zarr.open_group(filename, mode="r")
    if "write_completed" not in root.attrs:
        raise ValueError(
            "zarr 'write_completed' marker is missing. Should not have selected this checkpoint to load from."
        )

    missing: list[str] = []

    def load_one(path: Tuple, prev: jax.Array) -> jax.Array:
        path = path_prefix + jax.tree_util.keystr(path)
        shape = prev.shape
        sharding = prev.sharding
        if path not in root:
            # Leaf absent from this checkpoint: keep the freshly-initialized value.
            #
            # Checkpoint keys are derived from pytree paths, so ADDING a field to
            # Model or State makes every older checkpoint unloadable -- the new leaf
            # simply is not on disk. Without this fallback, introducing SPIRe's
            # w_memory would strand the trained target, vanilla draft, and Phase A
            # draft. Tolerating absence lets a new field arrive at its init value
            # (zeros for w_memory, an exact no-op) while every stored leaf still
            # loads exactly as before.
            #
            # The names are reported rather than swallowed: a typo'd field would
            # otherwise silently train from its initializer instead of the
            # checkpoint, which is far worse than a loud failure.
            missing.append(path)
            return prev
        arr = root[path]
        assert arr.shape == shape, f"Expected shape {shape} but got {arr.shape} for {path} in {filename}"
        assert arr.dtype == prev.dtype, f"Expected dtype {prev.dtype} but got {arr.dtype} for {path} in {filename}"
        del prev  # Deallocate memory before loading its replacement!
        return jax.make_array_from_callback(shape, sharding, lambda shard_index: arr[shard_index])

    state, treedef = jax.tree_util.tree_flatten_with_path(state)
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_io_threads) as executor:
        state_futures = [executor.submit(load_one, path, shape) for (path, shape) in state]
        states = [f.result() for f in state_futures]
    if missing:
        print(
            f"[checkpoint] {len(missing)} leaf/leaves absent from {filename}, kept at initialized "
            f"values: {', '.join(sorted(missing)[:8])}{' ...' if len(missing) > 8 else ''}"
        )
    return jax.tree_util.tree_unflatten(treedef, states)


def save_zarr(filename: str, state: PyTree, config: IOConfig):
    """Saves a zarr checkpoint to disk.

    See docs/pytree-zarr-checkpoint.md for the checkpoint format.
    """
    state, _treedef = jax.tree_util.tree_flatten_with_path(state)

    if jax.process_index() == 0:
        # Create the zarr file and all the arrays.
        try:
            root = zarr.open_group(filename, mode="w-")
        except zarr.errors.ContainsGroupError:
            raise ValueError(f"Checkpoint {filename} already exists.")

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_io_threads) as executor:
            futures = []
            for path, arr in state:
                path = jax.tree_util.keystr(path)
                chunk_shape = arr.sharding.shard_shape(arr.shape)
                futures.append(
                    executor.submit(root.empty, name=path, shape=arr.shape, chunks=chunk_shape, dtype=arr.dtype)
                )
            for future in futures:
                future.result()
    multihost_utils.sync_global_devices("save_zarr_begin")

    root = zarr.open_group(filename, mode="r+")

    def save_shard(dst: zarr.Array, shard: jax.Array, index: Tuple[int, ...]):
        dst[index] = np.asarray(shard)

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_io_threads) as executor:
        futures = []
        for path, arr in state:
            path = jax.tree_util.keystr(path)
            dst = root[path]
            assert dst.chunks == arr.sharding.shard_shape(arr.shape)
            for shard in arr.addressable_shards:
                if shard.replica_id == 0:
                    futures.append(executor.submit(save_shard, dst, shard.data, shard.index))
        for future in futures:
            future.result()

    multihost_utils.sync_global_devices("save_zarr_end")
    if jax.process_index() == 0:
        root.attrs["write_completed"] = True
    multihost_utils.sync_global_devices("save_zarr_committed")


def step_to_str(step: int) -> str:
    """Converts a step number to a string with leading zeros.

    We pad up to 10 digits so that lexicographic order matches numerical. 1e10 training steps
    should be enough for anyone: the biggest runs as of 2024 are probably around 1e7 tokens/batch,
    1e13 tokens total, so 1e6 training steps total.
    """
    return str(step).zfill(10)


_PROFILE_DIR = None


def start_profile():
    """Starts gathering a JAX profile."""
    # Get fresh temporary directory
    global _PROFILE_DIR
    _PROFILE_DIR = tempfile.mkdtemp()
    print(f"[{datetime.datetime.now()}] Starting profile, saving to {_PROFILE_DIR}")
    jax.profiler.start_trace(_PROFILE_DIR, create_perfetto_trace=True)


def stop_profile(working_dir: str):
    """Stops gathering the JAX profile and saves it to a file."""
    global _PROFILE_DIR
    jax.profiler.stop_trace()
    print(f"[{datetime.datetime.now()}] Finished profile, copying to {working_dir}")
    fsspec_put(_PROFILE_DIR + "/", working_dir + "/")
    shutil.rmtree(_PROFILE_DIR)
    print(f"[{datetime.datetime.now()}] Finished copying profile to {working_dir}")
    _PROFILE_DIR = None


def fsspec_put(local_src: str, remote_dst: str):
    """Copies a file from local disk to a remote location specified by a fsspec path."""
    f = fsspec.open(remote_dst)
    fs = f.fs
    path = f.path
    del f
    print(f"Put {local_src} to {path}")
    fs.put(local_src, path, recursive=True, create_parents=True)


def _modify_labels_of_hlo_dot(hlo_dot: str) -> str:
    """
    Modifies the labels of the HLO dot file to include the source code of the caller in the node and streamline the tooltips.
    """
    hlo_dot = hlo_dot.replace("source: ", "")
    # remove common path prefix, e.g. /tmp/localdisk/clearml/venvs-builds/3.10/task_repository/ML.git/train.py:loss:390
    common_prefix = re.search(r"^(\/.*?\/)train\.py:", hlo_dot, re.MULTILINE).group(1)
    hlo_dot = hlo_dot.replace(common_prefix, "")
    hlo_dot_patched = []
    for line in hlo_dot.splitlines():
        if "label=<" in line and 'tooltip="jit(' in line:
            extra_label = re.findall(r"/seqax_(.*?)_xaqes/", line)
            # handle cases like jit(training_step)/jit(main)/jit(shmap_body)/seqax_BWD_xaqes/transpose(seqax_FWD_xaqes)/...
            if "BWD" in extra_label and "FWD" in extra_label:
                extra_label.remove("FWD")
            extra_label = " ".join(extra_label)
            label_end = line.index(">, shape=")
            line1, line2 = line[:label_end], line[label_end:]
            line = f"{line1}<br/>{extra_label}{line2}"
        hlo_dot_patched.append(line)
    hlo_dot = "\n".join(hlo_dot_patched)
    return hlo_dot


def _remove_hlo_dot_hover_and_empty_lines(hlo_dot: str) -> str:
    """Removes all lines containing `hover` and empty lines from the HLO dot file."""
    # This is comparable to:
    # Edit the SVG to remove everything before <svg>. There's a bunch of hover CSS that massively slows down
    # rendering in Chrome and adds little value: it just highlights edges when you hover over them.
    return "\n".join([line for line in hlo_dot.splitlines() if "hover" not in line and line.strip()])


def save_hlo_dot(filespec: str, compiled: jax.stages.Compiled):
    """Saves a compiled function's HLO to an SVG file."""
    if jax.process_index() == 0:
        compiled_hlo_dot = xla_extension.hlo_module_to_dot_graph(compiled.runtime_executable().hlo_modules()[0])
        compiled_hlo_dot = _modify_labels_of_hlo_dot(compiled_hlo_dot)
        compiled_hlo_dot = _remove_hlo_dot_hover_and_empty_lines(compiled_hlo_dot)
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "hlo.dot"), "w") as f:
                f.write(compiled_hlo_dot)
            fsspec_put(f.name, filespec)


def mkdir(filespec: str):
    """Creates a directory at the specified (possibly remote) fsspec path."""
    f = fsspec.open(filespec)
    fs = f.fs
    path = f.path
    del f
    if not fs.exists(path):
        fs.mkdir(path, create_parents=False)


def get_flops_per_device():
    """Gets the FLOPS per device for the current device kind."""
    device = jax.devices()[0].device_kind
    if device.startswith("NVIDIA A100"):
        result = 312e12
    elif device.startswith("NVIDIA H100 80GB HBM3"):
        result = 989.4e12
    elif device.startswith("TPU v4"):
        result = 275e12
    else:
        print(f"Unrecognized device, assuming ridiculously low 1 MFLOPS. Device name: {device}")
        result = 1e6
    print(f"Device kind: {device}")
    print(f"FLOPS per device: {result:_}")
    return result


def run_command(command: List[str]):
    return subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def get_git_commit_hash():
    """Get the current git commit hash."""
    return run_command(["git", "rev-parse", "HEAD"]).stdout.strip()


def get_git_branch():
    return run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()

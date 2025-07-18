import os
import subprocess
import sys


def check_device():
    if "COLAB_TPU_ADDR" in os.environ or os.path.exists("/usr/lib/libtpu.so"):
        return "TPU"
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return "GPU"
    except Exception:
        pass

    return "CPU"


def is_tpu():
    return check_device() == "TPU"


def set_variables():
    device = check_device()
    if device == "GPU":
        os.environ["XLA_FLAGS"] = " ".join(
            [
                os.environ.get("XLA_FLAGS", ""),
                "--xla_gpu_enable_latency_hiding_scheduler=true",
                # Use deterministic ops for GPU?
                "--xla_gpu_deterministic_ops=false",
            ]
        )
        os.environ.update(
            {
                "NCCL_LL128_BUFFSIZE": "-2",
                "NCCL_LL_BUFFSIZE": "-2",
                "NCCL_PROTO": "SIMPLE,LL,LL128",
            }
        )
        os.environ.update({"JAX_PLATFORMS": "cuda"})
    elif device == "TPU":
        # TPU optimization flags from https://github.com/jax-ml/jax/blob/main/docs/xla_flags.md
        # Previously we also used
        # --xla_tpu_enable_async_collective_fusion=true
        # --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true
        # --xla_tpu_enable_async_collective_fusion_multiple_steps=true
        # --xla_enable_async_all_gather=true
        # but they caused non-deterministic behavior when tensor stat logging was enabled.
        os.environ["LIBTPU_INIT_ARGS"] = (
            "--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_overlap_compute_collective_tc=true"
        )
        # Running on the full 4x4 TPU pod, doesn't need any extra flags, but smaller slices require these extra flags:
        # (from: https://gist.github.com/skye/f82ba45d2445bb19d53545538754f9a3)
        # 4 hosts 4x4 devices, no extra flags needed.
        # 1 host 1 device
        # os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"
        # os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1"
        # os.environ["TPU_VISIBLE_DEVICES"] = "0"
        # 1 host 2 device
        # os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,2,1"
        # os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1"
        # os.environ["TPU_VISIBLE_DEVICES"] = "0,1"
        # 1 host 4 devices
        os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "2,2,1"
        os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1"
        os.environ["TPU_VISIBLE_DEVICES"] = "0,1,2,3"

        os.environ.update({"JAX_PLATFORMS": "tpu"})
    elif device == "CPU":
        os.environ.update({"JAX_PLATFORMS": "cpu"})


# Set environment variables
assert "jax" not in sys.modules, (
    "JAX requires XLA_FLAGS to be set before importing it (https://docs.jax.dev/en/latest/xla_flags.html)"
)
set_variables()

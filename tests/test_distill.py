"""Correctness tests for SPIRe's MixedLoss, independent of whether it trains.

The load-bearing test is SELF-DISTILLATION: when the student IS the teacher, the
two distributions are identical, so

    alpha = sum_x min(p(x), q(x)) = sum_x p(x) = 1        exactly
    distill_CE = -sum_x p(x) log q(x) = H(p)              the teacher's entropy
    MixedLoss  = omega*H(p) - (1-omega)

This is the distillation analogue of the tau = k+1 self-draft check: it pins the
loss against a value known a priori, rather than against another run of the same
code. A sign error, a missing normalization, or a p/q swap all break it.

Also checks that the sparse training mask actually restricts attention -- a
streaming_llm config whose loss equals the dense loss would mean the mask is
silently not being applied, which would look like a successful run.

    PYTHONPATH=. python tests/test_distill.py
"""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
import init_seqax  # noqa: E402, F401

os.environ["JAX_PLATFORMS"] = "cpu"

from functools import partial  # noqa: E402
from typing import Tuple  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import mesh_utils  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

import shardlib.shardtypes as shardtypes  # noqa: E402
from shardlib.shardtypes import f32  # noqa: E402

shardtypes.register_with_typeguard()

from input_loader import TokenBatch  # noqa: E402
from model import Model, ModelConfig  # noqa: E402
from train import loss_fn  # noqa: E402

VOCAB, L, B = 128, 16, 4
h_dense = ModelConfig(
    vocab=VOCAB, seq_len=L, layers=2, d_model=32, n_q_per_kv=1, n_kv=4, d_head=8, d_ff=64, rope_max_timescale=100
)
h_sparse = ModelConfig(**{**vars(h_dense), "attention_mask": "streaming_llm", "sink_size": 1, "window": 4})
h_teacher = ModelConfig(**{**vars(h_dense), "layers": 3})  # deliberately a different depth


def make_batch(rng):
    ids = jax.random.randint(rng, (B, L), 1, VOCAB).astype(jnp.uint32)
    starts = np.zeros((B, L), dtype=bool)
    starts[:, 0] = True
    return TokenBatch(targets=ids, is_seq_start=jnp.asarray(starts))


def run_loss(w, h, batch, teacher=None, h_t=None, omega=0.5):
    """Returns (loss, alpha, distill_ce). The teacher is closed over rather than
    passed, exactly as training_step does it, so its differing depth never reaches
    typed_shard_map's annotation-driven spec derivation."""

    @jax.jit
    @partial(shardtypes.typed_shard_map, check_rep=False)
    def go(w: Model, batch: TokenBatch) -> Tuple[f32[b""], f32[b""], f32[b""]]:
        loss, stats = loss_fn(w, h, batch, None, teacher, h_t, omega)
        loss = jax.lax.psum(loss, ("d", "t", "s"))
        zero = jnp.float32(0.0)
        alpha = stats["loss.alpha"].mean if "loss.alpha" in stats else zero
        ce = stats["loss.distill_ce"].mean if "loss.distill_ce" in stats else zero
        return loss, alpha, ce

    return go(w, batch)


def main():
    with Mesh(mesh_utils.create_device_mesh([1, 1, 1], jax.devices()[:1]), ("d", "t", "s")):
        rng = jax.random.PRNGKey(0)
        with shardtypes.Scope():
            w = jax.jit(Model.init, static_argnums=0)(h_dense, jax.random.key_data(rng).astype(jnp.uint32))
        batch = make_batch(jax.random.fold_in(rng, 1))

        # ---- 1. Self-distillation: alpha == 1, CE == entropy ----
        omega = 0.5
        loss, alpha, ce = run_loss(w, h_dense, batch, teacher=w, h_t=h_dense, omega=omega)
        alpha, ce = float(alpha), float(ce)
        print(f"self-distill: alpha = {alpha:.6f}   distill_CE = {ce:.4f}   loss = {float(loss):.4f}")
        assert abs(alpha - 1.0) < 1e-4, f"self-distilled alpha must be 1.0, got {alpha}"
        expected = omega * ce - (1 - omega) * 1.0
        assert abs(float(loss) - expected) < 1e-4, f"MixedLoss {float(loss)} != {expected}"
        print("  alpha == 1 and MixedLoss == omega*H(p) - (1-omega) PASSED")

        # ---- 2. A genuinely different teacher gives alpha in (0, 1) ----
        with shardtypes.Scope():
            w_t = jax.jit(Model.init, static_argnums=0)(h_teacher, jax.random.key_data(jax.random.fold_in(rng, 2)))
        _, a2, _ = run_loss(w, h_dense, batch, teacher=w_t, h_t=h_teacher, omega=omega)
        a2 = float(a2)
        print(f"cross-model (3-layer teacher, 2-layer student): alpha = {a2:.4f}")
        assert 0.0 < a2 < 1.0, f"alpha must lie in (0,1), got {a2}"
        assert a2 < alpha, "a different teacher must not match as well as self-distillation"
        print("  differing-depth teacher accepted, alpha in (0,1) PASSED")

        # ---- 3. The sparse training mask actually bites ----
        dense_loss, _, _ = run_loss(w, h_dense, batch)
        sparse_loss, _, _ = run_loss(w, h_sparse, batch)
        print(f"hard-target loss: dense = {float(dense_loss):.4f}  sparse(window=4) = {float(sparse_loss):.4f}")
        assert abs(float(dense_loss) - float(sparse_loss)) > 1e-3, (
            "sparse and dense losses are identical -- the streaming_llm mask is not being applied"
        )
        print("  sparse mask changes the loss PASSED")

        # ---- 4. window >= L reproduces dense exactly ----
        h_wide = ModelConfig(**{**vars(h_dense), "attention_mask": "streaming_llm", "sink_size": 0, "window": L})
        wide_loss, _, _ = run_loss(w, h_wide, batch)
        print(f"window=L loss = {float(wide_loss):.6f} vs dense {float(dense_loss):.6f}")
        assert abs(float(wide_loss) - float(dense_loss)) < 1e-5, "window >= L must reproduce dense causal exactly"
        print("  window >= L == dense PASSED")

    print("\nALL DISTILLATION TESTS PASSED")


if __name__ == "__main__":
    main()

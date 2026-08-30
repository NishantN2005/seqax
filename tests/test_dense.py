"""Correctness harness for the dense model.py:
1. Training-style forward pass runs under typed_shard_map.
2. KV-cache decode equivalence: logits from prefill+cached one-token steps
   must match logits from the full non-cached forward pass at every position.
"""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
import init_seqax  # noqa: F401  (must precede jax import)
from functools import partial
import numpy as np
import jax, jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh

import shardlib.shardtypes as shardtypes
from shardlib.shardtypes import bool_, bf16, f32, u32, i32, make_shardings
from typing import Optional, Tuple
from typeguard import typechecked
shardtypes.register_with_typeguard()

from model import Model, ModelConfig

h = ModelConfig(vocab=256, seq_len=32, layers=2, d_model=64, n_q_per_kv=1, n_kv=8,
                d_head=16, d_ff=128, rope_max_timescale=256)

B, T, P = 8, 32, 16   # batch, total len, prefill len

with Mesh(mesh_utils.create_device_mesh([8, 1, 1], jax.devices()), ("d", "t", "s")):
    rng = jax.random.PRNGKey(0)
    weights = jax.jit(Model.init, static_argnums=0)(h, jax.random.key_data(rng).astype(jnp.uint32))
    ids = jax.random.randint(rng, (B, T), 0, h.vocab).astype(jnp.uint32)

    # ---- 1. Full (training-style) forward pass, no cache ----
    @jax.jit
    @partial(shardtypes.typed_shard_map, check_rep=False)
    @typechecked
    def full_forward(w: Model, ids: u32[b"B/d L"]) -> f32[b"B/d L V"]:
        causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))[jnp.newaxis, ...]
        mask = jnp.broadcast_to(causal, (ids.shape[0], T, T))
        with shardtypes.Scope():
            logits, _, _ = w.forward_pass(h, ids, mask)
        return logits

    ref = full_forward(weights, ids)
    print("full forward OK:", ref.shape, "finite:", bool(jnp.all(jnp.isfinite(ref))))

    # ---- 2. Prefill + cached decode ----
    @jax.jit
    @partial(shardtypes.typed_shard_map, check_rep=False)
    @typechecked
    def prefill(w: Model, ids: u32[b"B/d P"]) -> Tuple[f32[b"B/d P V"], bf16[b"layers 2 B/d T K D"]]:
        lb = ids.shape[0]
        # mask over padded cache: query i (abs pos i) attends to j <= i
        q_pos = jnp.arange(P)[:, None]
        k_pos = jnp.arange(T)[None, :]
        mask = jnp.broadcast_to(k_pos <= q_pos, (lb, P, T))
        cache = jnp.zeros((h.layers, 2, lb, T, h.n_kv, h.d_head), dtype=jnp.bfloat16)
        with shardtypes.Scope():
            logits, cache, _ = w.forward_pass(h, ids, mask, kv_cache=cache, kv_offset=jnp.zeros((lb,), jnp.int32))
        return logits, cache

    @jax.jit
    @partial(shardtypes.typed_shard_map, check_rep=False)
    @typechecked
    def decode_step(w: Model, tok: u32[b"B/d 1"], cache: bf16[b"layers 2 B/d T K D"],
                    pos: i32[b"B/d"]) -> Tuple[f32[b"B/d 1 V"], bf16[b"layers 2 B/d T K D"]]:
        lb = tok.shape[0]
        k_pos = jnp.arange(T)[None, None, :]
        mask = jnp.broadcast_to(k_pos <= pos[:, None, None], (lb, 1, T))
        with shardtypes.Scope():
            logits, cache, _ = w.forward_pass(h, tok, mask, kv_cache=cache, kv_offset=pos)
        return logits, cache

    pre_logits, cache = prefill(weights, ids[:, :P])
    outs = [pre_logits]
    for t in range(P, T):
        step_logits, cache = decode_step(weights, ids[:, t:t+1], cache, jnp.full((B,), t, jnp.int32))
        outs.append(step_logits)
    got = jnp.concatenate(outs, axis=1)

    diff = jnp.max(jnp.abs(got - ref))
    agree = jnp.mean((jnp.argmax(got, -1) == jnp.argmax(ref, -1)).astype(jnp.float32))
    print(f"max |cached - full| logit diff: {diff:.4f}")
    print(f"greedy argmax agreement: {agree:.4f}")
    assert agree == 1.0, "greedy tokens diverge between cached and full forward"
    print("KV-cache decode equivalence PASSED")
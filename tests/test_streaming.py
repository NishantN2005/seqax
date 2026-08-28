"""Step-9 correctness tests for the StreamingLLM sink+window attention in decode.py.

1. Mask unit test: streaming_visibility matches a naive Python reference.
2. window >= P+G reproduces dense generation BIT-FOR-BIT (greedy).
3. A small window changes the output (mask is actually in effect).
4. Sparse cached decode agrees with the sparse non-cached forward pass
   (the greedy check, run under the same mask).
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
shardtypes.register_with_typeguard()
from model import Model, ModelConfig
from decode import make_generate, make_greedy_check, streaming_visibility

# ---- 1. mask unit test (no mesh needed) ----
def ref_visible(q, k, sink, window):
    return (k <= q) and (k < sink or k > q - window)

T = 17
for sink, window in [(0, 5), (1, 4), (2, 1), (3, 20)]:
    q = jnp.arange(T)[:, None]; k = jnp.arange(T)[None, :]
    got = np.array(streaming_visibility(q, k, sink, window))
    want = np.array([[ref_visible(i, j, sink, window) for j in range(T)] for i in range(T)])
    assert (got == want).all(), (sink, window)
print("mask unit test PASSED")

h = ModelConfig(vocab=256, seq_len=32, layers=2, d_model=64, n_q_per_kv=1, n_kv=8,
                d_head=16, d_ff=128, rope_max_timescale=256)
B, P, G = 8, 12, 12

with Mesh(mesh_utils.create_device_mesh([8, 1, 1], jax.devices()), ("d", "t", "s")):
    rng2 = jnp.zeros((2,), dtype=jnp.uint32)
    weights = jax.jit(Model.init, static_argnums=0)(h, rng2)
    prompt = jax.random.randint(jax.random.PRNGKey(0), (B, P), 0, h.vocab).astype(jnp.uint32)

    dense = make_generate(h, P, G, 0.0)(weights, prompt, rng2)

    # ---- 2. window >= P+G == dense, bit for bit ----
    wide = make_generate(h, P, G, 0.0, sink_size=1, window=P + G)(weights, prompt, rng2)
    assert (dense == wide).all(), "window >= P+G must reproduce dense exactly"
    print("wide-window == dense PASSED")

    # ---- 3. small window differs ----
    narrow = make_generate(h, P, G, 0.0, sink_size=1, window=4)(weights, prompt, rng2)
    assert not (dense == narrow).all(), "narrow window produced identical output; mask not in effect?"
    print(f"narrow-window differs from dense PASSED ({float((dense != narrow).mean()):.0%} of tokens differ)")

    # ---- 4. sparse cached decode == sparse non-cached forward ----
    check = make_greedy_check(h, P, G, sink_size=1, window=4)
    agree = check(weights, jnp.concatenate([prompt, narrow], axis=1))
    assert float(agree) == 1.0, f"sparse cached decode diverges from sparse full forward: {float(agree)}"
    print("sparse cached-vs-full greedy check PASSED")
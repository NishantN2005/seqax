"""Decode must agree with the TRAINING input convention.

Every other test in this suite compares the implementation against itself:
test_dense checks cached vs non-cached, test_spec checks draft vs target,
decode.py --check checks generation vs its own reference pass. All of them pass
under any constant shift of the sequence, which is how a real off-by-one lived in
decode.py through a fully green suite -- the model echoed its own last token
forever and nothing noticed.

This test closes that gap by rebuilding the reference from train.py's rule rather
than from decode.py's code. train.py loss_fn does:

    inputs = shift_right(targets)         # position 0 masked to token 0
    logits[i] is scored against targets[i]

so the model reads ids[i] as "the token BEFORE position i", and predicting the
token after a context c[0..m-1] means feeding [0, c[0], ..., c[m-1]] and reading
index m. If decode.py ever stops prepending that sentinel, or prepends it in the
wrong place, REFERENCE below diverges and this test fails.
"""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
import init_seqax  # noqa: F401,E402  (must precede jax import)

from functools import partial  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import mesh_utils  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from typeguard import typechecked  # noqa: E402

import shardlib.shardtypes as shardtypes  # noqa: E402
from shardlib.shardtypes import f32, u32  # noqa: E402

shardtypes.register_with_typeguard()

from decode import make_generate  # noqa: E402
from model import Model, ModelConfig  # noqa: E402

h = ModelConfig(vocab=256, seq_len=64, layers=2, d_model=64, n_q_per_kv=1, n_kv=8,
                d_head=16, d_ff=128, rope_max_timescale=256)
B, P, G = 8, 12, 10


def _fwd(w, ids):
    """Non-cached forward pass at whatever length ids happens to be."""
    n = ids.shape[1]

    @jax.jit
    @partial(shardtypes.typed_shard_map, check_rep=False)
    @typechecked
    def fwd(w: Model, x: u32[b"B/d L"]) -> f32[b"B/d L V"]:
        causal = jnp.tril(jnp.ones((n, n), dtype=jnp.bool_))[jnp.newaxis, ...]
        mask = jnp.broadcast_to(causal, (x.shape[0], n, n))
        with shardtypes.Scope():
            logits, _, _ = w.forward_pass(h, x, mask)
        return logits

    # shardlib dimension names are globally scoped and persist across calls, so a
    # second call at a different length fails without a fresh Scope.
    with shardtypes.Scope():
        return np.asarray(fwd(w, jnp.asarray(ids, dtype=jnp.uint32)))


def reference_greedy(w, prompt: np.ndarray, n_new: int) -> np.ndarray:
    """Greedy decode written directly from train.py's convention. No cache, no
    shared code with decode.py -- this is the independent oracle."""
    seq = [list(row) for row in prompt]
    for _ in range(n_new):
        ids = np.array([[0] + row for row in seq], dtype=np.uint32)  # shift_right
        logits = _fwd(w, ids)
        for b in range(len(seq)):
            seq[b].append(int(np.argmax(logits[b, -1])))
    return np.array([row[len(prompt[0]):] for row in seq], dtype=np.uint32)


with Mesh(mesh_utils.create_device_mesh([8, 1, 1], jax.devices()), ("d", "t", "s")):
    rng = jax.random.PRNGKey(0)
    w = jax.jit(Model.init, static_argnums=0)(h, jax.random.key_data(rng).astype(jnp.uint32))
    prompt = np.asarray(jax.random.randint(rng, (B, P), 0, h.vocab).astype(jnp.uint32))

    got = np.asarray(make_generate(h, P, G, 0.0)(w, jnp.asarray(prompt), jnp.zeros((2,), jnp.uint32)))
    want = reference_greedy(w, prompt, G)

    agree = float((got == want).mean())
    print(f"decode vs training-convention reference: {agree:.4f} agreement")
    assert agree == 1.0, (
        f"decode.py disagrees with train.py's input convention ({agree:.4f}).\n"
        f"  got  {got[0].tolist()}\n  want {want[0].tolist()}"
    )
    print("decode matches the training convention PASSED")

    # Guard the specific regression: without the BOS sentinel the model is fed a
    # sequence shifted one position late, and greedy decoding degenerates into
    # repeating whatever token it was last handed.
    seq = [list(row) for row in prompt]
    for _ in range(G):
        logits = _fwd(w, np.array(seq, dtype=np.uint32))  # no shift -- the old bug
        for b in range(len(seq)):
            seq[b].append(int(np.argmax(logits[b, -1])))
    unshifted = np.array([row[P:] for row in seq], dtype=np.uint32)
    assert not np.array_equal(unshifted, want), (
        "shifted and unshifted decoding agree, so this test cannot detect the "
        "missing-BOS bug it exists to catch"
    )
    print("unshifted decoding is measurably different (test has teeth) PASSED")

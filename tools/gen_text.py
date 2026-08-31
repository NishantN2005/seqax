"""Generate from the trained target with a REAL text prompt.

decode.py prompts with random token ids, which exercises the plumbing but says
nothing about whether the model learned language. This tokenizes actual English,
generates greedily, and decodes back, so the output is judgeable by eye.
"""

import init_seqax  # noqa: F401  # must precede jax import

import argparse
import os

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from omegaconf import OmegaConf
from transformers import GPT2TokenizerFast

import shardlib.shardtypes as shardtypes

shardtypes.register_with_typeguard()

from decode import load_weights, make_generate  # noqa: E402
from model import ModelConfig  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "In 1969, humans first walked on the",
    "Water boils at a temperature of",
    "She opened the door and saw",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="spire_target_1024")
    p.add_argument("--model-name", default="spire_target_1024")
    p.add_argument("--gen-len", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    cfg = OmegaConf.load(f"configs/{args.config}.yaml")
    h = ModelConfig(**cfg.model)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    # Left-pad to a common length so the batch is rectangular. Padding sits at the
    # front, so with causal attention it only ever precedes real tokens.
    ids = [tok.encode(s) for s in PROMPTS]
    plen = max(len(i) for i in ids)
    pad = tok.encode("\n")[0]
    prompt = jnp.asarray([[pad] * (plen - len(i)) + i for i in ids], dtype=jnp.uint32)

    mesh_cfg = cfg.mesh
    with Mesh(mesh_utils.create_device_mesh([mesh_cfg.d, mesh_cfg.t, mesh_cfg.s], jax.devices()), ("d", "t", "s")):
        rng = jnp.zeros((2,), dtype=jnp.uint32)
        weights, step = load_weights(h, os.path.join(cfg.root_working_dir, args.model_name), rng)
        print(f"loaded checkpoint step {step}")

        generate = make_generate(h, plen, args.gen_len, args.temperature)
        out = generate(weights, prompt, rng)

        for i, src in enumerate(PROMPTS):
            cont = tok.decode([int(t) for t in out[i]])
            print(f"\n  prompt:     {src!r}")
            print(f"  continues:  {cont!r}")


if __name__ == "__main__":
    main()

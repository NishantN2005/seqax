"""The definition of our transformer model."""

# Set XLA flags before importing JAX
import init_seqax  # noqa: F401  # isort: skip

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import einops
import jax
import jax.numpy as jnp
from jax import lax
from typeguard import typechecked

import jax_extra
import shardlib.shardops as shardops
import shardlib.shardtypes as shardtypes
from init_seqax import is_tpu
from jax_extra import explicit_activation_checkpointing, fold_in_str, save_for_backward
from shardlib.shardtypes import Array, bf16, bool_, f32, i32, make_shardings, pytree_dataclass, u32

shardtypes.register_with_typeguard()
PRNGKey = u32[b"2"]  # Type annotation to enable run-time typechecking of PRNGKeys by shardtypes


@dataclass(frozen=True)
class MeshConfig:
    d: int
    t: int
    s: int


@dataclass(frozen=True)
class ModelConfig:
    vocab: int
    seq_len: int
    layers: int
    d_model: int
    n_q_per_kv: int
    n_kv: int
    d_head: int
    d_ff: int
    rope_max_timescale: int
    nsa_l: int  # compression block size
    nsa_d: int  # sliding stride
    nsa_L: int  # selected block size
    nsa_n: int  # selected block count (including first block and last two blocks)
    nsa_w: int  # window size


@pytree_dataclass
class TransformerLayer:
    w_q: f32[b"d_model/d/s n_q_per_kv n_kv/t d_head"]
    w_k: f32[b"3 d_model/d/s n_kv/t d_head"]
    w_v: f32[b"3 d_model/d/s n_kv/t d_head"]
    w_o: f32[b"d_model/d/s n_q_per_kv n_kv/t d_head"]
    w_gate: f32[b"d_model/d/s d_ff/t"]
    w_up: f32[b"d_model/d/s d_ff/t"]
    w_down: f32[b"d_model/d/s d_ff/t"]
    ln_attn_in: f32[b"d_model/t/d/s"]
    ln_q: f32[b"n_q_per_kv n_kv/t d_head/d/s"]
    ln_k: f32[b"3 n_kv/t d_head/d/s"]
    ln_qkv: f32[b"n_q_per_kv n_kv/t d_head/d/s"]
    ln_attn_out: f32[b"d_model/t/d/s"]
    ln_ffn_in: f32[b"d_model/t/d/s"]
    ln_ffn_out: f32[b"d_model/t/d/s"]
    phi: f32[b"nsa_lxD n_kv/t d_head"]  # Token compression MLP
    k_intrablock_pe: f32[b"nsa_l n_kv/t d_head"]
    v_intrablock_pe: f32[b"nsa_l n_kv/t d_head"]
    w_nsa_gate: f32[b"d_head n_q_per_kv n_kv/t 3"]  # Gate MLP


Transformer = Array["layers", TransformerLayer]


@pytree_dataclass
class Model:
    embed: f32[b"vocab/t d_model/d/s"]
    unembed: f32[b"vocab/t d_model/d/s"]
    transformer: Transformer
    ln_embed: f32[b"d_model/t/d/s"]
    ln_final: f32[b"d_model/t/d/s"]

    @staticmethod
    @typechecked
    def init(h: ModelConfig, rng: PRNGKey) -> "Model":
        # All weight matrices except embedding, unembedding, and normalization layers are initialized with 'fan_in'
        # scaling, i.e. variance set to 1.0/fan_in.

        # The constant is stddev of standard normal truncated to (-2, 2)
        truncated_normal_stddev = 0.87962566103423978
        # scale for tensors with d_model fan_in and truncated normal truncated to (-2, 2)
        d_model_scale = 1 / (math.sqrt(h.d_model) * truncated_normal_stddev)

        embed_scale = 1 / math.sqrt(h.d_model)  # See Adafactor paper, §8.1.
        w_q_scale = d_model_scale
        ln_q_scale = 1 / math.sqrt(h.d_head)
        w_k_scale = d_model_scale
        total_head_dim = h.n_q_per_kv * h.n_kv * h.d_head
        w_o_scale = 1 / (math.sqrt(total_head_dim) * truncated_normal_stddev)
        w_up_scale = d_model_scale
        w_down_scale = 1 / (math.sqrt(h.d_ff) * truncated_normal_stddev)
        unembed_scale = d_model_scale

        embed = embed_scale * jax.random.normal(
            jax_extra.fold_in_str(rng, "embed"), (h.vocab, h.d_model), dtype=jnp.float32
        )

        w_q_shape = (h.layers, h.d_model, h.n_q_per_kv, h.n_kv, h.d_head)
        w_q = w_q_scale * jax.random.truncated_normal(fold_in_str(rng, "w_q"), -2, 2, w_q_shape, dtype=jnp.float32)
        w_k_shape = (h.layers, 3, h.d_model, h.n_kv, h.d_head)
        w_k = w_k_scale * jax.random.truncated_normal(fold_in_str(rng, "w_k"), -2, 2, w_k_shape, dtype=jnp.float32)
        w_v = w_k_scale * jax.random.truncated_normal(fold_in_str(rng, "w_v"), -2, 2, w_k_shape, dtype=jnp.float32)
        w_o_shape = w_q_shape
        w_o = w_o_scale * jax.random.truncated_normal(fold_in_str(rng, "w_o"), -2, 2, w_o_shape, dtype=jnp.float32)

        ff_shape = (h.layers, h.d_model, h.d_ff)
        w_gate = w_up_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_gate"), -2, 2, ff_shape, dtype=jnp.float32
        )
        w_up = w_up_scale * jax.random.truncated_normal(fold_in_str(rng, "w_up"), -2, 2, ff_shape, dtype=jnp.float32)
        w_down = w_down_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_down"), -2, 2, ff_shape, dtype=jnp.float32
        )

        unembed = unembed_scale * jax.random.truncated_normal(
            fold_in_str(rng, "unembed"), -2, 2, (h.vocab, h.d_model), dtype=jnp.float32
        )

        # https://github.com/google/jax/issues/20390 for ones_like with sharding.
        ln_embed = jnp.ones((h.d_model,), dtype=jnp.float32)
        ln_attn_in = jnp.ones((h.layers, h.d_model), dtype=jnp.float32)
        ln_q = ln_q_scale * jnp.ones((h.layers, h.n_q_per_kv, h.n_kv, h.d_head), dtype=jnp.float32)
        ln_k = jnp.ones((h.layers, 3, h.n_kv, h.d_head), dtype=jnp.float32)
        ln_qkv = jnp.ones((h.layers, h.n_q_per_kv, h.n_kv, h.d_head), dtype=jnp.float32)
        ln_attn_out = jnp.ones((h.layers, h.d_model), dtype=jnp.float32)
        ln_ffn_in = jnp.ones((h.layers, h.d_model), dtype=jnp.float32)
        ln_ffn_out = jnp.ones((h.layers, h.d_model), dtype=jnp.float32)
        ln_final = jnp.ones((h.d_model,), dtype=jnp.float32)

        # Token compression MLP
        phi_scale = 1 / (math.sqrt(h.nsa_l * h.d_head) * truncated_normal_stddev)
        phi_shape = (h.layers, h.nsa_l * h.d_head, h.n_kv, h.d_head)
        phi = phi_scale * jax.random.truncated_normal(fold_in_str(rng, "phi"), -2, 2, phi_shape, dtype=jnp.float32)

        # Token compression intra-block position encoding
        k_intrablock_pe = jnp.zeros((h.layers, h.nsa_l, h.n_kv, h.d_head), dtype=jnp.float32)
        v_intrablock_pe = jnp.zeros((h.layers, h.nsa_l, h.n_kv, h.d_head), dtype=jnp.float32)

        # Gate MLP
        w_nsa_gate_scale = 1 / (math.sqrt(h.d_head) * truncated_normal_stddev)
        w_nsa_gate_shape = (h.layers, h.d_head, h.n_q_per_kv, h.n_kv, 3)
        w_nsa_gate = w_nsa_gate_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_nsa_gate"), -2, 2, w_nsa_gate_shape, dtype=jnp.float32
        )

        arrays = Model(
            embed=embed,
            unembed=unembed,
            transformer=Transformer(
                w_q=w_q,
                w_k=w_k,
                w_v=w_v,
                w_o=w_o,
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                ln_attn_in=ln_attn_in,
                ln_q=ln_q,
                ln_k=ln_k,
                ln_qkv=ln_qkv,
                ln_attn_out=ln_attn_out,
                ln_ffn_in=ln_ffn_in,
                ln_ffn_out=ln_ffn_out,
                phi=phi,
                k_intrablock_pe=k_intrablock_pe,
                v_intrablock_pe=v_intrablock_pe,
                w_nsa_gate=w_nsa_gate,
            ),
            ln_embed=ln_embed,
            ln_final=ln_final,
        )
        shardings = make_shardings(Model)
        return jax.tree.map(lax.with_sharding_constraint, arrays, shardings)

    @typechecked
    def forward_pass(
        self,
        h: ModelConfig,
        ids: u32[b"B/d L/s"],
        attention_mask: bool_[b"B/d L/s Klen"],
        rng: Optional[PRNGKey] = None,
        kv_cache: Optional[bf16[b"layers 2 3 B/d Klen K/t D"]] = None,
        kv_offset: Optional[i32[b""]] = None,
    ) -> Tuple[f32[b"B/d L/s V/t"], bf16[b"layers 2 3 B/d Klen K/t D"], "StatsDict"]:
        ##### Initial embedding lookup.
        embed = shardops.all_gather("V/t M/d/s -> V/t M", jnp.bfloat16(self.embed))
        x = shardops.index_unreduced("[V/t] M, B/d L/s -> B/d L/s M", embed, ids, use_onehot=is_tpu())
        x *= math.sqrt(h.d_model)
        ln_embed = shardops.all_gather("M/t/d/s -> M", jnp.float32(self.ln_embed))
        x = jnp.bfloat16(rms_norm(x) * ln_embed)
        x = shardops.psum_scatter("B/d L/s M -> B/d L/s M/t", x)

        Klen = attention_mask.shape[2]
        rope_table = RopeTable.create(Klen, h)

        layer_rngs = jax.random.split(fold_in_str(rng, "layer"), h.layers) if rng is not None else None

        # Token compression attention mask
        n_cmp_blocks = math.floor((Klen - h.nsa_l) / h.nsa_d) + 1
        largest_k_index_in_cmp_block = jnp.array([i * h.nsa_d + h.nsa_l - 1 for i in range(n_cmp_blocks)])
        q_index = jnp.arange(Klen)[:, jnp.newaxis]
        cmp_mask = largest_k_index_in_cmp_block <= q_index
        # Importance score computation for token selection
        n_slc_blocks = Klen // h.nsa_L
        largest_k_index_in_slc_block = jnp.array([(i + 1) * h.nsa_L - 1 for i in range(n_slc_blocks)])
        smallest_k_index_in_cmp_block = jnp.array([i * h.nsa_d for i in range(n_cmp_blocks)])
        smallest_k_index_in_slc_block = jnp.array([i * h.nsa_L for i in range(n_slc_blocks)])
        total_overlap_mask = jnp.logical_and(
            largest_k_index_in_cmp_block[jnp.newaxis, :] <= largest_k_index_in_slc_block[:, jnp.newaxis],
            smallest_k_index_in_cmp_block[jnp.newaxis, :] >= smallest_k_index_in_slc_block[:, jnp.newaxis],
        )  # Entry i,j is True if the i-th cmp block fully overlaps with the j-th slc block
        partial_overlap_mask = jnp.logical_or(
            jnp.logical_and(
                largest_k_index_in_cmp_block[jnp.newaxis, :] > largest_k_index_in_slc_block[:, jnp.newaxis],
                smallest_k_index_in_cmp_block[jnp.newaxis, :] < largest_k_index_in_slc_block[:, jnp.newaxis],
            ),
            jnp.logical_and(
                largest_k_index_in_cmp_block[jnp.newaxis, :] > smallest_k_index_in_slc_block[:, jnp.newaxis],
                smallest_k_index_in_cmp_block[jnp.newaxis, :] < smallest_k_index_in_slc_block[:, jnp.newaxis],
            ),
        )  # Entry i,j is True if the i-th cmp block only partially overlaps with the j-th slc block
        p_slc_mask = 2 * total_overlap_mask + partial_overlap_mask
        # Static token selection mask
        k_index = jnp.arange(Klen)[jnp.newaxis, :]
        initial_slc_block_mask = jnp.concatenate(
            [jnp.zeros((h.nsa_L, Klen)), jnp.repeat(k_index < h.nsa_L, Klen - h.nsa_L, axis=0)], axis=0
        )
        q_slc_block_index = (q_index // h.nsa_L) * h.nsa_L
        local_slc_blocks_mask = jnp.logical_and(k_index < q_slc_block_index, k_index + 1 * h.nsa_L >= q_slc_block_index)
        chunk_mask = jnp.logical_and(k_index <= q_index, k_index >= q_slc_block_index)
        static_slc_mask = jnp.logical_or(chunk_mask, jnp.logical_or(initial_slc_block_mask, local_slc_blocks_mask))
        # Sliding window attention mask
        window_mask = jnp.logical_and(k_index <= q_index, k_index + h.nsa_w > q_index)

        @typechecked
        def compress(
            k: bf16[b"B/d Klen K/t D"],
            v: bf16[b"B/d Klen K/t D"],
            phi: f32[b"nsa_lxD K/t D"],
            k_intrablock_pe: f32[b"nsa_l K/t D"],
            v_intrablock_pe: f32[b"nsa_l K/t D"],
        ) -> Tuple[bf16[b"B/d n_cb K/t D"], bf16[b"B/d n_cb K/t D"]]:
            k_blocks = jnp.stack([k[:, i * h.nsa_d : i * h.nsa_d + h.nsa_l] for i in range(n_cmp_blocks)])
            v_blocks = jnp.stack([v[:, i * h.nsa_d : i * h.nsa_d + h.nsa_l] for i in range(n_cmp_blocks)])
            k_blocks = jnp.bfloat16(k_blocks + k_intrablock_pe[jnp.newaxis, jnp.newaxis, ...])
            v_blocks = jnp.bfloat16(v_blocks + v_intrablock_pe[jnp.newaxis, jnp.newaxis, ...])
            k_blocks = einops.rearrange(k_blocks, "n b l K D -> n b K (l D)")
            v_blocks = einops.rearrange(v_blocks, "n b l K D -> n b K (l D)")
            spec = "n_cb B/d K/t nsa_lxD, nsa_lxD K/t D -> B/d n_cb K/t D"
            k_cmp = shardops.einsum_unreduced(spec, k_blocks, jnp.bfloat16(phi))
            v_cmp = shardops.einsum_unreduced(spec, v_blocks, jnp.bfloat16(phi))
            return k_cmp, v_cmp

        def attn(q, k, v, mask):
            # spec = "B/d Qlen/s Q K/t D, B/d Klen K/t D -> B/d Qlen/s Klen Q K/t"
            logits = jnp.einsum("b q Q K D, b k K D -> b q k Q K", q, k, preferred_element_type=jnp.float32)
            # All entries in the first nsa_l - 1 rows of cmp_mask are False, so to prevent the activations of the first
            # nsa_l - 1 tokens from being a uniform average of all (including future) values, we replace masked logits
            # with -inf instead of -1e10.
            masked_logits = jnp.where(mask, logits, -jnp.inf)
            probs = jnp.nan_to_num(jax.nn.softmax(jnp.float32(masked_logits), axis=2), 0)
            # spec = "B/d Qlen/s Klen Q K/t, B/d Klen K/t D -> B/d Qlen/s Q K/t D"
            return jnp.einsum("b q k Q K, b k K D -> b q Q K D", jnp.bfloat16(probs), v), probs

        @typechecked
        def native_sparse_attention(
            q: bf16[b"B/d L/s Q K/t D"],
            k: bf16[b"3 B/d Klen K/t D"],
            v: bf16[b"3 B/d Klen K/t D"],
            layer_weights: TransformerLayer,
        ) -> bf16[b"B/d Qlen/s Q K/t D"]:
            # Token compression
            k_cmp, v_cmp = compress(
                k[0], v[0], layer_weights.phi, layer_weights.k_intrablock_pe, layer_weights.v_intrablock_pe
            )
            qkv_cmp, p_cmp = attn(q, k_cmp, v_cmp, cmp_mask[jnp.newaxis, :, :, jnp.newaxis, jnp.newaxis])

            q = jnp.bfloat16(rope_table.apply("L D -> 1 L 1 1 D", q, kv_offset))
            k = jnp.bfloat16(rope_table.apply("L d -> 1 1 L 1 d", k, kv_offset))

            # Token selection
            p_slc = shardops.einsum_unreduced("n_sb n_cb, B/d L/s n_cb Q K/t -> B/d Q K/t L/s n_sb", p_slc_mask, p_cmp)
            p_slc = jnp.sum(p_slc, axis=1, keepdims=True)  # Equation 10 (https://arxiv.org/pdf/2502.11089#page=7)

            """
            # Force each block of num_draft_tokens + 1 tokens to select the same KV.
            num_draft_tokens = 3
            p_slc = einops.rearrange(p_slc, "b 1 K (n_qb l) n -> b 1 K n_qb l n", l=num_draft_tokens + 1)
            p_slc = p_slc[..., 0, :]
            p_slc = jnp.repeat(p_slc, num_draft_tokens + 1, axis=3)
            # If num_draft_tokens >= nsa_L, remove this block of code and use get_dynamic_slc_mask_alt from blog.ipynb.
            """

            def get_dynamic_slc_mask(p_slc):
                block_indices_to_rank = [jnp.arange(1, t // h.nsa_L)[:-1] for t in range(Klen)]
                ranked_slc_block_indices = [
                    jnp.argsort(p_slc[t][block_indices_to_rank[t]], descending=True) for t in range(Klen)
                ]
                dynamic_slc_block_indices = [(1 + ranked_slc_block_indices[t])[: h.nsa_n - 3] for t in range(Klen)]
                dynamic_slc_mask = jnp.zeros((Klen, Klen))
                for t in range(Klen):
                    for i in dynamic_slc_block_indices[t]:
                        dynamic_slc_mask = jax.lax.dynamic_update_slice(
                            dynamic_slc_mask, jnp.ones((1, h.nsa_L)), (t, i * h.nsa_L)
                        )
                return dynamic_slc_mask

            dynamic_slc_mask = jax.vmap(jax.vmap(jax.vmap(get_dynamic_slc_mask)))(p_slc)
            slc_mask = jnp.logical_or(static_slc_mask[jnp.newaxis, jnp.newaxis, jnp.newaxis, ...], dynamic_slc_mask)
            qkv_slc, _ = attn(q, k[1], v[1], einops.rearrange(slc_mask, "b 1 K q k  -> b q k 1 K"))

            # Sliding window
            qkv_win, _ = attn(q, k[2], v[2], window_mask[jnp.newaxis, :, :, jnp.newaxis, jnp.newaxis])

            gate_scores = shardops.einsum_unreduced(
                "B/d L/s Q K/t D, D Q K/t 3 -> B/d L/s Q K/t 3", q, jnp.bfloat16(layer_weights.w_nsa_gate)
            )
            gate_scores = jax.nn.sigmoid(gate_scores)
            return shardops.einsum_unreduced(
                "B/d Qlen/s Q K/t 3, 3 B/d Qlen/s Q K/t D -> B/d Qlen/s Q K/t D",
                gate_scores,
                jnp.stack((qkv_cmp, qkv_slc, qkv_win)),
            )

        ##### Transformer blocks.
        @explicit_activation_checkpointing
        @typechecked
        def loop_body(
            x: bf16[b"B/d L/s M/t"],
            scanned_var: Tuple[TransformerLayer, Optional[bf16[b"2 3 B/d Klen K/t D"]], Optional[PRNGKey]],
        ) -> Tuple[bf16[b"B/d L/s M/t"], Tuple[bf16[b"2 3 B/d Klen K/t D"], "StatsDict"]]:
            layer_weights, kv_layer, layer_rng_key = scanned_var

            # Pre-attention RMSNorm
            gx = shardops.all_gather("B/d L/s M/t -> B/d L/s M", x)
            ln_attn_in = shardops.all_gather("M/t/d/s -> M", jnp.float32(layer_weights.ln_attn_in))
            nx = jnp.bfloat16(rms_norm(gx) * ln_attn_in)
            tensor_stats = {"attn_input.act": gx, "attn_input_normed.act": nx}

            # Attention, using Grouped Query Attention and RoPE position embeddings.
            w_q = shardops.all_gather("M/d/s Q K/t D -> M Q K/t D", jnp.bfloat16(layer_weights.w_q))
            q = shardops.einsum_unreduced("B/d L/s M, M Q K/t D -> B/d L/s Q K/t D", nx, w_q)
            q = save_for_backward(q)
            ln_q = shardops.all_gather("Q K/t D/d/s -> Q K/t D", jnp.float32(layer_weights.ln_q))
            q = jnp.bfloat16(rms_norm(q) * ln_q)
            w_k = shardops.all_gather("3 M/d/s K/t D -> 3 M K/t D", jnp.bfloat16(layer_weights.w_k))
            w_v = shardops.all_gather("3 M/d/s K/t D -> 3 M K/t D", jnp.bfloat16(layer_weights.w_v))
            k = shardops.einsum_unreduced("B/d L/s M, 3 M K/t D -> 3 B/d L/s K/t D", nx, w_k)
            v = shardops.einsum_unreduced("B/d L/s M, 3 M K/t D -> 3 B/d L/s K/t D", nx, w_v)
            k = save_for_backward(k)
            v = save_for_backward(v)
            ln_k = shardops.all_gather("3 K/t D/d/s -> 3 K/t D", jnp.float32(layer_weights.ln_k))
            k = jnp.bfloat16(rms_norm(k) * ln_k[:, jnp.newaxis, jnp.newaxis, ...])
            k = shardops.all_gather("3 B/d L/s K/t D -> 3 B/d L K/t D", k)
            v = shardops.all_gather("3 B/d L/s K/t D -> 3 B/d L K/t D", v)
            if kv_layer is not None:
                prev_k, prev_v = kv_layer
                k = jax.lax.dynamic_update_slice(prev_k, k, (0, 0, kv_offset, 0, 0))
                v = jax.lax.dynamic_update_slice(prev_v, v, (0, 0, kv_offset, 0, 0))
            qkv = native_sparse_attention(q, k, v, layer_weights)
            ln_qkv = shardops.all_gather("Q K/t D/d/s -> Q K/t D", jnp.float32(layer_weights.ln_qkv))
            qkv = jnp.bfloat16(rms_norm(qkv) * ln_qkv)
            w_o = shardops.all_gather("M/d/s Q K/t D -> M Q K/t D", jnp.bfloat16(layer_weights.w_o))
            attn_out = shardops.einsum_unreduced("B/d Qlen/s Q K/t D, M Q K/t D -> B/d Qlen/s M", qkv, w_o)
            attn_out = shardops.psum_scatter("B/d Qlen/s M -> B/d Qlen/s M/t", attn_out)
            ln_attn_out = shardops.all_gather("M/t/d/s -> M/t", jnp.float32(layer_weights.ln_attn_out))
            attn_out = jnp.bfloat16(rms_norm(attn_out) * ln_attn_out)
            x = save_for_backward(x + attn_out)

            # Pre-FFN RMSNorm
            gx = shardops.all_gather("B/d L/s M/t -> B/d L/s M", x)
            ln_ffn_in = shardops.all_gather("M/t/d/s -> M", jnp.float32(layer_weights.ln_ffn_in))
            nx = jnp.bfloat16(rms_norm(gx) * ln_ffn_in)

            # FFN, using SwiGLU
            w_gate = shardops.all_gather("M/d/s F/t -> M F/t", jnp.bfloat16(layer_weights.w_gate))
            gate_proj = shardops.einsum_unreduced("B/d L/s M, M F/t -> B/d L/s F/t", nx, w_gate)
            gate_proj = save_for_backward(gate_proj)
            w_up = shardops.all_gather("M/d/s F/t -> M F/t", jnp.bfloat16(layer_weights.w_up))
            up_proj = shardops.einsum_unreduced("B/d L/s M, M F/t -> B/d L/s F/t", nx, w_up)
            up_proj = save_for_backward(up_proj)
            y = jax.nn.swish(gate_proj) * up_proj
            w_down = shardops.all_gather("M/d/s F/t -> M F/t", jnp.bfloat16(layer_weights.w_down))
            ffn_out = shardops.einsum_unreduced("B/d L/s F/t, M F/t -> B/d L/s M", y, w_down)
            ffn_out = shardops.psum_scatter("B/d L/s M -> B/d L/s M/t", ffn_out)
            ln_ffn_out = shardops.all_gather("M/t/d/s -> M/t", jnp.float32(layer_weights.ln_ffn_out))
            ffn_out = jnp.bfloat16(rms_norm(ffn_out) * ln_ffn_out)
            x = x + ffn_out

            kv_layer = jnp.stack((k, v))

            tensor_stats.update(
                {
                    "attn_q.act": q,
                    "attn_k.act": k,
                    "attn_v.act": v,
                    "attn_qkv.act": qkv,
                    "attn_out.act": attn_out,
                    "ffn_input.act": gx,
                    "ffn_input_normed.act": nx,
                    "ffn_gate.act": gate_proj,
                    "ffn_up.act": up_proj,
                    "ffn_swiglu.act": y,
                    "ffn_out.act": ffn_out,
                }
            )
            tensor_stats = jax.tree.map(lambda x: TensorStats.from_tensor(x), tensor_stats)

            return x, (kv_layer, tensor_stats)

        scanned_vars = (self.transformer, kv_cache, layer_rngs)
        x, (kv_cache, ts) = jax.lax.scan(loop_body, jnp.bfloat16(x), scanned_vars)

        ##### Final layernorm and output projection.
        x = shardops.all_gather("B/d L/s M/t -> B/d L/s M", x)
        ln_final = shardops.all_gather("M/t/d/s -> M", jnp.float32(self.ln_final))
        nx = jnp.bfloat16(rms_norm(x) * ln_final)
        unembed = shardops.all_gather("V/t M/d/s -> V/t M", jnp.bfloat16(self.unembed))
        logits = shardops.einsum_unreduced(
            "B/d L/s M, V/t M -> B/d L/s V/t", nx, unembed, preferred_element_type=jnp.float32
        )

        tensor_stats = {
            f"{i}.{k}": TensorStats(
                mean=ts[k].mean[i],
                rms_norm=ts[k].rms_norm[i],
                maxabs=ts[k].maxabs[i],
                meanabs=ts[k].meanabs[i],
            )
            for i in range(h.layers)
            for k in ts
        }
        tensor_stats.update(
            {
                "out_x.act": TensorStats.from_tensor(x),
                "out_x_normed.act": TensorStats.from_tensor(nx),
                "out.logits": TensorStats.from_tensor(logits),
            }
        )

        return logits, kv_cache, tensor_stats


@pytree_dataclass
class RopeTable:
    sin: f32[b"len/s d_head2"]
    cos: f32[b"len/s d_head2"]

    @staticmethod
    def create(max_len: int, h: ModelConfig) -> "RopeTable":
        rope_max_timescale = h.rope_max_timescale
        d_head = h.d_head
        d = d_head // 2
        # endpoint=False is equivalent to what MaxText does. endpoint=True would be more natural, though.
        timescale = jnp.logspace(0, jnp.log10(jnp.float32(rope_max_timescale)), d, endpoint=False)
        position = shardops.sharded_arange(max_len, "s")
        sinusoid_inp = jnp.float32(position[:, jnp.newaxis]) / timescale[jnp.newaxis, :]
        sin = jnp.sin(sinusoid_inp)
        cos = jnp.cos(sinusoid_inp)
        return RopeTable(sin=sin, cos=cos)

    def apply(self, rearrange_spec, x, offset):
        x1, x2 = jnp.split(x, 2, axis=-1)
        if offset is None:
            sin = self.sin
            cos = self.cos
        else:
            sin = jax.lax.dynamic_slice(self.sin, (offset, 0), (x.shape[1], self.sin.shape[1]))
            cos = jax.lax.dynamic_slice(self.cos, (offset, 0), (x.shape[1], self.cos.shape[1]))
        sin = einops.rearrange(sin, rearrange_spec)
        cos = einops.rearrange(cos, rearrange_spec)
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin
        return jnp.append(r1, r2, axis=-1)


@typechecked
@shardtypes.scope
def rms_norm(x: bf16[b"*shape"]) -> bf16[b"*shape"]:
    mean2 = jnp.mean(jax.lax.square(jnp.float32(x)), axis=-1, keepdims=True)
    mean2 = save_for_backward(mean2)
    return jnp.bfloat16(x * jax.lax.rsqrt(mean2 + 1e-6))


@pytree_dataclass
class TensorStats:
    mean: f32[b""]
    rms_norm: f32[b""]
    maxabs: f32[b""]
    meanabs: f32[b""]

    @classmethod
    def from_tensor(cls, x: bf16[b""]) -> "TensorStats":
        x = jnp.float32(lax.stop_gradient(x))
        mean = jax.lax.pmean(jnp.mean(x), ("d", "t", "s"))
        norm = jnp.sqrt(jax.lax.pmean(jnp.mean(jnp.square(x)), ("d", "t", "s")))
        maxabs = jax.lax.pmax(jnp.max(jnp.abs(x)), ("d", "t", "s"))
        meanabs = jax.lax.pmean(jnp.mean(jnp.abs(x)), ("d", "t", "s"))
        return cls(mean=mean, rms_norm=norm, maxabs=maxabs, meanabs=meanabs)


StatsDict = dict[str, TensorStats]

"""The definition of our transformer model.

Dense-attention variant for the SPIRe follow-up: this is the NSA tag's model.py
with all NSA-specific machinery removed (token compression, token selection,
gating, and the 3x K/V projections that fed those branches), keeping the
kv_cache/kv_offset plumbing for autoregressive decoding.

Differences from the NSA tag, in full:
- ModelConfig: nsa_l/nsa_d/nsa_L/nsa_n/nsa_w removed.
- TransformerLayer: phi, k_intrablock_pe, v_intrablock_pe, w_nsa_gate removed;
  w_k, w_v, ln_k lose their leading `3` axis.
- forward_pass: native_sparse_attention replaced by dense masked attention with
  RoPE; kv_cache shape is now [layers, 2, B, Klen, K, D] (no branch axis).
- kv_offset is a PER-SEQUENCE vector: row b of the batch writes its L new
  tokens at cache positions kv_offset[b]..kv_offset[b]+L-1. This is what lets
  speculative decoding advance each sequence by its own number of accepted
  tokens. Uniform-progress callers pass jnp.full((B,), pos).
- RoPE: q is rotated at absolute positions kv_offset[b]..kv_offset[b]+L-1; k is
  rotated over the full cache (positions 0..Klen-1) after the cache update, so
  the cache stores *unrotated* keys, as in the NSA tag.
"""

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

    # StreamingLLM sink + sliding-window attention, applied during TRAINING as
    # well as inference. This is what distinguishes SPIRe's draft (trained sparse)
    # from MagicDec's baseline (dense weights, restricted only at decode).
    #
    # Defaults reproduce dense causal attention exactly, so every existing config,
    # tool, and test is unaffected. Callers still build their own masks; these
    # fields are the single source of truth for what mask to build, so training
    # and decode cannot silently disagree about a draft's attention pattern.
    attention_mask: str = "dense"  # "dense" | "streaming_llm"
    sink_size: int = 0
    window: Optional[int] = None

    # Number of feedback-memory slots each layer mixes over. 0 disables the
    # pathway and makes w_memory a zero-element array, so models that do not use
    # it (target, vanilla draft) carry no extra parameters or storage.
    n_mem: int = 0
    # How the memory vectors enter the model.
    #   "additive": memory is summed into the residual stream at the SAME position.
    #   "kv":       memory REPLACES the keys and values, which is the operator
    #               paper 3.3 actually specifies: z^i_t = Attention(x^i_t, m_<t).
    # "additive" is kept only so existing checkpoints still load; it is not a
    # configuration any paper describes. New SPIRe configs use "kv".
    memory_mode: str = "additive"
    # Training-time block size for the "kv" pathway (paper 3.3 / Figure 2). For a
    # prefix S, positions t <= S-k read TARGET activations and the last k read the
    # draft's own, so with prefixes every k-th position the source of a key
    # depends on which block the QUERY sits in. 0 disables the split, which is the
    # decode-time condition (one source per position, chosen by memory_mask).
    memory_block_k: int = 0
    # Feedback memory (paper 3.3, Fan et al. 2021). With this off, the k positions
    # in flight fall back to ordinary self-attention, which is the Figure 5 arm
    # "without feedback memory" (tau 3.352). With it on they instead read
    #     m^i_t = sum_l softmax(w_i)_l * x^l_t
    # a mix over ALL of the draft's layer states at t, layer 0 being the
    # embedding. Because that includes the FINAL layer's output, m_t cannot be
    # read while position t is still being computed -- hence attention over
    # m_<t strictly, and hence k sequential forward passes during training.
    feedback_memory: bool = False


def streaming_visibility(q_pos: jax.Array, k_pos: jax.Array, sink_size: int, window: int) -> jax.Array:
    """StreamingLLM visibility: causal AND (sink OR within sliding window).

    q_pos, k_pos are absolute positions (broadcastable). `window` includes the
    current token: window=1 means each token sees only itself (plus sinks).

    Lives here, not in decode.py, because training and decode must build the same
    mask from the same code. A SPIRe draft trained under one visibility rule and
    decoded under another would show up only as a depressed tau -- indistinguishable
    from a failed reproduction.
    """
    causal = k_pos <= q_pos
    visible = jnp.logical_or(k_pos < sink_size, k_pos > q_pos - window)
    return jnp.logical_and(causal, visible)


@pytree_dataclass
class TransformerLayer:
    w_q: f32[b"d_model/d/s n_q_per_kv n_kv/t d_head"]
    w_k: f32[b"d_model/d/s n_kv/t d_head"]
    w_v: f32[b"d_model/d/s n_kv/t d_head"]
    w_o: f32[b"d_model/d/s n_q_per_kv n_kv/t d_head"]
    w_gate: f32[b"d_model/d/s d_ff/t"]
    w_up: f32[b"d_model/d/s d_ff/t"]
    w_down: f32[b"d_model/d/s d_ff/t"]
    ln_attn_in: f32[b"d_model/t/d/s"]
    ln_q: f32[b"n_q_per_kv n_kv/t d_head/d/s"]
    ln_k: f32[b"n_kv/t d_head/d/s"]
    ln_qkv: f32[b"n_q_per_kv n_kv/t d_head/d/s"]
    ln_attn_out: f32[b"d_model/t/d/s"]
    ln_ffn_in: f32[b"d_model/t/d/s"]
    ln_ffn_out: f32[b"d_model/t/d/s"]


Transformer = Array["layers", TransformerLayer]


@pytree_dataclass
class Model:
    embed: f32[b"vocab/t d_model/d/s"]
    unembed: f32[b"vocab/t d_model/d/s"]
    transformer: Transformer
    ln_embed: f32[b"d_model/t/d/s"]
    ln_final: f32[b"d_model/t/d/s"]
    # SPIRe feedback-memory mixing weights, one row per layer. Zero-initialized so
    # the pathway starts as an exact no-op; shape [layers, 0] when n_mem is 0.
    # Replicated rather than sharded: it is at most layers x (layers+1) scalars.
    w_memory: f32[b"layers n_mem"]

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
        w_k_shape = (h.layers, h.d_model, h.n_kv, h.d_head)
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
        w_memory = jnp.zeros((h.layers, h.n_mem), dtype=jnp.float32)
        ln_embed = jnp.ones((h.d_model,), dtype=jnp.float32)
        ln_attn_in = jnp.ones((h.layers, h.d_model), dtype=jnp.float32)
        ln_q = ln_q_scale * jnp.ones((h.layers, h.n_q_per_kv, h.n_kv, h.d_head), dtype=jnp.float32)
        ln_k = jnp.ones((h.layers, h.n_kv, h.d_head), dtype=jnp.float32)
        ln_qkv = jnp.ones((h.layers, h.n_q_per_kv, h.n_kv, h.d_head), dtype=jnp.float32)
        ln_attn_out = jnp.ones((h.layers, h.d_model), dtype=jnp.float32)
        ln_ffn_in = jnp.ones((h.layers, h.d_model), dtype=jnp.float32)
        ln_ffn_out = jnp.ones((h.layers, h.d_model), dtype=jnp.float32)
        ln_final = jnp.ones((h.d_model,), dtype=jnp.float32)

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
            ),
            ln_embed=ln_embed,
            ln_final=ln_final,
            w_memory=w_memory,
        )
        shardings = make_shardings(Model)
        return jax.tree.map(lax.with_sharding_constraint, arrays, shardings)

    @typechecked
    def unembed_hidden(self, nx):
        """Project an already-normed final hidden state to logits.

        Split out of forward_pass so a caller running several passes over one
        sequence can select the positions each pass owns and pay for the
        unembedding exactly once -- see the note at the call site.
        """
        unembed = shardops.all_gather("V/t M/d/s -> V/t M", jnp.bfloat16(self.unembed))
        return shardops.einsum_unreduced(
            "B/d L/s M, V/t M -> B/d L/s V/t", nx, unembed, preferred_element_type=jnp.float32
        )

    @staticmethod
    def mix_memory(w_memory, states):
        """m^i_t = sum_l softmax(w_i)_l * x^l_t   (paper 3.3).

        states: [layers+1, B, L, M] -- slot 0 the post-embedding state, slot l the
        output of layer l-1. w_memory: [layers, layers+1], one softmax row per
        draft layer; the paper writes it as exp(w_il) normalized over l, which is
        a softmax. Returns [layers, B, L, M].

        Unlike Fan et al. the paper shares neither the key/value projections nor
        the memory vectors across layers, which is why the weights are a matrix
        rather than a vector.
        """
        wts = jax.nn.softmax(jnp.float32(w_memory), axis=-1)      # [layers, layers+1]
        return jnp.bfloat16(jnp.einsum("il,lbtm->ibtm", wts, jnp.float32(states)))

    def memory_to_kv(self, h: ModelConfig, memory):
        """Project memory vectors into this model's keys and values.

        SPIRe's draft reads keys and values from memory vectors rather than from
        its own hidden states (paper 3.3), and for every position the TARGET has
        already processed, that memory IS a target activation. Two consequences,
        both of which this method exists to express:

        1. The draft never runs a forward pass over the prompt. Its cache over any
           committed prefix is a pure projection of activations the target
           computed during prefill and verification.
        2. After each verification round the committed positions' cache entries
           must be REWRITTEN from the target's activations. Skipping that leaves
           self-sourced entries in the cache, and with a 64-token window every
           position the draft can still see is a generated one -- so after ~64
           tokens the entire visible cache would be the wrong source.

        The appendix prices exactly this and nothing else as SPIRe's extra draft
        cost: `FLOPs_draft += 2 * N_draft_kv_params * B`, "FLOPs to project 1
        memory vector into a key and a value for the next iteration".

        memory: [n_mem, B/d, L/s, M/t]; slot j feeds draft layer j.
        Returns [layers, 2, B/d, L, K/t, D], laid out like `kv_cache`.
        """
        assert h.memory_mode == "kv", "memory_to_kv is the kv pathway's projection"
        assert memory.shape[0] >= h.layers, (
            f"need one memory slot per layer: got {memory.shape[0]} for {h.layers} layers")

        @typechecked
        def body(_, scanned: Tuple):
            layer_weights, mem_j = scanned
            gm = shardops.all_gather("B/d L/s M/t -> B/d L/s M", mem_j)
            ln_attn_in = shardops.all_gather("M/t/d/s -> M", jnp.float32(layer_weights.ln_attn_in))
            nkv = jnp.bfloat16(rms_norm(gm) * ln_attn_in)
            w_k = shardops.all_gather("M/d/s K/t D -> M K/t D", jnp.bfloat16(layer_weights.w_k))
            w_v = shardops.all_gather("M/d/s K/t D -> M K/t D", jnp.bfloat16(layer_weights.w_v))
            k = shardops.einsum_unreduced("B/d L/s M, M K/t D -> B/d L/s K/t D", nkv, w_k)
            v = shardops.einsum_unreduced("B/d L/s M, M K/t D -> B/d L/s K/t D", nkv, w_v)
            ln_k = shardops.all_gather("K/t D/d/s -> K/t D", jnp.float32(layer_weights.ln_k))
            k = jnp.bfloat16(rms_norm(k) * ln_k)
            k = shardops.all_gather("B/d L/s K/t D -> B/d L K/t D", k)
            v = shardops.all_gather("B/d L/s K/t D -> B/d L K/t D", v)
            return None, jnp.stack([k, v], axis=0)

        _, kv = jax.lax.scan(body, None, (self.transformer, memory[: h.layers]))
        return kv

    def forward_pass(
        self,
        h: ModelConfig,
        ids: u32[b"B/d L/s"],
        attention_mask: bool_[b"B/d L/s Klen"],
        rng: Optional[PRNGKey] = None,
        kv_cache: Optional[bf16[b"layers 2 B/d Klen K/t D"]] = None,
        kv_offset: Optional[i32[b"B/d"]] = None,
        kv_write_index: Optional[i32[b"B/d"]] = None,
        rope_q_positions: Optional[i32[b"B/d L/s"]] = None,  # explicit RoPE positions for queries
        rope_k_positions: Optional[i32[b"B/d Klen"]] = None,  # explicit RoPE positions for cache keys
        rope_table_len: Optional[int] = None,
        emit_activations: bool = False,
        memory: Optional[bf16[b"n_mem B/d L/s M/t"]] = None,
        w_memory: Optional[f32[b"layers n_mem"]] = None,
        memory_mask: Optional[bool_[b"B/d L/s"]] = None,
        memory_block_k: Optional[int] = None,
        self_memory: Optional[bf16[b"layers B/d L/s M/t"]] = None,
        hidden_only: bool = False,
    ) -> Tuple:
        """Returns (logits, kv_cache, stats), plus per-layer residual-stream
        activations as a 4th element when `emit_activations` is set.

        Those activations are what SPIRe's target-activation substitution feeds into
        the draft: draft layer j consumes the target's layer-(j+6-1) output, mirroring
        the pruned-init correspondence where draft layer j IS target layer j+6.
        Emission is gated by a static flag and appends rather than replaces, so all
        twelve existing call sites are unaffected and pay nothing.

        Return type is a bare Tuple because the arity depends on that flag, which
        @typechecked cannot express.

        `memory` + `w_memory` are SPIRe's feedback-memory pathway: layer j adds
        `sum_m w_memory[j, m] * memory[m]` into its residual stream before the
        pre-attention norm. The bank is shared across layers (so it is closed over)
        while the mixing weights are per-layer (so they ride the scan). During
        training the bank holds the TARGET's activations -- the substitution the
        paper prices at 0.393 tau; at inference it must hold the draft's own, since
        no target is available. With w_memory all zeros this is an exact no-op,
        which is how it should be initialized so it cannot regress a working draft."""
        ##### Initial embedding lookup.
        embed = shardops.all_gather("V/t M/d/s -> V/t M", jnp.bfloat16(self.embed))
        x = shardops.index_unreduced("[V/t] M, B/d L/s -> B/d L/s M", embed, ids, use_onehot=is_tpu())
        x *= math.sqrt(h.d_model)
        ln_embed = shardops.all_gather("M/t/d/s -> M", jnp.float32(self.ln_embed))
        x = jnp.bfloat16(rms_norm(x) * ln_embed)
        x = shardops.psum_scatter("B/d L/s M -> B/d L/s M/t", x)

        # A ring slot's contents are unrelated to its index, so a multi-token
        # write would have to wrap; every ring write in this codebase is a single
        # decode step, and the multi-token writes (prefill, post-verification
        # refresh) are scattered explicitly by the caller.
        assert kv_write_index is None or ids.shape[1] == 1, (
            f"kv_write_index is for single-token ring writes, got L={ids.shape[1]}")
        Klen = attention_mask.shape[2]
        # The RoPE table is indexed by POSITION, which a compact sliding-window
        # cache decouples from cache size: a 65-slot ring still carries absolute
        # positions in the thousands. Sizing the table by Klen would silently
        # clamp every position past the end of the cache -- not an error, just
        # wrong rotations, and only for the long contexts the method exists to
        # serve. Callers holding positions beyond their cache pass the bound.
        rope_table = RopeTable.create(
            Klen if rope_table_len is None else rope_table_len, h)

        layer_rngs = jax.random.split(fold_in_str(rng, "layer"), h.layers) if rng is not None else None

        @typechecked
        def dense_attention(
            q: bf16[b"B/d L/s Q K/t D"],
            k: bf16[b"B/d Klen K/t D"],
            v: bf16[b"B/d Klen K/t D"],
        ) -> bf16[b"B/d L/s Q K/t D"]:
            # Rotate queries at their absolute positions; rotate keys over the full cache.
            # The cache stores unrotated keys, so this is correct for both training
            # (kv_offset None, L == Klen) and decoding (kv_offset[b] = row b's write position).
            # Explicit rope positions override the absolute-position scheme; this
            # supports MagicDec's positions-within-the-cache convention, where a key's
            # effective position is its rank in the compacted sparse cache.
            if rope_q_positions is not None:
                q = jnp.bfloat16(rope_table.apply_positions(q, rope_q_positions))
            elif kv_offset is None:
                q = jnp.bfloat16(rope_table.apply("L D -> 1 L 1 1 D", q, None))
            else:
                q_positions = kv_offset[:, jnp.newaxis] + jnp.arange(q.shape[1])[jnp.newaxis, :]  # [B, L]
                q = jnp.bfloat16(rope_table.apply_positions(q, q_positions))
            if rope_k_positions is not None:
                k = jnp.bfloat16(rope_table.apply_rows(k, rope_k_positions))
            else:
                k = jnp.bfloat16(rope_table.apply("L D -> 1 L 1 D", k, None))
            # spec = "B/d L/s Q K/t D, B/d Klen K/t D -> B/d L/s Klen Q K/t"
            logits = jnp.einsum("b q Q K D, b k K D -> b q k Q K", q, k, preferred_element_type=jnp.float32)
            mask = attention_mask[:, :, :, jnp.newaxis, jnp.newaxis]
            masked_logits = jnp.where(mask, logits, -jnp.inf)
            # Padded cache rows can be fully masked in theory; nan_to_num keeps those safe.
            probs = jnp.nan_to_num(jax.nn.softmax(jnp.float32(masked_logits), axis=2), 0)
            # spec = "B/d L/s Klen Q K/t, B/d Klen K/t D -> B/d L/s Q K/t D"
            return jnp.einsum("b q k Q K, b k K D -> b q Q K D", jnp.bfloat16(probs), v)

        @typechecked
        def two_source_attention(
            q: bf16[b"B/d L/s Q K/t D"],
            k1: bf16[b"B/d Klen K/t D"],
            v1: bf16[b"B/d Klen K/t D"],
            m1: bool_[b"B/d L/s Klen"],
            k2: bf16[b"B/d Klen K/t D"],
            v2: bf16[b"B/d Klen K/t D"],
            m2: bool_[b"B/d L/s Klen"],
        ) -> bf16[b"B/d L/s Q K/t D"]:
            """Attention over two key/value sources under one softmax.

            SPIRe's draft reads TARGET activations for positions the target has
            processed and its OWN states for the k tokens in flight (paper 3.3).
            During training every k-th prefix is a separate example, so whether a
            key is target-sourced depends on the QUERY's block -- which no single
            key/value tensor can represent. Two tensors with complementary masks
            can, provided they share one softmax denominator.

            m1 and m2 must be disjoint; their union is the ordinary attention
            mask. When k1 == k2 and v1 == v2 this returns exactly what
            dense_attention returns for that union, which is how it is tested.
            """
            if rope_q_positions is not None:
                q = jnp.bfloat16(rope_table.apply_positions(q, rope_q_positions))
            elif kv_offset is None:
                q = jnp.bfloat16(rope_table.apply("L D -> 1 L 1 1 D", q, None))
            else:
                q_positions = kv_offset[:, jnp.newaxis] + jnp.arange(q.shape[1])[jnp.newaxis, :]
                q = jnp.bfloat16(rope_table.apply_positions(q, q_positions))
            def rot(kk):
                if rope_k_positions is not None:
                    return jnp.bfloat16(rope_table.apply_rows(kk, rope_k_positions))
                return jnp.bfloat16(rope_table.apply("L D -> 1 L 1 D", kk, None))
            k1, k2 = rot(k1), rot(k2)
            lg1 = jnp.einsum("b q Q K D, b k K D -> b q k Q K", q, k1, preferred_element_type=jnp.float32)
            lg2 = jnp.einsum("b q Q K D, b k K D -> b q k Q K", q, k2, preferred_element_type=jnp.float32)
            lg1 = jnp.where(m1[:, :, :, jnp.newaxis, jnp.newaxis], lg1, -jnp.inf)
            lg2 = jnp.where(m2[:, :, :, jnp.newaxis, jnp.newaxis], lg2, -jnp.inf)
            # One denominator over both sources: take the joint max, then exponentiate.
            mx = jnp.maximum(jnp.max(lg1, axis=2, keepdims=True), jnp.max(lg2, axis=2, keepdims=True))
            mx = jnp.where(jnp.isfinite(mx), mx, 0.0)   # a row masked everywhere
            e1 = jnp.exp(lg1 - mx)
            e2 = jnp.exp(lg2 - mx)
            denom = jnp.sum(e1, axis=2, keepdims=True) + jnp.sum(e2, axis=2, keepdims=True)
            p1 = jnp.nan_to_num(e1 / denom, 0)
            p2 = jnp.nan_to_num(e2 / denom, 0)
            out1 = jnp.einsum("b q k Q K, b k K D -> b q Q K D", jnp.bfloat16(p1), v1)
            out2 = jnp.einsum("b q k Q K, b k K D -> b q Q K D", jnp.bfloat16(p2), v2)
            return jnp.bfloat16(out1 + out2)

        ##### Transformer blocks.
        @explicit_activation_checkpointing
        @typechecked
        def loop_body(
            x_carry,
            # Bare Tuple: gains a per-layer w_memory row when the memory pathway is on.
            scanned_var: Tuple,
            # Bare Tuple: the per-layer output gains a third element (the residual
            # stream) when emit_activations is set, which a fixed arity cannot express.
        ) -> Tuple:
            if use_bank:
                x, bank = x_carry
            else:
                x = x_carry
            kv_src = None    # None => keys and values come from x, as usual
            if use_memory:
                layer_weights, kv_layer, layer_rng_key, w_row, slot_idx = scanned_var
                # Feedback memory into the residual stream. The bank holds the
                # n_layer+1 residual states: slot 0 is the post-embedding state and
                # slot m is layer m-1's output, so slot j is exactly what layer j
                # consumes. The depth-causal weights (built below) zero every slot
                # above j, so a layer can only read states beneath it.
                #
                # SOURCE of the bank differs by phase, and that is the whole
                # mechanism. Training substitutes the TARGET's corresponding states
                # (draft layer j is target layer j+first, so its input is target
                # layer j+first-1's output). Inference has no target for the tokens
                # being drafted, so the draft feeds back its OWN states -- which is
                # what makes it "feedback" memory rather than a target side-channel.
                # The bank always accumulates the draft's OWN states. Where the
                # substitution applies we read the TARGET's instead. Blending at
                # read time (rather than at write) keeps the self-accumulation
                # intact, which is what the unmasked positions must consume.
                #
                # memory_mask is the rollout: True where the target has genuinely
                # processed that position, False for the k in-flight positions the
                # draft must handle on its own. Training under the same split the
                # decoder will face is what `train_rollout_k` is for -- without it
                # the draft leans on a signal that vanishes at deployment.
                if kv_memory:
                    # ---- SPIRe 3.3, the operator as written ----
                    # m^i_t = y_t^{i+first-1} wherever the TARGET has processed
                    # position t. The caller has already sliced `memory` so that
                    # slot i is exactly what draft layer i needs, so this is a
                    # DIRECT substitution of one activation -- not a weighted mix
                    # over slots. The mix is the draft's own feedback memory, and
                    # it applies only to the k positions the draft is speculating,
                    # where no target activation exists.
                    #
                    # These vectors then replace the KEYS AND VALUES below. That
                    # is the whole difference from the additive pathway, which
                    # summed a memory vector into the residual stream at the same
                    # position and is not an operator any paper describes.
                    if mem_arr is not None:
                        sub = jax.lax.dynamic_index_in_dim(
                            jnp.bfloat16(mem_arr), slot_idx, axis=0, keepdims=False
                        )
                        if mem_mask_b is None or (
                            memory_block_k if memory_block_k is not None else h.memory_block_k
                        ):
                            kv_src = sub                       # target everywhere
                        else:
                            # Unsubstituted positions fall back to the draft's own
                            # residual state, i.e. ordinary self-attention there.
                            # That is the "without feedback memory" arm of the
                            # paper's Figure 5 ablation (tau 3.352), and it is what
                            # decode does for tokens still in flight.
                            kv_src = jnp.where(mem_mask_b[0], sub, x)
                else:
                    if mem_arr is None:
                        eff = bank
                    elif mem_mask_b is None:
                        eff = jnp.bfloat16(mem_arr)          # full substitution
                    else:
                        eff = jnp.where(mem_mask_b, jnp.bfloat16(mem_arr), bank)
                    mem_mix = jnp.einsum("m,mbld->bld", w_row, jnp.float32(eff))
                    x = x + jnp.bfloat16(mem_mix)
            else:
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
            w_k = shardops.all_gather("M/d/s K/t D -> M K/t D", jnp.bfloat16(layer_weights.w_k))
            w_v = shardops.all_gather("M/d/s K/t D -> M K/t D", jnp.bfloat16(layer_weights.w_v))
            # Queries always come from the draft's own hidden state; only the keys
            # and values are re-sourced. With kv_src None (every non-SPIRe path)
            # nkv IS nx, so this is bit-identical to what it replaced.
            # In block mode `k`/`v` are the SELF source and the target source is
            # projected separately below, so nkv does not come from kv_src.
            #
            # Without feedback memory the self source is the draft's own hidden
            # state, i.e. ordinary self-attention over the k in-flight positions.
            # With it, the self source is the feedback memory computed for those
            # positions by an EARLIER forward pass -- supplied here rather than
            # derived, because m_t depends on the final layer's output at t and so
            # cannot be formed during the pass that produces it.
            if self_memory is not None and block_masks is not None:
                sm = jax.lax.dynamic_index_in_dim(
                    jnp.bfloat16(self_memory), slot_idx, axis=0, keepdims=False
                )
                gsm = shardops.all_gather("B/d L/s M/t -> B/d L/s M", sm)
                nkv = jnp.bfloat16(rms_norm(gsm) * ln_attn_in)
            elif kv_src is None or block_masks is not None:
                nkv = nx
            else:
                gkv = shardops.all_gather("B/d L/s M/t -> B/d L/s M", kv_src)
                nkv = jnp.bfloat16(rms_norm(gkv) * ln_attn_in)
            k = shardops.einsum_unreduced("B/d L/s M, M K/t D -> B/d L/s K/t D", nkv, w_k)
            v = shardops.einsum_unreduced("B/d L/s M, M K/t D -> B/d L/s K/t D", nkv, w_v)
            k = save_for_backward(k)
            v = save_for_backward(v)
            ln_k = shardops.all_gather("K/t D/d/s -> K/t D", jnp.float32(layer_weights.ln_k))
            k = jnp.bfloat16(rms_norm(k) * ln_k)
            k = shardops.all_gather("B/d L/s K/t D -> B/d L K/t D", k)
            v = shardops.all_gather("B/d L/s K/t D -> B/d L K/t D", v)
            if kv_layer is not None:
                prev_k, prev_v = kv_layer
                # WHERE a token's key and value land is not the same question as
                # WHICH position it is. A dense cache makes them the same number,
                # so kv_offset served both. A compact sliding-window cache does
                # not: the slot is `sink + (pos - sink) % window` while the
                # position stays absolute, and RoPE still needs the position.
                # kv_write_index carries the slot; kv_offset keeps its meaning.
                write_at = kv_offset if kv_write_index is None else kv_write_index
                update_row = jax.vmap(lambda prev, new, off: jax.lax.dynamic_update_slice(prev, new, (off, 0, 0)))
                k = update_row(prev_k, k, write_at)
                v = update_row(prev_v, v, write_at)
            if block_masks is None:
                qkv = dense_attention(q, k, v)
            else:
                # Block split: keys before the query's block come from the target,
                # keys inside it from the draft itself. kv_src holds the target
                # activations here (never blended), because which of the two applies
                # is decided per (query, key) by the masks, not per position.
                m_tgt, m_self = block_masks
                gsel = shardops.all_gather("B/d L/s M/t -> B/d L/s M", kv_src)
                nsel = jnp.bfloat16(rms_norm(gsel) * ln_attn_in)
                k_t = shardops.einsum_unreduced("B/d L/s M, M K/t D -> B/d L/s K/t D", nsel, w_k)
                v_t = shardops.einsum_unreduced("B/d L/s M, M K/t D -> B/d L/s K/t D", nsel, w_v)
                k_t = jnp.bfloat16(rms_norm(k_t) * ln_k)
                k_t = shardops.all_gather("B/d L/s K/t D -> B/d L K/t D", k_t)
                v_t = shardops.all_gather("B/d L/s K/t D -> B/d L K/t D", v_t)
                qkv = two_source_attention(q, k_t, v_t, m_tgt, k, v, m_self)
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

            if use_bank:
                # Write this layer's output into slot j+1 for the layers above.
                # Only the additive pathway carries a bank; the "kv" pathway reads
                # one slot per layer and never accumulates the draft's own states.
                # Under external (training) memory the bank is frozen -- the target's
                # states are the supervision and must not be overwritten by ours.
                bank_out = jax.lax.dynamic_update_slice(
                    bank, jnp.bfloat16(x)[jnp.newaxis], (slot_idx + 1, 0, 0, 0)
                )
                carry_out = (x, bank_out)
            else:
                carry_out = x
            if emit_activations:
                return carry_out, (kv_layer, tensor_stats, x)
            return carry_out, (kv_layer, tensor_stats)

        # Two memory phases, one pathway:
        #   external (training): `memory` supplies the TARGET's states -- the
        #       substitution the paper specifies as m_t^i = y_t^{i+6-1}.
        #   self (inference): no target exists for tokens being drafted, so the
        #       draft feeds back its OWN residual states, accumulated in `bank`
        #       as the layer scan proceeds. This is what makes it FEEDBACK memory;
        #       the target activations are training supervision, not a runtime input.
        # w_memory defaults to the model's OWN field. It is a weight, not a caller
        # argument, and requiring callers to pass it means any decode path that
        # forgets silently runs a memory-trained draft without its memory -- which
        # is precisely the failure that produced eval loss 12.88 and tau 1.000.
        if w_memory is None and h.n_mem > 0:
            w_memory = self.w_memory
        use_memory = w_memory is not None and h.n_mem > 0
        assert h.memory_mode in ("additive", "kv"), f"unknown memory_mode {h.memory_mode!r}"
        # The "kv" pathway reads one slot per layer and never accumulates the
        # draft's own states, so it does not need the bank -- which is [n_mem, B,
        # L, M] and is not free to carry through a scan under autodiff.
        kv_memory = use_memory and h.memory_mode == "kv"
        use_bank = use_memory and not kv_memory
        if use_memory:
            assert w_memory is not None, "memory requires w_memory"
            n_mem = memory.shape[0] if memory is not None else h.n_mem
            if h.memory_mode == "kv":
                assert n_mem >= h.layers, (
                    f"kv memory reads one slot per layer, so n_mem ({n_mem}) must be "
                    f">= layers ({h.layers})")
            assert w_memory.shape == (h.layers, n_mem), (
                f"w_memory must be [layers={h.layers}, n_mem={n_mem}], got {w_memory.shape}"
            )
            # DEPTH-CAUSAL MASK. Slot m holds the state at depth m; draft layer j
            # sits at depth j and may only read states strictly beneath it, i.e.
            # m <= j. Without this the top slot carries the TARGET'S FINAL HIDDEN
            # STATE, and since pruned init hands the draft a copy of the target's
            # unembedding, the draft can route that state straight to the output and
            # reproduce the target's distribution without learning anything. It does
            # exactly that if allowed: an unmasked run put 0.38 and 0.13 of its
            # weight on the final-layer slot and ~0.0004 everywhere else, drove
            # training loss BELOW a correct run, and then scored tau = 1.000 with
            # acceptance 0.000 at decode, where the leaked state is unavailable.
            #
            # Masked entries receive zero gradient and stay at their zero init, so
            # a trained w_memory should show an all-zero upper triangle -- which is
            # a cheap post-hoc check that the mask was actually in force.
            depth = jnp.arange(h.layers)[:, None]
            slot = jnp.arange(n_mem)[None, :]
            w_causal = jnp.where(slot <= depth, w_memory, 0.0)
            # Bank of residual states: slot 0 is the post-embedding state, slot m is
            # layer m-1's output. Under self-memory only slot 0 is known up front;
            # each layer writes its own output into slot j+1 as the scan proceeds,
            # so layer j always finds slots 0..j filled and the causal mask makes
            # the still-empty ones unreachable.
            bank0 = jnp.zeros((n_mem,) + x.shape, dtype=jnp.bfloat16)
            bank0 = bank0.at[0].set(jnp.bfloat16(x))
            mem_arr = memory
            # broadcast [B, L] -> [n_mem, B, L, M] for the read-time blend
            mem_mask_b = (
                memory_mask[jnp.newaxis, :, :, jnp.newaxis] if memory_mask is not None else None
            )
            # Per-(query, key) split of the ordinary mask into "the target has
            # processed this key" and "the draft produced it itself". Positions
            # only, so it is built once rather than per layer.
            block_masks = None
            bk = memory_block_k if memory_block_k is not None else h.memory_block_k
            if kv_memory and bk and memory is not None:
                assert shardops.axis_size("s") == 1, "block-split memory needs an unsharded sequence axis"
                Lq = attention_mask.shape[1]
                qpos = jnp.arange(Lq)[jnp.newaxis, :, jnp.newaxis]
                kpos = jnp.arange(attention_mask.shape[2])[jnp.newaxis, jnp.newaxis, :]
                block_start = (qpos // bk) * bk
                from_target = kpos < block_start
                self_vis = jnp.logical_not(from_target)
                if h.feedback_memory:
                    # z^i_t = Attention(x^i_t, m_<t): STRICTLY previous positions.
                    # m_t is a mix over every layer state at t including the last,
                    # so it does not exist until the pass at t has finished.
                    self_vis = jnp.logical_and(self_vis, kpos < qpos)
                block_masks = (
                    jnp.logical_and(attention_mask, from_target),
                    jnp.logical_and(attention_mask, self_vis),
                )
            scanned_vars = (self.transformer, kv_cache, layer_rngs, w_causal, jnp.arange(h.layers))
            init_carry = (jnp.bfloat16(x), bank0) if use_bank else jnp.bfloat16(x)
        else:
            block_masks = None
            scanned_vars = (self.transformer, kv_cache, layer_rngs)
            init_carry = jnp.bfloat16(x)
        embed_state = jnp.bfloat16(x)   # slot 0 of the feedback-memory state stack
        if emit_activations:
            carry, (kv_cache, ts, layer_acts) = jax.lax.scan(loop_body, init_carry, scanned_vars)
        else:
            carry, (kv_cache, ts) = jax.lax.scan(loop_body, init_carry, scanned_vars)
            layer_acts = None
        x = carry[0] if use_bank else carry

        ##### Final layernorm and output projection.
        x = shardops.all_gather("B/d L/s M/t -> B/d L/s M", x)
        ln_final = shardops.all_gather("M/t/d/s -> M", jnp.float32(self.ln_final))
        nx = jnp.bfloat16(rms_norm(x) * ln_final)
        if hidden_only:
            # Return the normed hidden state and let the caller unembed. A caller
            # running k passes over one sequence needs only L/k positions from
            # each, and the logits tensor is by far the largest thing here --
            # 13.2GB at B=64, L=1024, V=50304 in f32, against 67MB for the hidden
            # state. Materialising it per pass is what put a 4-pass
            # feedback-memory step 117GB over an 80GB card.
            logits = nx
        else:
            logits = self.unembed_hidden(nx)

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
            }
        )
        if not hidden_only:
            tensor_stats["out.logits"] = TensorStats.from_tensor(logits)

        if emit_activations:
            # 5th element: the post-embedding state, slot 0 of the feedback-memory
            # state stack. APPENDED rather than prepended so that every existing
            # consumer's `out[3]` slice of layer outputs is untouched.
            return logits, kv_cache, tensor_stats, layer_acts, embed_state
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

    def apply_positions(self, x, positions):
        """Rotate x[B, L, Q, K, D] at per-row absolute positions[B, L] (gather-based)."""
        x1, x2 = jnp.split(x, 2, axis=-1)
        sin = self.sin[positions][:, :, jnp.newaxis, jnp.newaxis, :]
        cos = self.cos[positions][:, :, jnp.newaxis, jnp.newaxis, :]
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin
        return jnp.append(r1, r2, axis=-1)

    def apply_rows(self, x, positions):
        """Rotate x[B, L, ...heads..., D] at per-row absolute positions[B, L]."""
        x1, x2 = jnp.split(x, 2, axis=-1)
        extra = x.ndim - 3  # head dims between L and D
        sin = jnp.take(self.sin, positions, axis=0).reshape(positions.shape + (1,) * extra + (-1,))
        cos = jnp.take(self.cos, positions, axis=0).reshape(positions.shape + (1,) * extra + (-1,))
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin
        return jnp.append(r1, r2, axis=-1)

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
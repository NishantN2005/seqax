"""Speculative decoding (draft-verify with Leviathan-style rejection sampling).

Structure of one round of speculation, starting from per-row absolute position
`pos` (the position of `cur_tok`, the most recently committed token, which is
not yet written to either cache):

  1. DRAFT: the draft model produces draft tokens d_1..d_k at positions
     pos+1..pos+k and their full distributions q_1..q_k. It runs k+1
     single-token steps: first prev_tok at pos-1, then cur_tok at pos
     (yielding d_1), then d_1..d_{k-1}. Feeding prev_tok closes a cache
     hole: when all k drafts of the previous round were accepted, d_k was
     never fed to the draft, so its KV is missing; prev_tok is exactly that
     token. On rejection rounds the rewrite is redundant but writes
     identical values. Every draft call has L == 1, which is what makes
     MagicDec's per-key rope positions (below) exactly expressible.
     Cost note: the draft therefore performs k+1 forward passes per round,
     not the k assumed by the SPIRe cost model's k * t_draft term — a real
     implementation-vs-model gap to report.
  2. VERIFY: the target runs ONE forward pass over [cur_tok, d_1..d_{k-1}, d_k]
     (k+1 tokens) at offset pos, producing target distributions p_1..p_{k+1}
     for positions pos+1..pos+k+1.
  3. ACCEPT/REJECT (temperature > 0): d_i is accepted with probability
     min(1, p_i(d_i) / q_i(d_i)); at the first rejection index j, the
     replacement token is sampled from norm(max(0, p_j - q_j)). If all k are
     accepted, a bonus token is sampled from p_{k+1}. Padding q with a zero
     distribution at index k+1 makes the bonus a special case of residual
     sampling. This yields samples EXACTLY from the target distribution
     (Leviathan et al. 2023, Theorem 1).
     At temperature == 0 the scheme degenerates to: accept d_i iff
     d_i == argmax p_i; replacement/bonus = argmax p at the first mismatch.
  4. COMMIT: n = (#leading accepts) + 1 tokens are committed; pos advances by
     n per row; the committed final token becomes cur_tok.

Cache discipline (why rollback is free): both caches are fixed-size and
written at per-row kv_offset. After a rejection, entries beyond the committed
prefix are stale, but every stale entry sits at a position strictly greater
than any query that could attend to it before it is overwritten by the next
round's writes — causality masks stale state until it is replaced. No copying,
no cache rollback.

Batching: acceptance counts vary per sequence, so rows advance independently
(per-row kv_offset, per-row masks, per-row output cursors). This is the
correct distribution; lockstep batching would leak rejected tokens into some
rows' prefixes.

Positions: by default RoPE uses original-text (absolute) positions for both
models — the SPIRe convention. With magicdec_rope=True the draft instead uses
positions-within-the-cache (paper footnote 4): for a query at absolute
position q with sink s and window w, let start = max(s, q - w + 1); a key at
absolute position j takes rope position j if j < s, else s + (j - start), and
the query takes s + (q - start). This is the rank each entry would have in a
compacted StreamingLLM cache. It is exact because every draft decode step
has L == 1, so key positions may depend on the single query. Draft PREFILL
keeps absolute positions, matching MagicDec's ordinary dense prefill.

The draft is anything with Model weights + a ModelConfig + a visibility
pattern: a separate small model (vanilla SD), the target itself with a
sink+window mask (MagicDec-style), or later a trained SPIRe draft.
"""

# Set XLA flags before importing JAX
import init_seqax  # noqa: F401  # isort: skip

from typing import Optional

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import shardlib.shardtypes as shardtypes
from decode import streaming_visibility
from model import Model, ModelConfig

shardtypes.register_with_typeguard()


def _dists(logits: jax.Array, temperature: float) -> jax.Array:
    """Token distributions from logits. At temperature 0, a one-hot argmax."""
    if temperature == 0.0:
        return jax.nn.one_hot(jnp.argmax(logits, -1), logits.shape[-1], dtype=jnp.float32)
    return jax.nn.softmax(logits / temperature, axis=-1)


def _key(rng: jax.Array, *folds) -> jax.Array:
    key = jax.random.wrap_key_data(rng)
    for f in folds:
        key = jax.random.fold_in(key, f)
    return jax.random.fold_in(key, jax.lax.axis_index("d"))


def make_speculative_generate(
    h_target: ModelConfig,
    h_draft: ModelConfig,
    prompt_len: int,
    num_rounds: int,
    k: int,
    temperature: float,
    draft_sink: int = 0,
    draft_window: Optional[int] = None,
    draft_prefill_dense: bool = True,
    magicdec_rope: bool = False,
):
    """Build a jitted, sharded speculative generate function.

    Returns spec_generate(target_w, draft_w, prompt[B, P], rng[2]) ->
      tokens  u32[B, 1 + num_rounds*(k+1)]  committed tokens for positions
              P, P+1, ...; entries at index >= n_generated[b] are garbage.
      n_generated i32[B]  number of valid tokens per row.
      n_accepted  i32[B, num_rounds]  leading accepts per round (0..k);
              tau = mean(n_accepted) + 1.

    magicdec_rope=True selects the paper's MagicDec position convention for
    draft decode steps (see module docstring); it requires draft_window.
    draft_window=None -> dense draft attention. draft_prefill_dense chooses the
    mask used to build the draft's cache over the prompt: True matches
    MagicDec-style restrict-at-decode drafting; False matches a draft trained
    with sparse attention (SPIRe-style).
    """
    assert h_target.vocab == h_draft.vocab, "draft and target must share a vocabulary"
    assert not (magicdec_rope and draft_window is None), "magicdec_rope requires a draft_window"
    Pn, R, V = prompt_len, num_rounds, h_target.vocab
    S = 1 + R * (k + 1)  # output buffer length (upper bound on tokens generated)
    # BOS convention, same as decode.make_generate: the model reads ids[i] as the
    # token BEFORE position i, so both caches are prefilled with [BOS, prompt...]
    # and every absolute position is one greater than the prompt index it carries.
    Pf = Pn + 1
    Klen = Pf + S + k + 1  # cache bound: last verify writes k+1 entries past the last committed pos
    d_window = Klen if draft_window is None else draft_window
    d_prefill_window = Klen if (draft_prefill_dense or draft_window is None) else draft_window

    k_pos = jnp.arange(Klen)[jnp.newaxis, jnp.newaxis, :]

    def target_forward(w, ids, pos):
        """Target forward over ids[B, L] at per-row offsets pos[B]; dense causal."""
        lb, L = ids.shape
        q_pos = pos[:, None, None] + jnp.arange(L)[None, :, None]
        mask = jnp.broadcast_to(k_pos <= q_pos, (lb, L, Klen))

        def run(cache):
            with shardtypes.Scope():
                return w.forward_pass(h_target, ids, mask, kv_cache=cache, kv_offset=pos)

        return run

    def draft_forward(w, ids, pos, window, cache_relative_rope=False):
        lb, L = ids.shape
        q_pos = pos[:, None, None] + jnp.arange(L)[None, :, None]
        mask = jnp.broadcast_to(streaming_visibility(q_pos, k_pos, draft_sink, window), (lb, L, Klen))
        rope_q = rope_k = None
        if cache_relative_rope:
            assert L == 1, "cache-relative rope requires single-token queries"
            start = jnp.maximum(draft_sink, pos - window + 1)              # [B]
            flat_k = jnp.arange(Klen)[None, :]                            # [1, Klen]
            rope_k = jnp.where(flat_k < draft_sink, flat_k, draft_sink + (flat_k - start[:, None]))
            rope_k = jnp.maximum(rope_k, 0).astype(jnp.int32)             # masked entries: value irrelevant
            rope_q = (draft_sink + (pos - start))[:, None].astype(jnp.int32)

        def run(cache):
            with shardtypes.Scope():
                return w.forward_pass(
                    h_draft, ids, mask, kv_cache=cache, kv_offset=pos,
                    rope_q_positions=rope_q, rope_k_positions=rope_k,
                )

        return run

    def spec_generate_local(w_t: Model, w_d: Model, prompt: jax.Array, rng: jax.Array):
        lb = prompt.shape[0]
        zero_pos = jnp.zeros((lb,), jnp.int32)

        # ---- Prefill both caches over [BOS || prompt]; first token comes from the target ----
        t_cache = jnp.zeros((h_target.layers, 2, lb, Klen, h_target.n_kv, h_target.d_head), jnp.bfloat16)
        d_cache = jnp.zeros((h_draft.layers, 2, lb, Klen, h_draft.n_kv, h_draft.d_head), jnp.bfloat16)
        ids0 = jnp.concatenate([jnp.zeros((lb, 1), jnp.uint32), prompt], axis=1)  # [lb, Pf]
        t_logits, t_cache, _ = target_forward(w_t, ids0, zero_pos)(t_cache)
        _, d_cache, _ = draft_forward(w_d, ids0, zero_pos, d_prefill_window)(d_cache)

        p0 = _dists(t_logits[:, -1], temperature)
        if temperature == 0.0:
            cur_tok = jnp.argmax(p0, -1).astype(jnp.uint32)
        else:
            cur_tok = jax.random.categorical(_key(rng, 7), jnp.log(p0 + 1e-30), axis=-1).astype(jnp.uint32)

        out = jnp.zeros((lb, S), jnp.uint32)
        out = out.at[:, 0].set(cur_tok)

        # ---- One round of speculation ----
        def round_body(carry, round_idx):
            t_cache, d_cache, cur_tok, prev_tok, pos, out = carry

            # 1. DRAFT k tokens autoregressively. First step feeds
            # [prev_tok, cur_tok] to close the full-accept cache hole (see
            # module docstring); remaining k-1 steps feed one token each.
            def draft_sample(logits, i):
                q = _dists(logits, temperature)
                if temperature == 0.0:
                    nxt = jnp.argmax(q, -1).astype(jnp.uint32)
                else:
                    nxt = jax.random.categorical(
                        _key(rng, 11, round_idx, i), jnp.log(q + 1e-30), axis=-1
                    ).astype(jnp.uint32)
                return nxt, q

            # Hole-closing step: rewrite prev_tok's KV at pos-1. Its logits are
            # discarded (they predict pos, which is already committed as cur_tok).
            _, d_cache, _ = draft_forward(w_d, prev_tok[:, None], pos - 1, d_window, magicdec_rope)(d_cache)
            logits0, d_cache, _ = draft_forward(w_d, cur_tok[:, None], pos, d_window, magicdec_rope)(d_cache)
            d1, q1 = draft_sample(logits0[:, -1], jnp.int32(0))

            def draft_step(dc, i):
                d_cache, tok, dpos = dc
                logits, d_cache, _ = draft_forward(w_d, tok[:, None], dpos, d_window, magicdec_rope)(d_cache)
                nxt, q = draft_sample(logits[:, -1], i)
                return (d_cache, nxt, dpos + 1), (nxt, q)

            (d_cache, _, _), (d_rest, q_rest) = jax.lax.scan(
                draft_step, (d_cache, d1, pos + 1), jnp.arange(1, k, dtype=jnp.int32)
            )
            d_toks = jnp.concatenate([d1[:, None], jnp.transpose(d_rest, (1, 0))], axis=1)        # [B, k]
            q_dists = jnp.concatenate([q1[:, None], jnp.transpose(q_rest, (1, 0, 2))], axis=1)    # [B, k, V]

            # 2. VERIFY: one target pass over [cur_tok, d_1..d_k].
            ids = jnp.concatenate([cur_tok[:, None], d_toks], axis=1)  # [B, k+1]
            t_logits, t_cache, _ = target_forward(w_t, ids, pos)(t_cache)
            p_dists = _dists(t_logits, temperature)         # [B, k+1, V]; p_i predicts pos+i+1

            # 3. ACCEPT/REJECT.
            p_sel = jnp.take_along_axis(p_dists[:, :k], d_toks[..., None], axis=-1)[..., 0]  # [B, k]
            q_sel = jnp.take_along_axis(q_dists, d_toks[..., None], axis=-1)[..., 0]         # [B, k]
            if temperature == 0.0:
                accept = p_sel > 0.5  # one-hot dists: accepted iff d_i == argmax p_i
            else:
                u = jax.random.uniform(_key(rng, 13, round_idx), (lb, k))
                accept = u * q_sel < p_sel
            n_acc = jnp.sum(jnp.cumprod(accept.astype(jnp.int32), axis=1), axis=1)           # [B], 0..k

            # Replacement (first rejection) or bonus (all accepted): residual sampling.
            q_pad = jnp.concatenate([q_dists, jnp.zeros((lb, 1, V))], axis=1)                # [B, k+1, V]
            p_g = jnp.take_along_axis(p_dists, n_acc[:, None, None], axis=1)[:, 0]           # [B, V]
            q_g = jnp.take_along_axis(q_pad, n_acc[:, None, None], axis=1)[:, 0]             # [B, V]
            residual = jnp.maximum(p_g - q_g, 0.0)
            residual_mass = jnp.sum(residual, axis=-1, keepdims=True)
            residual = jnp.where(residual_mass > 1e-9, residual, p_g)
            if temperature == 0.0:
                final_tok = jnp.argmax(p_g, -1).astype(jnp.uint32)
            else:
                final_tok = jax.random.categorical(
                    _key(rng, 17, round_idx), jnp.log(residual + 1e-30), axis=-1
                ).astype(jnp.uint32)

            # 4. COMMIT: [d_1..d_{n_acc}, final_tok]; entries past n_acc are garbage
            # and are overwritten by later rounds or masked by n_generated.
            slot = jnp.arange(k + 1)[None, :]
            t_vec = jnp.where(
                slot == n_acc[:, None],
                final_tok[:, None],
                jnp.concatenate([d_toks, jnp.zeros((lb, 1), jnp.uint32)], axis=1),
            )
            cursor = pos - Pf + 1  # output index of position pos+1
            out = jax.vmap(lambda row, vec, c: jax.lax.dynamic_update_slice(row, vec, (c,)))(out, t_vec, cursor)

            # prev' = the committed token at new pos - 1: d_{n_acc} if any
            # drafts were accepted, else cur_tok.
            d_last = jnp.take_along_axis(d_toks, jnp.maximum(n_acc - 1, 0)[:, None], axis=1)[:, 0]
            prev_tok = jnp.where(n_acc > 0, d_last, cur_tok)
            pos = pos + n_acc + 1
            return (t_cache, d_cache, final_tok, prev_tok, pos, out), n_acc

        (_, _, _, _, pos, out), n_accepted = jax.lax.scan(
            round_body,
            # cur_tok sits at absolute position Pf (BOS occupies 0, prompt 1..Pn);
            # prev_tok is the token at Pf-1, i.e. the last real prompt token.
            (t_cache, d_cache, cur_tok, prompt[:, -1], jnp.full((lb,), Pf, jnp.int32), out),
            jnp.arange(R, dtype=jnp.int32),
        )
        return out, pos - Pf + 1, jnp.transpose(n_accepted, (1, 0))

    model_spec = shardtypes.make_partition_specs(Model)

    def spec_generate(w_t: Model, w_d: Model, prompt: jax.Array, rng: jax.Array):
        mesh = jax._src.mesh.thread_resources.env.physical_mesh
        fn = jax.experimental.shard_map.shard_map(
            spec_generate_local,
            mesh=mesh,
            in_specs=(model_spec, model_spec, P("d", None), P(None)),
            out_specs=(P("d", None), P("d"), P("d", None)),
            check_rep=False,
        )
        return jax.jit(fn)(w_t, w_d, prompt, rng)

    return spec_generate
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
import numpy as np
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
    klen: Optional[int] = None,
    return_drafts: bool = False,
    compact_draft_cache: bool = False,
    _ring_slack: Optional[int] = None,
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
    # Default Klen is the WORST CASE (every round accepting all k). For a fixed
    # token budget that bound scales with k -- k=8 reserves ~2x what k=1 does for
    # the same tokens, because more slack separates R*(k+1) from what tau actually
    # delivers. Attention READS the whole allocation, so a benchmark comparing
    # depths must pin klen or higher k is charged for a buffer it never fills.
    Klen = klen if klen is not None else Pf + S + k + 1
    d_window = Klen if draft_window is None else draft_window
    d_prefill_window = Klen if (draft_prefill_dense or draft_window is None) else draft_window
    # SPIRe's draft reads keys and values from memory vectors (paper 3.3), and for
    # every position the target has processed that memory is a target activation.
    # So the draft's cache over committed text is a projection of the target's
    # activations, refreshed after every verification -- not something the draft
    # computes for itself.
    draft_kv_memory = (h_draft.memory_mode == "kv" and h_draft.n_mem > 0)
    # Full SPIRe: the k in-flight positions carry feedback memory rather than
    # falling back to ordinary self-attention.
    draft_feedback = draft_kv_memory and h_draft.feedback_memory
    mem_lo = h_target.layers - h_draft.layers - 1

    # ---- Compact sliding-window draft cache -------------------------------
    # SPIRe's central claim is that the draft's KV cache is CONSTANT in the
    # decoding sequence length. A mask over a full-length cache reproduces the
    # distributions exactly but still READS every position, so it can demonstrate
    # tau and nothing about cost -- and the cache the draft does not carry is the
    # entire point of the method. With compact_draft_cache the draft's cache is
    # physically sink + window entries whatever L is.
    #
    # Both paths are kept. They must agree bitwise, and tests/test_ring_buffer.py
    # asserts exactly that; the full-cache path is the oracle.
    ring = compact_draft_cache and draft_window is not None and draft_window < Klen
    # The ring is window + k wide, NOT window. Ordinary autoregressive decoding
    # can hold exactly the window it reads, but a draft writes k tokens ahead
    # SPECULATIVELY. After drafting out to pos+k a window-sized ring has evicted
    # everything older than pos+k-window; if only n_acc of those k are accepted,
    # the next query sits at pos+n_acc+1 and still needs positions back to
    # pos+n_acc+1-window -- which, at n_acc=0, are exactly the ones the rejected
    # writes just destroyed. The extra k slots absorb speculation that gets thrown
    # away. They are RETAINED but not VISIBLE: the mask below still uses the true
    # window, so tau is unaffected.
    #
    # This is a correction to the paper's cost model, which charges the draft for
    # sink + L/window_factor. The honest figure is sink + window + k -- 69 rather
    # than 65 at the paper's window 64 and k=4, a 6% understatement of KV_draft.
    # _ring_slack exists so a test can set it to 0 -- the size the paper's cost
    # model charges for -- and demonstrate that the extra k slots are load-bearing
    # rather than defensive padding.
    _slack = k if _ring_slack is None else _ring_slack
    ring_mod = (draft_window + _slack) if draft_window is not None else Klen
    C_d = (draft_sink + ring_mod) if ring else Klen

    def ring_slot(p):
        """Absolute position -> cache slot. Sink slots are pinned, the rest cycle.

        Identity when not compacting, which is what lets one code path serve both.
        """
        if not ring:
            return p
        return jnp.where(p < draft_sink, p,
                         draft_sink + jnp.mod(p - draft_sink, ring_mod))

    def ring_write(dc, dpos, kv, p):
        """Write one token's [layers, 2, B, 1, K, D] at absolute positions p[B]."""
        sl = ring_slot(p)
        dc = jax.vmap(
            lambda prev, new, off: jax.lax.dynamic_update_slice(prev, new, (0, 0, off, 0, 0)),
            in_axes=(2, 2, 0), out_axes=2,
        )(dc, kv, sl)
        dpos = jax.vmap(lambda row, i, v: row.at[i].set(v))(dpos, sl, p)
        return dc, dpos

    def mark(dpos, p):
        """Record that slot ring_slot(p) now holds absolute position p."""
        sl = ring_slot(p)
        return jax.vmap(lambda row, i, v: row.at[i].set(v))(dpos, sl, p)

    def ring_write_many(dc, dpos, kv, p0, n):
        """Scatter n consecutive positions starting at p0[B]. n is static and small."""
        for i in range(n):
            dc, dpos = ring_write(dc, dpos, kv[:, :, :, i : i + 1], p0 + i)
        return dc, dpos

    k_pos = jnp.arange(Klen)[jnp.newaxis, jnp.newaxis, :]

    def target_forward(w, ids, pos, emit_acts: bool = False):
        """Target forward over ids[B, L] at per-row offsets pos[B]; dense causal.

        `emit_acts` returns per-layer activations as a 4th value, which is how the
        draft gets SPIRe's feedback memory at prefill -- the target processes the
        prompt anyway, so keeping those activations costs nothing."""
        lb, L = ids.shape
        q_pos = pos[:, None, None] + jnp.arange(L)[None, :, None]
        mask = jnp.broadcast_to(k_pos <= q_pos, (lb, L, Klen))

        def run(cache):
            with shardtypes.Scope():
                return w.forward_pass(
                    h_target, ids, mask, kv_cache=cache, kv_offset=pos, emit_activations=emit_acts
                )

        return run

    def draft_forward(w, ids, pos, window, cache_relative_rope=False, memory=None,
                      emit_acts: bool = False, slot_pos=None, dense_write: bool = False):
        """One draft pass. `slot_pos` [B, C_d] is the absolute position each cache
        slot holds, or -1 for empty.

        Visibility is derived from dpos rather than from the slot index, which is
        what makes the compact and full caches one code path: on a full cache
        dpos is arange, and the condition below collapses term for term into
        streaming_visibility.
        """
        lb, L = ids.shape
        q_pos = pos[:, None, None] + jnp.arange(L)[None, :, None]          # [B, L, 1]
        if not dense_write and L == 1:
            # This pass writes the token's own key and value at ring_slot(pos),
            # and a draft without feedback memory attends to that slot -- the
            # diagonal. The slot map therefore has to reflect the write BEFORE
            # the mask is built. A full cache got this for free, because there
            # the slot index and the position are the same number; a ring has to
            # be told. (Under feedback memory the mask is dp < q_pos, so the
            # diagonal is excluded either way and this is a no-op.)
            slot_pos = mark(slot_pos, pos)
        dp = slot_pos[:, jnp.newaxis, :]                                   # [B, 1, C_d]
        mask = jnp.logical_and(
            dp >= 0,                                                       # slot ever written
            jnp.logical_and(
                dp <= q_pos,                                               # causal
                jnp.logical_or(dp < draft_sink, dp > q_pos - window),       # sink or window
            ),
        )
        mask = jnp.broadcast_to(mask, (lb, L, slot_pos.shape[1]))
        if draft_feedback:
            # z^i_t = Attention(x^i_t, m_<t). The query never reads its own slot,
            # because m_t is a mix over every layer state at t including the last
            # and so does not exist until the pass at t has finished. This is the
            # same visibility the block split enforces during training.
            mask = jnp.logical_and(mask, dp < q_pos)
        dp_safe = jnp.maximum(slot_pos, 0).astype(jnp.int32)
        if cache_relative_rope:
            assert L == 1, "cache-relative rope requires single-token queries"
            # MagicDec's convention: a key's effective position is its rank within
            # the live window, not its position in the text (paper footnote 4).
            start = jnp.maximum(draft_sink, pos - window + 1)              # [B]
            rope_k = jnp.where(dp_safe < draft_sink, dp_safe,
                               draft_sink + (dp_safe - start[:, None]))
            rope_k = jnp.maximum(rope_k, 0).astype(jnp.int32)
            rope_q = (draft_sink + (pos - start))[:, None].astype(jnp.int32)
        else:
            rope_k = dp_safe
            rope_q = q_pos[:, :, 0].astype(jnp.int32)

        def run(cache):
            with shardtypes.Scope():
                return w.forward_pass(
                    h_draft, ids, mask, kv_cache=cache, kv_offset=pos,
                    kv_write_index=(ring_slot(pos) if (ring and not dense_write) else None),
                    rope_q_positions=rope_q, rope_k_positions=rope_k,
                    rope_table_len=Klen,
                    memory=memory, w_memory=(w.w_memory if memory is not None else None),
                    emit_activations=emit_acts,
                )

        return run

    def spec_generate_local(w_t: Model, w_d: Model, prompt: jax.Array, rng: jax.Array):
        lb = prompt.shape[0]
        zero_pos = jnp.zeros((lb,), jnp.int32)

        # ---- Prefill both caches over [BOS || prompt]; first token comes from the target ----
        t_cache = jnp.zeros((h_target.layers, 2, lb, Klen, h_target.n_kv, h_target.d_head), jnp.bfloat16)
        d_cache = jnp.zeros((h_draft.layers, 2, lb, C_d, h_draft.n_kv, h_draft.d_head), jnp.bfloat16)
        ids0 = jnp.concatenate([jnp.zeros((lb, 1), jnp.uint32), prompt], axis=1)  # [lb, Pf]
        # SPIRe feedback memory at prefill. Both models consume the SAME committed
        # prompt tokens here, so the target's activations at these positions are
        # valid and already computed -- injecting them costs no extra forward pass
        # and does not touch the speculative-decoding economics.
        #
        # This covers the prompt only. The k in-flight positions the draft
        # speculates have no target activations by construction (the target has not
        # seen those tokens), so they get none -- reading A of fidelity-ledger
        # 3.7. For a 512-token context that is ~508 positions with real memory
        # against the handful without.
        use_mem = h_draft.n_mem > 0
        t_out = target_forward(w_t, ids0, zero_pos, emit_acts=use_mem)(t_cache)
        t_logits, t_cache = t_out[0], t_out[1]
        prefill_mem = None
        if use_mem:
            # Draft layer j is target layer j+first, whose input is layer
            # (j+first-1)'s output; the bank is the window starting there.
            assert mem_lo >= 0 and mem_lo + h_draft.n_mem <= h_target.layers, (
                f"memory window [{mem_lo}, {mem_lo + h_draft.n_mem}) does not fit "
                f"a {h_target.layers}-layer target"
            )
            prefill_mem = t_out[3][mem_lo : mem_lo + h_draft.n_mem]
        # Which absolute position each draft slot holds once the prompt is in.
        # Static: Pf, C_d, sink and window are all Python ints.
        _last = Pf - 1
        _s = np.arange(C_d)
        if ring:
            # For a ring slot, the position it holds is the largest p <= last with
            # p congruent to (sink + r) modulo the window -- i.e. the most recent
            # token that mapped there. Sinks are pinned to their own index.
            _r = _s - draft_sink
            _cand = draft_sink + _r + ring_mod * ((_last - draft_sink - _r) // ring_mod)
            _is_sink = _s < draft_sink
            _src = np.where(_is_sink, _s, _cand)
            _valid = np.where(_is_sink, _s <= _last, (_cand >= draft_sink) & (_cand <= _last))
        else:
            _src, _valid = _s, np.ones(C_d, dtype=bool)
        _src_safe = np.clip(_src, 0, max(Pf - 1, 0))
        d_pos = jnp.broadcast_to(
            jnp.asarray(np.where(_valid, _src, -1), dtype=jnp.int32)[None, :], (lb, C_d))

        if draft_kv_memory:
            # The draft does NOT process the prompt. Its keys and values there are
            # projections of activations the target has already computed, which is
            # the only draft cost the appendix's cost model charges for SPIRe
            # beyond an ordinary forward pass.
            #
            # Gathering BEFORE projecting means the projection runs on C_d
            # positions rather than the whole prompt -- the compact cache is
            # cheaper to build as well as to read.
            with shardtypes.Scope():
                kv0 = w_d.memory_to_kv(h_draft, prefill_mem[:, :, jnp.asarray(_src_safe), :])
            d_cache = kv0
        else:
            # A draft without the memory pathway computes its own prompt keys and
            # values, so it needs a dense pass first; the compaction is a gather
            # afterwards. Prefill is outside the round loop and so outside every
            # timing measurement -- what the cost model charges for is the
            # per-round read, and that reads C_d.
            tmp = jnp.zeros((h_draft.layers, 2, lb, Klen, h_draft.n_kv, h_draft.d_head), jnp.bfloat16)
            tmp_pos = jnp.broadcast_to(jnp.arange(Klen, dtype=jnp.int32)[None, :], (lb, Klen))
            _, tmp, _ = draft_forward(w_d, ids0, zero_pos, d_prefill_window,
                                      memory=prefill_mem, slot_pos=tmp_pos, dense_write=True)(tmp)
            d_cache = tmp[:, :, :, jnp.asarray(_src_safe), :, :] if ring else tmp

        p0 = _dists(t_logits[:, -1], temperature)
        if temperature == 0.0:
            cur_tok = jnp.argmax(p0, -1).astype(jnp.uint32)
        else:
            cur_tok = jax.random.categorical(_key(rng, 7), jnp.log(p0 + 1e-30), axis=-1).astype(jnp.uint32)

        out = jnp.zeros((lb, S), jnp.uint32)
        out = out.at[:, 0].set(cur_tok)

        def draft_fb_step(tok, dpos, dc, dmap):
            """One draft step under feedback memory.

            The pass at position t writes a placeholder key and value at t derived
            from the draft's hidden state. Nothing reads it: attention here is
            over m_<t strictly. We then overwrite that slot with the real memory
            projection, which is what the NEXT drafted token attends to.

            The extra cost is one memory vector projected into a key and a value
            per drafted token -- exactly the term the appendix adds to SPIRe's
            draft FLOPs and nothing more.
            """
            out = draft_forward(w_d, tok[:, None], dpos, d_window, magicdec_rope,
                                emit_acts=True, slot_pos=dmap)(dc)
            logits, dc, acts, emb = out[0], out[1], out[3], out[4]
            states = jnp.concatenate([emb[jnp.newaxis], acts], axis=0)   # [layers+1, B, 1, M]
            m_t = Model.mix_memory(w_d.w_memory, states)                 # [layers, B, 1, M]
            with shardtypes.Scope():
                kv_t = w_d.memory_to_kv(h_draft, m_t)                    # [layers, 2, B, 1, K, D]
            dc, dmap = ring_write(dc, dmap, kv_t, dpos)
            return logits, dc, dmap

        # ---- One round of speculation ----
        def round_body(carry, round_idx):
            t_cache, d_cache, d_pos, cur_tok, prev_tok, pos, out = carry

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
            if not draft_kv_memory:
                # Closes the full-accept cache hole (module docstring). Under the
                # kv pathway that entry is rewritten from the target's activations
                # after every verification, so re-running the draft on it would
                # replace a correct target-sourced entry with a self-sourced one.
                _, d_cache, _ = draft_forward(w_d, prev_tok[:, None], pos - 1, d_window,
                                              magicdec_rope, slot_pos=d_pos)(d_cache)
                d_pos = mark(d_pos, pos - 1)
            if draft_feedback:
                logits0, d_cache, d_pos = draft_fb_step(cur_tok, pos, d_cache, d_pos)
            else:
                logits0, d_cache, _ = draft_forward(w_d, cur_tok[:, None], pos, d_window,
                                                    magicdec_rope, slot_pos=d_pos)(d_cache)
                d_pos = mark(d_pos, pos)
            d1, q1 = draft_sample(logits0[:, -1], jnp.int32(0))

            def draft_step(dc, i):
                d_cache, d_pos, tok, dpos = dc
                if draft_feedback:
                    logits, d_cache, d_pos = draft_fb_step(tok, dpos, d_cache, d_pos)
                else:
                    logits, d_cache, _ = draft_forward(w_d, tok[:, None], dpos, d_window,
                                                       magicdec_rope, slot_pos=d_pos)(d_cache)
                    d_pos = mark(d_pos, dpos)
                nxt, q = draft_sample(logits[:, -1], i)
                return (d_cache, d_pos, nxt, dpos + 1), (nxt, q)

            (d_cache, d_pos, _, _), (d_rest, q_rest) = jax.lax.scan(
                draft_step, (d_cache, d_pos, d1, pos + 1), jnp.arange(1, k, dtype=jnp.int32)
            )
            d_toks = jnp.concatenate([d1[:, None], jnp.transpose(d_rest, (1, 0))], axis=1)        # [B, k]
            q_dists = jnp.concatenate([q1[:, None], jnp.transpose(q_rest, (1, 0, 2))], axis=1)    # [B, k, V]

            # 2. VERIFY: one target pass over [cur_tok, d_1..d_k].
            ids = jnp.concatenate([cur_tok[:, None], d_toks], axis=1)  # [B, k+1]
            t_v = target_forward(w_t, ids, pos, emit_acts=draft_kv_memory)(t_cache)
            t_logits, t_cache = t_v[0], t_v[1]
            if draft_kv_memory:
                # The target has now processed positions pos .. pos+k, so the
                # draft's memory for them is defined. Entries past n_acc belong to
                # rejected tokens; they sit above every later query's position and
                # are masked out until a later round overwrites them.
                mem_v = t_v[3][mem_lo : mem_lo + h_draft.n_mem]
                with shardtypes.Scope():
                    kv_v = w_d.memory_to_kv(h_draft, mem_v)
                d_cache, d_pos = ring_write_many(d_cache, d_pos, kv_v, pos, k + 1)
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
            return (t_cache, d_cache, d_pos, final_tok, prev_tok, pos, out), (n_acc, d_toks, q_dists)

        # `first_drafts` exposes round 0's k proposals so a test can compare the
        # cached decode path against a no-cache forward pass built from TRAINING
        # semantics. Without it the only observable is the committed stream, which
        # the target's accept/reject decisions confound.
        (_, _, _, _, _, pos, out), (n_accepted, all_drafts, all_qs) = jax.lax.scan(
            round_body,
            # cur_tok sits at absolute position Pf (BOS occupies 0, prompt 1..Pn);
            # prev_tok is the token at Pf-1, i.e. the last real prompt token.
            (t_cache, d_cache, d_pos, cur_tok, prompt[:, -1],
             jnp.full((lb,), Pf, jnp.int32), out),
            jnp.arange(R, dtype=jnp.int32),
        )
        if return_drafts:
            # The DISTRIBUTIONS, not just the sampled tokens. With an untrained
            # draft the logits are near-uniform, so an argmax is decided by noise
            # at the 1e-7 level and comparing tokens cannot tell a correct cache
            # from a broken one. Comparing q separates "different summation order"
            # (~1e-7) from "different attention inputs" (~1e-1).
            # EVERY round's draft distributions, not just round 0's. A cache
            # that evicts a live position cannot show it in round 0 -- nothing has
            # been evicted yet -- so a round-0-only comparison is blind to exactly
            # the failure a sliding window is most likely to have.
            return (out, pos - Pf + 1, jnp.transpose(n_accepted, (1, 0)),
                    all_drafts[0], jnp.transpose(all_qs, (1, 0, 2, 3)))
        return out, pos - Pf + 1, jnp.transpose(n_accepted, (1, 0))

    model_spec = shardtypes.make_partition_specs(Model)

    def spec_generate(w_t: Model, w_d: Model, prompt: jax.Array, rng: jax.Array):
        mesh = jax._src.mesh.thread_resources.env.physical_mesh
        fn = jax.experimental.shard_map.shard_map(
            spec_generate_local,
            mesh=mesh,
            in_specs=(model_spec, model_spec, P("d", None), P(None)),
            out_specs=(P("d", None), P("d"), P("d", None))
            + ((P("d", None), P("d", None, None, None)) if return_drafts else ()),
            check_rep=False,
        )
        return jax.jit(fn)(w_t, w_d, prompt, rng)

    return spec_generate
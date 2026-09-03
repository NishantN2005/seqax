"""Main training loop, loads the model, and includes the loss function and optimizer."""

# Set XLA flags before importing JAX
import init_seqax  # noqa: F401  # isort: skip

import datetime
import math
import multiprocessing
import operator
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh
from jax.tree_util import tree_leaves
from omegaconf import DictConfig, OmegaConf
from typeguard import typechecked

import input_loader
import jax_extra
import shardlib.shardops as shardops
import shardlib.shardtypes as shardtypes
import training_io
from init_seqax import is_tpu
from input_loader import ShufflingLoader, TokenBatch
from jax_extra import fold_in_str
from model import MeshConfig, Model, ModelConfig, StatsDict, TensorStats, streaming_visibility
from shardlib.shardtypes import bool_, f32, pytree_dataclass, u32

shardtypes.register_with_typeguard()
OmegaConf.register_new_resolver("eval", eval, use_cache=False)  # Allows expressions like ${eval:${a} + 1} in configs
PRNGKey = u32[b"2"]  # Type annotation to enable run-time typechecking of PRNGKeys by shardtypes


@typechecked
def loss_fn(
    model: Model,
    h: ModelConfig,
    batch: TokenBatch,
    rng: Optional[PRNGKey] = None,
    # Annotated Any, not Model, deliberately: shardtypes' runtime checker binds the
    # `layers` dimension from the student and would reject a teacher of different
    # depth (8 vs 2) as a shape error. It is a Model.
    teacher: Optional[Any] = None,
    h_teacher: Optional[ModelConfig] = None,
    omega: float = 0.5,
) -> Tuple[f32[b""], StatsDict]:
    """Next-token loss, optionally sparse-masked and distilled.

    With `teacher` supplied this computes SPIRe's MixedLoss instead of hard-target
    cross-entropy:

        MixedLoss(w) = w * CE(p_teacher || q_student) + (1 - w) * (-alpha)

    where alpha is Leviathan's expected acceptance probability. For a single token,
    alpha = E_{x~q}[min(1, p(x)/q(x))] = sum_x min(p(x), q(x)) -- the overlap of the
    two distributions, i.e. 1 - TV(p, q). Optimizing it directly targets the
    deployment metric rather than a proxy for it.

    Figure 5 prices these at 0.332 tau for distillation and 0.026 for the alpha term,
    against 3.401 for full SPIRe."""
    # Given sequence-packed targets:
    #   [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
    # we want inputs:
    #   [[0, 1], [0, 3, 4], [0, 6, 7, 8]]
    # which we get by shifting the targets right by 1 and
    # masking sequence-start tokens to 0.
    is_seq_start: bool_[b"B/d L/s"] = batch.is_seq_start
    # Pass along previous token when sharding across sequence "s". First segment is padded with 0.
    inputs: u32[b"B/d L/s"] = shardops.ring_permute_sharded(batch.targets, axis_index=1, mesh_axis="s")
    inputs = jnp.where(is_seq_start, 0, inputs)

    L = inputs.shape[1] * shardops.axis_size("s")

    segment_ids = jnp.cumsum(shardops.all_gather("B/d L/s -> B/d L", is_seq_start), axis=1)
    segment_mask: bool_[b"B/d L L"] = segment_ids[:, :, jnp.newaxis] == segment_ids[:, jnp.newaxis, :]
    causal_mask: bool_[b"1 L L"] = jnp.tril(jnp.ones((L, L), dtype=jnp.bool_), 0)[jnp.newaxis, ...]
    dense_mask: bool_[b"B/d L L"] = jnp.logical_and(segment_mask, causal_mask)

    # SPIRe's draft trains WITH the sparse mask, not just at decode -- that is the
    # whole difference between it and the MagicDec baseline. Build the student's
    # mask from ModelConfig so training and decode cannot disagree.
    if h.attention_mask == "streaming_llm":
        assert h.window is not None, "streaming_llm attention requires model.window"
        pos = jnp.arange(L)
        vis: bool_[b"1 L L"] = streaming_visibility(pos[:, jnp.newaxis], pos[jnp.newaxis, :], h.sink_size, h.window)[
            jnp.newaxis, ...
        ]
        student_mask: bool_[b"B/d L L"] = jnp.logical_and(dense_mask, vis)
    elif h.attention_mask == "dense":
        student_mask = dense_mask
    else:
        raise ValueError(f"unknown attention_mask {h.attention_mask!r}")

    causal_mask: bool_[b"B/d L/s L"] = shardops.shard(student_mask, "B/d L L -> B/d L/s L")

    # ---- Teacher first: it supplies both the distillation targets and, when the
    # student declares memory slots, the activations that fill them. ----
    t_acts = None
    if teacher is not None:
        assert h_teacher is not None, "distillation requires the teacher's ModelConfig"
        # The teacher is the dense-attention target; it must see its own mask, not
        # the student's sparse one, or we would distill from a crippled teacher.
        teacher_mask: bool_[b"B/d L/s L"] = shardops.shard(dense_mask, "B/d L L -> B/d L/s L")
        # shardlib dimension names are global; a teacher of different depth collides
        # with the student's `layers` binding without its own Scope.
        with shardtypes.Scope():
            t_out = teacher.forward_pass(
                h_teacher, inputs, teacher_mask, emit_activations=(h.n_mem > 0)
            )
        t_logits = lax.stop_gradient(t_out[0])
        if h.n_mem > 0:
            t_acts = lax.stop_gradient(t_out[3])

    # ---- SPIRe target-activation substitution ----
    # Pruned init sets draft layer j := target layer j+first, so the input that
    # target layer j+first saw is target layer (j+first-1)'s output. Handing the
    # draft that activation restores the context those layers lost when they were
    # torn out of the stack -- which is what `m_t^i = y_t^{i+6-1}` says, with the
    # paper's 1-indexing. n_mem = layers+1 covers the draft's own residual states
    # (embedding output plus each layer output), whose target counterparts are the
    # window [first-1, first-1+n_mem).
    # No teacher (evaluation, or plain CE training) means no activations to inject,
    # so the pathway is simply off and w_memory is unused. This is NOT an error: it
    # is exactly the decode-time condition, where the target's activations for the
    # positions being drafted do not exist. Asserting here instead crashed a
    # completed 20,841-step run during its final eval pass.
    memory = None
    if h.n_mem > 0 and t_acts is not None:
        first = h_teacher.layers - h.layers
        lo = first - 1
        assert lo >= 0 and lo + h.n_mem <= h_teacher.layers, (
            f"memory window [{lo}, {lo + h.n_mem}) does not fit a {h_teacher.layers}-layer teacher"
        )
        memory = t_acts[lo : lo + h.n_mem]

        # ---- Rollout split: train the draft under the DEPLOYMENT memory condition ----
        # At decode the draft only ever has target activations for prompt positions:
        # its cache over the prompt is built during prefill, where both models
        # consume the same committed tokens. Every position it *generates* is
        # produced with no memory at all -- the target has not seen those tokens,
        # and the newly committed token is one the target produced rather than
        # consumed, so even the first draft step has nothing.
        #
        # Training with memory everywhere therefore teaches the draft to rely on a
        # signal that is absent exactly when it matters. Two runs confirmed this the
        # hard way: the draft learns a 0.418-weighted memory term, then scores
        # eval loss 12.88 and tau ~= 1.00 without it -- unusable, not merely worse.
        #
        # So we zero the memory past a split point, mimicking the prompt/generation
        # boundary. The split is resampled each step so the draft is never tuned to
        # one prompt length. This is our reading of `t <= S-k` and `train_rollout_k`;
        # the paper pins neither the split nor the decode-time source (ledger 3.7).
        if rng is not None:
            split = jax.random.randint(fold_in_str(rng, "rollout_split"), (), L // 4, L + 1)
            keep = (jnp.arange(L) < split)[jnp.newaxis, jnp.newaxis, :, jnp.newaxis]
            memory = memory * keep.astype(memory.dtype)

    with shardtypes.Scope():
        logits, _, tensor_stats = model.forward_pass(
            h, inputs, causal_mask, rng, memory=memory, w_memory=(model.w_memory if h.n_mem > 0 else None)
        )
    max_logits: f32[b"B/d L/s 1"] = lax.pmax(jnp.max(lax.stop_gradient(logits), axis=-1, keepdims=True), "t")
    logits = logits - max_logits
    sum_logits = lax.psum(jnp.sum(jnp.exp(logits), axis=-1, keepdims=True), "t")
    logsumexp = jnp.log(sum_logits)
    logprobs: f32[b"B/d L/s V/t"] = logits - logsumexp
    logprobs_at_targets = shardops.index_unreduced(
        "B/d L/s [V/t], B/d L/s -> B/d L/s", logprobs, batch.targets, use_onehot=is_tpu()
    )
    logprobs_at_targets = shardops.psum_scatter("B/d L/s -> B/d L/s/t", logprobs_at_targets)
    tokens_in_global_batch = logprobs_at_targets.size * jax.lax.psum(1, ("d", "t", "s"))

    if teacher is None:
        return -jnp.sum(logprobs_at_targets) / jnp.float32(tokens_in_global_batch), tensor_stats

    # ---- SPIRe MixedLoss: distillation CE against the frozen target, plus alpha ----
    t_max: f32[b"B/d L/s 1"] = lax.pmax(jnp.max(t_logits, axis=-1, keepdims=True), "t")
    t_logits = t_logits - t_max
    t_logsumexp = jnp.log(lax.psum(jnp.sum(jnp.exp(t_logits), axis=-1, keepdims=True), "t"))
    p: f32[b"B/d L/s V/t"] = jnp.exp(t_logits - t_logsumexp)

    # Both terms follow the hard-target pattern above: compute a partial sum over
    # the local vocab shard, psum_scatter to complete and re-shard it, then divide
    # by the global token count. The outer psum in training_step finishes the job.
    ce_partial: f32[b"B/d L/s"] = -jnp.sum(p * logprobs, axis=-1)
    ce_per_token = shardops.psum_scatter("B/d L/s -> B/d L/s/t", ce_partial)
    distill_ce = jnp.sum(ce_per_token) / jnp.float32(tokens_in_global_batch)

    q: f32[b"B/d L/s V/t"] = jnp.exp(logprobs)
    alpha_partial: f32[b"B/d L/s"] = jnp.sum(jnp.minimum(p, q), axis=-1)
    alpha_per_token = shardops.psum_scatter("B/d L/s -> B/d L/s/t", alpha_partial)
    alpha = jnp.sum(alpha_per_token) / jnp.float32(tokens_in_global_batch)

    # Report both components and the hard-target CE, so the loss curve stays
    # interpretable and alpha can be watched directly -- it is the quantity tau
    # ultimately depends on, and it should climb toward ~0.7 (Table 1's acceptance
    # rates) rather than toward 1.
    #
    # These are per-device partials, like `loss` itself, which training_step
    # finishes with a psum. TensorStats.from_tensor applies a pmean, so scale by
    # the device count first to make pmean recover the sum. Exact at any mesh size;
    # a no-op at d=t=s=1.
    n_dev = jax.lax.psum(1, ("d", "t", "s"))
    hard_ce = -jnp.sum(logprobs_at_targets) / jnp.float32(tokens_in_global_batch)
    tensor_stats["loss.distill_ce"] = TensorStats.from_tensor(distill_ce * n_dev)
    tensor_stats["loss.alpha"] = TensorStats.from_tensor(alpha * n_dev)
    tensor_stats["loss.hard_ce"] = TensorStats.from_tensor(hard_ce * n_dev)

    return omega * distill_ce + (1.0 - omega) * (-alpha), tensor_stats


@pytree_dataclass
class Metrics:
    loss: f32[b""]
    learning_rate: f32[b""]
    grad_norm: f32[b""]
    raw_grad_norm: f32[b""]
    tensor_stats: StatsDict


@dataclass(frozen=True)
class OptimizerConfig:
    batch_size: int
    learning_rate: float
    steps: int
    warmup_steps: int
    steps_for_lr: int
    cosine_learning_rate_final_fraction: float
    adam_b1: float
    adam_b2: float
    adam_eps: float
    adam_atan2_a: Optional[float]
    adam_atan2_b: Optional[float]
    adam_eps_root: float
    weight_decay: float
    adam_completed_steps_offset: int
    dataloader_steps_offset: int


@pytree_dataclass
class State:
    weights: Model
    adam_mu: Model
    adam_nu: Model

    @staticmethod
    def init(h: ModelConfig, rng: PRNGKey) -> "State":
        weights = Model.init(h, rng)
        adam_mu = jax.tree.map(lambda p: p * 0.0, weights)
        adam_nu = jax.tree.map(lambda p: p * 0.0, weights)
        return State(weights=weights, adam_mu=adam_mu, adam_nu=adam_nu)

    def reset_optimizer(self):
        adam_mu = jax.tree.map(lambda p: p * 0.0, self.weights)
        adam_nu = jax.tree.map(lambda p: p * 0.0, self.weights)
        return State(weights=self.weights, adam_mu=adam_mu, adam_nu=adam_nu)


@partial(jax.jit, static_argnames=("config", "h_teacher"), donate_argnums=(0,))
def training_step(
    state: State,
    config: "TrainingConfig",
    step: u32[b""],
    batch: TokenBatch,
    teacher: Optional[Model] = None,
    h_teacher: Optional[ModelConfig] = None,
) -> Tuple[State, Metrics]:
    """One optimizer step. `teacher` is frozen weights for SPIRe distillation;
    it is passed as a traced argument (not part of `state`) so no gradient can
    reach it and it is never checkpointed."""

    def _body(state: State, step, batch: TokenBatch, teacher) -> Tuple[State, Metrics]:
        # Create PRNG key from base seed and current step
        rng = jax.random.PRNGKey(config.einsum_seed)
        rng = fold_in_str(rng, f"step_{step}")

        (loss, tensor_stats), grad = jax.value_and_grad(
            lambda weights: loss_fn(
                weights, config.model, batch, rng, teacher, h_teacher, config.distill.omega
            ),
            has_aux=True,
        )(state.weights)

        # if you want to include the FWD / BWD / OPT annotations in the HLO SVG, comment the last three lines
        # and uncomment the following five lines and indent until including new_state = (...)
        # with jax.named_scope("seqax_FWD_xaqes"):
        #     loss, vjpfun, tensor_stats = jax.vjp(lambda weights: loss_fn(weights, h, batch), state.weights, has_aux=True)
        # with jax.named_scope("seqax_BWD_xaqes"):
        #     grad = vjpfun(jnp.float32(1.0))[0]
        # with jax.named_scope("seqax_OPT_xaqes"):

        # Gradients have already been reduced across chips because the gradient of the weight `all_gather`
        # is weight-gradient `psum_scatter`. Loss, on the other hand, hasn't been reduced across chips: if we
        # did that inside the autodiff, we'd be double-reducing the loss, effectively multiplying it by the
        # amount of data parallelism.
        #
        # So we reduce the loss across chips _outside_ the autodiff.
        loss = jax.lax.psum(loss, ("d", "t", "s"))
        # All params are fully sharded across "d", "t", "s", so autograd already distributed their gradients.
        # For other mesh axes, that might use replicated params, we need to average grads across replicas.
        grad_leaves, grad_treedef = jax.tree_util.tree_flatten(grad)
        grad_leaves = [
            shardops.pmean_across_replicas(pspec, g)
            for g, pspec in zip(grad_leaves, tree_leaves(shardtypes.make_partition_specs(State)))
        ]

        # Other than global-norm of gradients, no other communication is needed during the weight update,
        # because weights and grads are already fully sharded, as checked below.

        # Add grads to tensor_stats before clipping.
        tensor_stats.update(
            {f"{key}.grad": TensorStats.from_tensor(t) for key, t in vars(grad).items() if key != "transformer"}
        )
        n_layers = config.model.layers
        for key, t in vars(grad.transformer).items():
            tensor_stats.update({f"{i}.{key}.grad": TensorStats.from_tensor(t[i]) for i in range(n_layers)})

        # Calculate learning rate from step number.
        # We use linear warmup then cosine decay. See https://arxiv.org/pdf/2307.09288.pdf section 2.2
        hopt = config.optimizer
        warmup_lr = (jnp.float32(step) / jnp.float32(hopt.warmup_steps)) * hopt.learning_rate
        cosine = jnp.cos(
            jnp.pi * (jnp.float32(step - hopt.warmup_steps) / jnp.float32(hopt.steps_for_lr - hopt.warmup_steps))
        )
        cosine_lr = hopt.learning_rate * (
            hopt.cosine_learning_rate_final_fraction
            + (1 - hopt.cosine_learning_rate_final_fraction) * (cosine * 0.5 + 0.5)
        )
        lr = jnp.where(step < hopt.warmup_steps, warmup_lr, cosine_lr)

        # AdamW optimizer with global gradient clipping.
        global_norm_square = jnp.float32(0.0)
        for g in grad_leaves:
            assert g.dtype == jnp.float32
            global_norm_square += jnp.sum(jax.lax.square(g))
        global_norm_square = jax.lax.psum(global_norm_square, ("d", "t", "s"))
        global_norm = jnp.sqrt(global_norm_square)
        rescale = jnp.minimum(1.0, 1.0 / global_norm)

        new_ps = []
        new_mus = []
        new_nus = []
        updates = []
        for p, g, mu, nu in zip(
            tree_leaves(state.weights),
            grad_leaves,
            tree_leaves(state.adam_mu),
            tree_leaves(state.adam_nu),
        ):
            # Gradient clipping
            g = g * rescale
            # Adam scaling
            mu = (1 - hopt.adam_b1) * g + hopt.adam_b1 * mu
            nu = (1 - hopt.adam_b2) * jax.lax.square(g) + hopt.adam_b2 * nu
            # We need step numbers to start at 1, not 0. Otherwise the bias correction produces NaN.
            completed_steps = step + 1 + hopt.adam_completed_steps_offset
            mu_hat = mu / (1 - jnp.float32(hopt.adam_b1) ** completed_steps)
            nu_hat = nu / (1 - jnp.float32(hopt.adam_b2) ** completed_steps)
            # Appendix C.5 of https://arxiv.org/pdf/2407.05872
            # small angle approximation of arctan2.
            # b     a, if m << sqrt(v)      a, if m ~ sqrt(v)
            # 1         1                   1/arctan(1)     = 1.27
            # 2         2                   1/arctan(1/2)   = 2.16
            # 4         4                   1/arctan(1/4)   = 4.08
            # 8         8                   1/arctan(1/8)   = 8.04
            # 16        16                  1/arctan(1/16)  = 16.02
            # 32        32                  1/arctan(1/32)  = 32.01
            if hopt.adam_atan2_a:
                g = hopt.adam_atan2_a * jnp.arctan2(mu_hat, hopt.adam_atan2_b * jnp.sqrt(nu_hat))
            else:
                g = mu_hat / (jnp.sqrt(nu_hat + hopt.adam_eps_root) + hopt.adam_eps)
            # Weight decay
            g += hopt.weight_decay * p
            # Learning rate
            g *= lr

            # Apply update
            new_ps.append(p - g)
            new_mus.append(mu)
            new_nus.append(nu)
            updates.append(g)

        new_state = State(
            weights=jax.tree_util.tree_unflatten(grad_treedef, new_ps),
            adam_mu=jax.tree_util.tree_unflatten(grad_treedef, new_mus),
            adam_nu=jax.tree_util.tree_unflatten(grad_treedef, new_nus),
        )
        # Add parameters to tensor_stats after update.
        w = new_state.weights
        tensor_stats.update(
            {f"{key}.weight": TensorStats.from_tensor(t) for key, t in vars(w).items() if key != "transformer"}
        )
        for key, t in vars(w.transformer).items():
            tensor_stats.update({f"{i}.{key}.weight": TensorStats.from_tensor(t[i]) for i in range(n_layers)})
        updates = jax.tree_util.tree_unflatten(grad_treedef, updates)
        tensor_stats.update(
            {f"{key}.update": TensorStats.from_tensor(t) for key, t in vars(updates).items() if key != "transformer"}
        )
        for key, t in vars(updates.transformer).items():
            tensor_stats.update({f"{i}.{key}.update": TensorStats.from_tensor(t[i]) for i in range(n_layers)})

        metrics = Metrics(
            loss=loss,
            learning_rate=lr,
            grad_norm=global_norm * rescale,
            raw_grad_norm=global_norm,
            tensor_stats=tensor_stats if config.log_tensor_stats else {},
        )
        return new_state, metrics

    # The teacher is CLOSED OVER rather than passed as a parameter. Two shardlib
    # constraints force this:
    #   * an `Optional[Model]` annotation reaches typed_shard_map's issubclass()
    #     check as a Union and raises;
    #   * a concrete `Model` annotation type-checks the teacher against dimension
    #     names already bound by the student, and `layers` differs (8 vs 2).
    # Closing over sidesteps both, at the cost of the teacher being replicated
    # rather than sharded inside shard_map. That is exactly correct on a 1x1x1
    # mesh and wrong on anything larger, so main() asserts the mesh is single-device
    # whenever distillation is enabled.
    @partial(shardtypes.typed_shard_map, check_rep=False)
    def sharded_step(state: State, step: u32[b""], batch: TokenBatch) -> Tuple[State, Metrics]:
        return _body(state, step, batch, teacher)

    return sharded_step(state, step, batch)


@dataclass(frozen=True)
class DistillConfig:
    """SPIRe MixedLoss. `enabled: false` (the base.yaml default) leaves training as
    plain hard-target cross-entropy, so every pre-existing config is unaffected."""

    enabled: bool
    teacher_config: str  # config name under configs/, for the teacher's ModelConfig
    teacher_name: str  # run dir under root_working_dir holding the teacher checkpoint
    omega: float  # w in w*distill_CE + (1-w)*(-alpha); paper uses 0.5


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str
    root_working_dir: str
    clone_from: str
    clone_checkpoint_index: int
    reset_optimizer: bool
    log_tensor_stats: bool
    model: ModelConfig
    optimizer: OptimizerConfig
    distill: DistillConfig
    dataset: input_loader.FlatTokensParams
    weight_init_seed: int
    einsum_seed: int
    mesh: MeshConfig
    use_firestore: bool
    firestore_gcp_project: str
    zarr_log_flush_frequency: int
    checkpoint_interval: int
    io: training_io.IOConfig
    max_retries: int


def train_attempt(config: DictConfig):
    # We want to be preemptible till we start the training loop.
    def early_exit(*args):
        print("Preempting before train loop")
        sys.exit(0)

    signal.signal(signal.SIGINT, early_exit)
    using_mlq = "MLQ_RANK" in os.environ and "MLQ_NUM_NODES" in os.environ and "MLQ_NODE_0" in os.environ
    if using_mlq:
        jax.distributed.initialize(
            os.environ["MLQ_NODE_0"] + ":8999",
            num_processes=int(os.environ["MLQ_NUM_NODES"]),
            process_id=int(os.environ["MLQ_RANK"]),
        )

    config: TrainingConfig = jax_extra.make_dataclass_from_dict(TrainingConfig, config)
    batch_params = input_loader.TokenBatchParams(
        len=config.model.seq_len,
        batch=config.optimizer.batch_size,
    )
    num_batch_tokens = batch_params.batch * batch_params.len

    with Mesh(
        mesh_utils.create_device_mesh([config.mesh.d, config.mesh.t, config.mesh.s], jax.devices()), ("d", "t", "s")
    ):
        if config.dataset.eval_tokens > 0:
            max_val_steps = ShufflingLoader("validation", config.dataset, batch_params).step_count
            suffix = (
                f" (possible values for eval_tokens: 0, {num_batch_tokens}, ..., {max_val_steps * num_batch_tokens})."
            )
            if config.dataset.eval_tokens % num_batch_tokens != 0:
                raise ValueError(f"Batch size {num_batch_tokens} has to divide {config.dataset.eval_tokens=}{suffix}")
            if max_val_steps * num_batch_tokens < config.dataset.eval_tokens:
                raise ValueError(f"Validation set is too small, need {config.dataset.eval_tokens=} tokens{suffix}")

        weight_rng = jax.random.PRNGKey(config.weight_init_seed)

        loader = ShufflingLoader("train", config.dataset, batch_params)

        assert config.model.vocab > loader.max_token_id, f"{config.model.vocab} vs {loader.max_token_id}"

        model_dir = os.path.join(config.root_working_dir, config.model_name)
        clone_dir = os.path.join(config.root_working_dir, config.clone_from)
        training_io.mkdir(model_dir)
        state = jax.jit(partial(State.init, config.model))(fold_in_str(weight_rng, "init"))
        state, start_step = training_io.load_checkpoint_if_it_exists(model_dir, state, config.io)
        resuming = True if start_step > 0 else False
        clone_checkpoint_index = config.clone_checkpoint_index
        if config.clone_from and not resuming:
            assert clone_checkpoint_index >= 0, "clone_checkpoint_index must be non-negative"
            state, start_step = training_io.load_checkpoint_if_it_exists(
                clone_dir, state, config.io, clone_checkpoint_index
            )
        state = state.reset_optimizer() if config.reset_optimizer and not resuming else state
        cloning = config.clone_from and resuming
        try:
            zarr_log = training_io.ZarrLog(os.path.join(model_dir, "log.zarr"), config, cloning)
            # TODO: Streamline cloning for multi-stage training, fine-tuning, and validation.
            if jax.process_index() == 0 and config.clone_from and not resuming:
                zarr_log.last_valid_index = min(config.optimizer.steps, clone_checkpoint_index)
        except RuntimeError:
            traceback.print_exc()
            sys.exit(os.EX_SOFTWARE)
        firestore_log = training_io.FirestoreLog(
            project_name=config.model_name,
            config=config,
            create=not resuming,
            use_firestore=config.use_firestore,
            gcp_project=config.firestore_gcp_project,
        )

        if jax.process_index() == 0:
            attributes = {
                "commit_hash": training_io.get_git_commit_hash(),
                "command": " ".join(sys.argv[1:]),
            }
            training_io.log_attributes(zarr_log, firestore_log, attributes)

        # ---- SPIRe distillation: load the frozen teacher once, before compiling ----
        teacher, h_teacher = None, None
        if config.distill.enabled:
            # training_step closes over the teacher instead of passing it through
            # shard_map (see the comment there), which replicates rather than shards
            # it. Correct on one device, silently wrong on more.
            assert config.mesh.d == config.mesh.t == config.mesh.s == 1, (
                "distillation currently requires a 1x1x1 mesh; the teacher is closed "
                "over rather than sharded. Plumb it through shard_map before scaling out."
            )
            teacher_cfg = OmegaConf.load(f"configs/{config.distill.teacher_config}.yaml")
            h_teacher = ModelConfig(**teacher_cfg.model)
            teacher_dir = os.path.join(config.root_working_dir, config.distill.teacher_name)
            # Fresh Scope: the teacher's `layers` differs from the student's, and
            # shardlib's dimension names are global.
            with shardtypes.Scope():
                t_state = jax.jit(partial(State.init, h_teacher))(fold_in_str(weight_rng, "teacher_init"))
                t_state, t_step = training_io.load_checkpoint_if_it_exists(teacher_dir, t_state, config.io)
            if t_step == 0:
                raise FileNotFoundError(f"distillation enabled but no teacher checkpoint in {teacher_dir}")
            # Weights only. Dropping the optimizer state keeps ~950MB off the device
            # and makes it structurally impossible to update the teacher.
            teacher = jax.tree.map(lambda x: x, t_state.weights)
            del t_state
            print(
                f"distilling from {config.distill.teacher_name} step {t_step} "
                f"({h_teacher.layers} layers), omega={config.distill.omega}"
            )

        # Explicitly compile training step, to record XLA HLO graph.
        # See https://bnikolic.co.uk/blog/python/jax/2022/02/22/jax-outputgraph-rev
        c_training_step = training_step.lower(
            state, config, jnp.uint32(0), loader.load(0), teacher, h_teacher
        ).compile()
        date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        training_io.save_hlo_dot(os.path.join(model_dir, f"training_step_optimized_hlo_{date}.dot"), c_training_step)
        multihost_utils.sync_global_devices("train_attempt_init")

        interrupting = False

        def signal_handler(sig, frame):
            nonlocal interrupting
            interrupting = True

        signal.signal(signal.SIGINT, signal_handler)

        for step in range(start_step, config.optimizer.steps):
            # We profile on the second step, because the first step has a long pause for XLA
            # compilation and initial shuffle buffer loading.
            if jax.process_index() == 0 and step == 1:
                jax.block_until_ready(state)
                training_io.start_profile()
                profile_start = time.time()

            current_run_step = step - clone_checkpoint_index
            state, output = c_training_step(
                state,
                jnp.uint32(current_run_step),
                loader.load(current_run_step + config.optimizer.dataloader_steps_offset),
                teacher,
            )

            # Run profile for two steps, to include data loading time in between them.
            if jax.process_index() == 0 and step == 2:
                jax.block_until_ready(state)
                profile_duration = time.time() - profile_start
                training_io.stop_profile(model_dir)

                # Print MFU, including (one step of) data loading time.
                print(f"Profile time: {profile_duration}s for 2 steps.")
                model_params = jax.tree.reduce(operator.add, jax.tree.map(lambda w: w.size, state.weights))
                device_flops = training_io.get_flops_per_device()
                num_devices = jax.device_count()
                # remove embed (which does not perform a matrix multiplication) and include q*k and probs*v
                flops_per_inference_token = 2 * (
                    model_params
                    - state.weights.embed.size
                    + 2 * state.weights.transformer.w_q.size * config.model.seq_len / config.model.d_model
                )
                MFU = 100 * 2 * 3 * flops_per_inference_token * num_batch_tokens
                MFU /= num_devices * profile_duration * device_flops
                MFU = round(MFU, 2)
                kv_fetched_numbers_per_decode_token = (
                    config.model.d_head * config.model.n_kv * 2 * config.model.layers * config.model.seq_len
                )
                attributes = {
                    "model_params": model_params,
                    "tokens": num_batch_tokens,
                    "MFU": MFU,
                    "flops_per_inference_token": flops_per_inference_token,
                    "kv_fetched_numbers_per_decode_token": kv_fetched_numbers_per_decode_token,
                }
                training_io.log_attributes(zarr_log, firestore_log, attributes)
                zarr_log.flush()
            training_io.log_training_step(zarr_log, firestore_log, step, output)
            loss = float(output.loss)
            if math.isnan(loss) or loss > 100.0:
                raise ValueError(f"Aborting early due to likely divergence: loss={loss}")

            last_step = step == config.optimizer.steps - 1
            if (step + 1) % config.checkpoint_interval == 0 or last_step:
                training_io.save_checkpoint(model_dir, step + 1, state, config.io)

            if interrupting and step > 2 and not last_step:
                # We only pre-empt after the second step, so that the profiling / MFU calculation can
                # complete before we preempt.
                print(f"Preempting at step {step + 1}")
                if (step + 1) % config.checkpoint_interval != 0:
                    training_io.save_checkpoint(model_dir, step + 1, state, config.io)
                zarr_log.flush(wait_for_previous_flush=True, wait_for_this_flush=True)
                return  # skip the eval steps and the finalization of log.zarr

        if config.dataset.eval_tokens > 0:
            del c_training_step
            del loader
            loader = ShufflingLoader("validation", config.dataset, batch_params)
            eval_steps = config.dataset.eval_tokens // num_batch_tokens

            @jax.jit
            def eval_loss_fn(weights: Model, batch: TokenBatch) -> f32[b""]:
                @partial(shardtypes.typed_shard_map, check_rep=False)
                def sharded_eval_loss_fn(weights: Model, batch: TokenBatch) -> f32[b""]:
                    loss, _ = loss_fn(weights, config.model, batch)
                    loss = jax.lax.psum(loss, ("d", "t", "s"))
                    return loss

                return sharded_eval_loss_fn(weights, batch)

            eval_losses = np.full(eval_steps, np.nan, dtype=np.float32)
            for step in range(eval_steps):
                loss = eval_loss_fn(state.weights, loader.load(step))
                eval_losses[step] = loss
                if step % 100 == 0:
                    print(f"Eval step {step}: {loss}")
            zarr_log.log_eval_losses(eval_losses)
            attributes = {
                "eval_loss": float(np.mean(eval_losses)),
                "eval_loss_std": float(np.std(eval_losses)),
                "eval_steps": eval_steps,
            }
            training_io.log_attributes(zarr_log, firestore_log, attributes)

        training_io.log_finalize(zarr_log, firestore_log)


@hydra.main(config_path="configs", version_base=None)
def main(config: DictConfig):
    assert config.mesh.t == config.mesh.s == 1, "Tensor and sequence parallelism are not supported"
    assert config.dataset.sequence_packing, "Training without sequence packing is not supported"
    assert config.max_retries >= 0, "max_retries must be non-negative"
    if config.max_retries == 0:
        train_attempt(config)
        return
    max_attempts = 1 + config.max_retries
    for attempt in range(1, max_attempts + 1):
        print(f"Starting attempt {attempt}/{max_attempts}")
        p = multiprocessing.Process(target=train_attempt, args=(config,))
        p.start()
        # Parent process should ignore SIGINT, and just wait for the child process to finish.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        p.join()

        if p.exitcode == 0:
            print("Training completed successfully")
            sys.exit(0)
        else:
            print(f"Attempt {attempt} failed with exit code {p.exitcode}")

    # If we've reached this point, we've failed all retries.
    sys.exit(1)


if __name__ == "__main__":
    main()

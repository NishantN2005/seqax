import inspect
import re

import jax
import jax.numpy as jnp
from jax import lax

import shardlib.shardtypes as shardtypes


def named_scope(func):
    """
    Wraps a function into a jax.named_scope context manager that includes the caller's source code to display in the HLO SVG.
    """

    def wrapper(*args, **kwargs):
        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back
        caller_info = inspect.getframeinfo(caller_frame)
        file_name = caller_info.filename
        line_number = caller_info.lineno - 1
        with open(file_name, "r") as f:
            lines = f.readlines()
        for start_line in range(line_number, -1, -1):
            if "=" in lines[start_line]:
                break
        else:
            start_line = line_number
        for end_line in range(line_number, len(lines)):
            if ")" in lines[end_line]:
                break
        else:
            end_line = line_number
        caller_code = lines[start_line : end_line + 1]
        caller_code = " ".join(caller_code)
        caller_code = caller_code.replace("\n", " ")
        caller_code = caller_code.strip()
        caller_code = re.sub(r"  +", " ", caller_code)
        scope_name = f"seqax_<{caller_code}>_xaqes"
        with jax.named_scope(scope_name):
            return func(*args, **kwargs)

    return wrapper


@named_scope
def all_gather(spec: str, x):
    """String-specified all-gather operation.

    For example:
      all_gather('A/x/y B/z C/w -> A B C/w', x)
    """
    before, after = spec.split("->")
    before = shardtypes.ShapeSpec.parse(before)
    after = shardtypes.ShapeSpec.parse(after)
    shardtypes.check(x.dtype, before, x)
    for i, (before_dim, after_dim) in enumerate(zip(before.dims, after.dims)):
        # Check that after_dim.sharding is a prefix of before_dim.sharding
        after_n = len(after_dim.sharding)
        if before_dim.shape != after_dim.shape or before_dim.sharding[:after_n] != after_dim.sharding:
            raise ValueError(f"Cannot all-gather {before_dim} into {after_dim}")
        if len(before_dim.sharding) == after_n:
            continue
        x = lax.all_gather(x, tuple(before_dim.sharding[after_n:]), axis=i, tiled=True)
    shardtypes.check(x.dtype, after, x)
    return x


@named_scope
def psum_scatter(spec: str, x):
    """String-specified reduce-scatter operation.

    For example:
      psum_scatter('A B C/w -> A/x/y B/z C/w', x)
    """
    before, after = spec.split("->")
    before = shardtypes.ShapeSpec.parse(before)
    after = shardtypes.ShapeSpec.parse(after)
    shardtypes.check(x.dtype, before, x)
    for i, (before_dim, after_dim) in enumerate(zip(before.dims, after.dims)):
        # Check that before_dim.sharding is a prefix of after_dim.sharding
        before_n = len(before_dim.sharding)
        if before_dim.shape != after_dim.shape or after_dim.sharding[:before_n] != before_dim.sharding:
            raise ValueError(f"Cannot reduce-scatter {before_dim} into {after_dim}")
        if len(after_dim.sharding) == before_n:
            continue
        x = lax.psum_scatter(x, tuple(after_dim.sharding[before_n:]), scatter_dimension=i, tiled=True)
    shardtypes.check(x.dtype, after, x)
    return x


@named_scope
def einsum_unreduced(spec: str, x, y, **kwargs):
    """Ordinary chip-local einsum, but with sharding-aware typechecking.

    Note that this function does not do any chip-to-chip communication. If the inputs are
    sharded over the contraction dimensions, the caller is responsible for reducing the result
    over those dimensions. For example:

      c = einsum_unreduced('A/x B/y, B/y C/z -> A/x/z', a, b)
      # c still needs to be reduced over the y axis.
      d = psum_scatter('A/x/z -> A/x/z/y', c)
      # Now the post-einsum reduction is complete.
    """
    tmp, result = spec.split("->")
    lhs, rhs = tmp.split(",")
    lhs = shardtypes.ShapeSpec.parse(lhs)
    rhs = shardtypes.ShapeSpec.parse(rhs)
    result = shardtypes.ShapeSpec.parse(result)
    shardtypes.check(x.dtype, lhs, x)
    shardtypes.check(y.dtype, rhs, y)
    # Convert to jax einsum syntax, with single-letter variables.
    jaxspec = ""

    vars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    var_i = 0
    dim_table = {}

    def map_var(dim):
        if dim in dim_table:
            return dim_table[dim]
        nonlocal var_i
        if var_i >= len(vars):
            raise ValueError("Too many dimensions in einsum, we ran out of variables")
        var = vars[var_i]
        var_i += 1
        dim_table[dim] = var
        return var

    for dim in lhs.dims:
        jaxspec += map_var(dim)
    jaxspec += ","
    for dim in rhs.dims:
        jaxspec += map_var(dim)
    jaxspec += "->"
    for dim in result.dims:
        jaxspec += map_var(dim)
    r = jnp.einsum(jaxspec, x, y, **kwargs)
    shardtypes.check(r.dtype, result, r)
    return r


@named_scope
def pmean_across_replicas(pspec: jax.sharding.PartitionSpec, x):
    """Computes pmean across all replicated (non-sharded) dimensions of the tensor"""
    sharded_axes = set()
    for axis in pspec:
        if axis is None:
            continue
        elif isinstance(axis, str):
            sharded_axes.add(axis)
        elif isinstance(axis, tuple):
            for a in axis:
                if isinstance(a, str):
                    sharded_axes.add(a)
        else:
            raise ValueError(f"Unknown axis type {axis}")

    pmean_axes = []
    for axis in jax._src.core.get_axis_env().axis_sizes:
        if axis not in sharded_axes:
            pmean_axes.append(axis)

    if pmean_axes:
        return jax.lax.pmean(x, tuple(pmean_axes))
    else:
        return x


@named_scope
def index_unreduced(spec: str, table, indices, use_onehot=False):
    """String-specified sharded table lookup operation.

    For example:
      index_unreduced('A [B/x/y] C/z, D/w A -> C/z A D/w', table, indices)

    In this example, the integers in `indices` are used as lookup addresses into the
    `B` dimension of `table`, and all other dimensions (`A`, `C`, `D`) are vmapped over.

    This operation does not do any chip-to-chip communication, even though the table
    may be sharded. If the axis inside square brackets is sharded, corresponding to
    different table indices on different shards, a table lookup will be performed on each
    shard, but only one shard will return a nonzero result: the other shards, where the
    index is out of bounds, will return zero. The caller is required to reduce the output
    over the axes specified by the square brackets: in the above example, the caller must
    reduce over `x` and `y` axes.

    use_onehot: If True, onehot multiply is performed instead of embeddings lookups. This is faster on TPU.
    """
    tmp, result = spec.split("->")
    lhs, rhs = tmp.split(",")
    lhs_split = lhs.split()
    index_axis = None
    for i, dim in enumerate(lhs_split):
        if dim.startswith("["):
            index_axis = i
            if not dim.endswith("]"):
                raise ValueError(f"Expected closing bracket in {dim}")
            lhs_split[i] = dim[1:-1]
            break
    if index_axis is None:
        raise ValueError(f"Expected an index axis in {lhs}")

    lhs_dims = [shardtypes.DimSpec.parse(dim) for dim in lhs_split]
    lhs_spec = shardtypes.ShapeSpec(lhs_dims)
    rhs_spec = shardtypes.ShapeSpec.parse(rhs)
    result_spec = shardtypes.ShapeSpec.parse(result)
    shardtypes.check(table.dtype, lhs_spec, table)
    shardtypes.check(indices.dtype, rhs_spec, indices)

    len_per_chip = table.shape[index_axis]
    lower_bound = len_per_chip * lax.axis_index(lhs_dims[index_axis].sharding)
    upper_bound = lower_bound + len_per_chip

    if use_onehot:
        onehot_lhs = rhs + lhs_split[index_axis]
        onehot_rhs = " ".join(lhs_split)
        return einsum_unreduced(
            f"{onehot_lhs}, {onehot_rhs} -> {result}",
            jax.nn.one_hot(indices - lower_bound, len_per_chip, dtype=table.dtype),
            table,
        )

    # Do the base operation on scalars, then do a sequence of vmap operations to bring it up
    # to the desired shape.
    def base_op(table, index):
        in_bounds = (lower_bound <= index) & (index < upper_bound)
        return jnp.where(in_bounds, table[jnp.where(in_bounds, index - lower_bound, 0)], 0)

    op = base_op

    lhs_dims_handled = [False] * len(lhs_dims)
    lhs_dims_handled[index_axis] = True
    rhs_dims_handled = [False] * len(rhs_spec.dims)
    for dim in reversed(result_spec.dims):
        try:
            lhs_index = lhs_dims.index(dim)
            lhs_vmap_axis = sum(lhs_dims_handled[:lhs_index])
            assert not lhs_dims_handled[lhs_index]
            lhs_dims_handled[lhs_index] = True
        except ValueError:
            lhs_index = None
            lhs_vmap_axis = None

        try:
            rhs_index = rhs_spec.dims.index(dim)
            rhs_vmap_axis = sum(rhs_dims_handled[:rhs_index])
            assert not rhs_dims_handled[rhs_index]
            rhs_dims_handled[rhs_index] = True
        except ValueError:
            rhs_index = None
            rhs_vmap_axis = None

        op = jax.vmap(op, in_axes=(lhs_vmap_axis, rhs_vmap_axis), out_axes=0)

    assert all(lhs_dims_handled)
    assert all(rhs_dims_handled)

    result = op(table, indices)
    shardtypes.check(result.dtype, result_spec, result)
    return result


@named_scope
def axis_size(name: str) -> int:
    """Return the size of the axis with the given name."""
    return jax.lax.psum(1, name)


@named_scope
def sharded_arange(n: int, mesh_axis: str) -> jnp.array:
    assert n % axis_size(mesh_axis) == 0, f"n={n} is not divisible by {mesh_axis} size {axis_size(mesh_axis)}"
    shard_size = n // axis_size(mesh_axis)
    return jax.lax.axis_index(mesh_axis) * shard_size + jnp.arange(shard_size)


@named_scope
def shard(x: jnp.array, spec: str) -> jnp.array:
    """
    Shard tensor's axis along a given given mesh axis.

    Note, this function requires the input tensor to be replicated, so it can be worth
    considering if you can create the shards independently on each device.
    """
    before, after = spec.split("->")
    before = shardtypes.ShapeSpec.parse(before)
    after = shardtypes.ShapeSpec.parse(after)

    sharded = set()
    starts = []
    lens = []
    for i, (before_dim, after_dim) in enumerate(zip(before.dims, after.dims)):
        if sharded.intersection(after_dim.sharding):
            raise ValueError(f"Cannot shard {before_dim} into {after_dim}. Mesh axis is already sharded")
        sharded.update(after_dim.sharding)
        if before_dim.shape != after_dim.shape:
            raise ValueError(
                f"Cannot shard {before_dim} into {after_dim}. The ordering of dimensions should not change."
            )

        if len(before_dim.sharding) + 1 < len(after_dim.sharding):
            raise ValueError(
                f"Cannot shard {before_dim} into {after_dim}. We can only shard along one mesh axis per dimension"
            )

        if before_dim.sharding != after_dim.sharding[: len(before_dim.sharding)]:
            raise ValueError(f"Cannot shard {before_dim} into {after_dim}. New sharding can only be added")

        if before_dim.sharding == after_dim.sharding:
            starts.append(0)
            lens.append(x.shape[i])
        else:
            new_sharding = after_dim.sharding[-1]
            shard_size = x.shape[i] // axis_size(new_sharding)
            starts.append(jax.lax.axis_index(new_sharding) * shard_size)
            lens.append(shard_size)
    return jax.lax.dynamic_slice(x, starts, lens)


@named_scope
def ring_permute_sharded(x, axis_index: int, mesh_axis: str):
    """Ring permute `x` by one index in the `axis_index` dimension which is sharded along `mesh_axis`.

    The elements crossing from last to first are zeroed out, for example:
    {dev0:[1, 2, 3], dev1: [4, 5, 6]} -> {dev0:[0, 1, 2], dev1: [3, 4, 5]}.
    """
    sl = [slice(None)] * x.ndim
    sl[axis_index] = slice(-1, None)
    x_last = x[tuple(sl)]
    n_s = axis_size(mesh_axis)
    permutation = [(i, (i + 1) % n_s) for i in range(n_s)]
    x_last = lax.ppermute(x_last, axis_name=mesh_axis, perm=permutation)
    # Zero out the elements passed from last to first
    x_last = jnp.where(lax.axis_index(mesh_axis) == 0, jnp.zeros_like(x_last), x_last)
    sl[axis_index] = slice(None, -1)
    x_shifted = x[tuple(sl)]
    x = jnp.concatenate([x_last, x_shifted], axis=axis_index)
    return x

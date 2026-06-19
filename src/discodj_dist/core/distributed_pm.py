"""Distributed (multi-GPU) Particle-Mesh primitives for DISCO-DJ.

This module is *additive*: nothing in this file modifies DISCO-DJ's existing
single-GPU code. It provides drop-in distributed analogues of the density
assignment and force interpolation steps used by ``calc_acc_PM_``, plus a few
small helpers that the new ``calc_acc_PM_distributed`` function (in
``discodj/nbody/acc_distributed.py``) depends on.

Design summary
--------------
* Local physics kernels are pure JAX functions that operate on a *padded*
  shard. They are wrapped with ``shard_map`` so each rank executes them on its
  own subdomain.
* Inter-rank communication is delegated entirely to ``jaxdecomp.halo_exchange``
  (which has a registered VJP, so the chain remains differentiable).
* Two distinct halo patterns are implemented (see JaxPM CLAUDE_SUMMARY §3):
    - ``scatter_dx_distributed``: pre-pad ``H``, exchange ``H//2``, fold + trim
      via ``slice_unpad``.
    - ``gather_dx_distributed``: pre-pad ``H//2``, exchange ``H//2``, no fold.
* The Lagrangian-painting trick (JaxPM ``cic_paint_dx``) is used so that the
  legacy ``np.mod(neighbor_coords, res)`` global wrap is no longer needed --
  particles are identified by integer LOCAL Lagrangian index plus a small
  displacement, and out-of-bound deposits are silently dropped via the
  existing ``mode="drop"`` of ``jax.lax.scatter_add``.
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

import jaxdecomp

# Reuse DISCO-DJ's existing CIC/TSC/PCS kernel functions and stencil generator.
# We do NOT modify them.
from .scatter_and_gather import kernel_functions, get_stencil
from .kernels import gradient_kernel as _disco_gradient_kernel


__all__ = [
    "get_local_pdims",
    "get_mesh_axis_names",
    "get_local_q_grid",
    "slice_pad",
    "slice_unpad",
    "scatter_dx_distributed",
    "gather_dx_distributed",
    "gradient_kernel_dist",
]


# ---------------------------------------------------------------------------
# Mesh / sharding utilities
# ---------------------------------------------------------------------------

def get_local_pdims(sharding: NamedSharding) -> Tuple[int, int]:
    """Return ``(px, py)`` -- the number of shards along the X and Y mesh
    axes. Z is never sharded.
    """
    devices = sharding.mesh.devices
    if devices.ndim != 2:
        raise ValueError(
            f"Expected a 2-D device mesh, got shape {devices.shape}. "
            "Use jax.make_mesh(pdims, ('x', 'y')) with a 2-D pdims tuple."
        )
    return int(devices.shape[0]), int(devices.shape[1])


def get_mesh_axis_names(sharding: NamedSharding):
    """Return the first two mesh axis names used for X/Y domain sharding."""
    axis_names = tuple(sharding.mesh.axis_names)
    if len(axis_names) < 2:
        raise ValueError(
            f"Expected at least two mesh axis names, got {axis_names}. "
            "Distributed PM needs a 2-D mesh for X/Y sharding."
        )
    return axis_names[0], axis_names[1]


def get_local_q_grid(res_pm: int, boxsize: float, sharding: NamedSharding,
                     dtype=jnp.float32):
    """Produce the LOCAL Lagrangian grid ``q`` (in Mpc/h) per rank.

    Returns an array of shape ``(Lx, Ly, Lz, 3)`` sharded ``P('x','y',None,None)``
    where each rank holds the slab ``[rx*Lx, (rx+1)*Lx) x [ry*Ly, (ry+1)*Ly) x [0, Lz)``
    of the global Lagrangian grid.
    """
    px, py = get_local_pdims(sharding)
    Lx = res_pm // px
    Ly = res_pm // py
    Lz = res_pm
    cell_size = boxsize / res_pm
    axis_x, axis_y = get_mesh_axis_names(sharding)

    spec = P(axis_x, axis_y, None, None)

    @partial(shard_map, mesh=sharding.mesh, in_specs=(), out_specs=spec,
             check_rep=False)
    def _q():
        rx = lax.axis_index(axis_x)
        ry = lax.axis_index(axis_y)
        qx = (jnp.arange(Lx, dtype=dtype) + rx * Lx) * cell_size
        qy = (jnp.arange(Ly, dtype=dtype) + ry * Ly) * cell_size
        qz = jnp.arange(Lz, dtype=dtype) * cell_size
        return jnp.stack(jnp.meshgrid(qx, qy, qz, indexing='ij'), axis=-1)

    return _q()


# ---------------------------------------------------------------------------
# slice_pad / slice_unpad  (JaxPM-style fold-and-trim)
# ---------------------------------------------------------------------------

def slice_pad(x, pad_width, sharding: NamedSharding):
    """Pad each rank's local array with zeros (no inter-rank communication).

    ``pad_width`` is the global pad spec ``((Hx,Hx),(Hy,Hy),(0,0))``; each rank
    receives the same per-rank padding.
    """
    spec = sharding.spec
    in_spec = spec
    out_spec = spec  # padded array shares the same sharding pattern

    @partial(shard_map, mesh=sharding.mesh, in_specs=(in_spec,),
             out_specs=out_spec, check_rep=False)
    def _pad(y):
        return jnp.pad(y, pad_width, mode='constant')

    return _pad(x)


def _slice_unpad_local(x, hx: int, hy: int):
    """Per-rank fold-and-trim. Adds the OUTER ``H/2`` ghost cells (now holding
    the contributions from the neighbor) into the first/last ``H/2`` interior
    cells, then trims all ghost cells off both axis-0 and axis-1.
    """
    if hx > 0:
        hx_half = hx // 2
        # Left ghost outer half -> first hx_half interior cells
        x = x.at[hx:hx + hx_half].add(x[:hx_half])
        # Right ghost outer half -> last hx_half interior cells
        x = x.at[-(hx + hx_half):-hx].add(x[-hx_half:])
    if hy > 0:
        hy_half = hy // 2
        x = x.at[:, hy:hy + hy_half].add(x[:, :hy_half])
        x = x.at[:, -(hy + hy_half):-hy].add(x[:, -hy_half:])

    sx = slice(hx, -hx) if hx > 0 else slice(None)
    sy = slice(hy, -hy) if hy > 0 else slice(None)
    return x[sx, sy, :]


def slice_unpad(x, pad_width, sharding: NamedSharding):
    """Distributed wrapper for the per-rank fold-and-trim."""
    hx = int(pad_width[0][0])
    hy = int(pad_width[1][0])
    # Truncate spec to the actual rank of x (x may be rank-3 density while
    # sharding carries a rank-4 displacement spec like P('x','y',None,None)).
    spec = P(*sharding.spec[:x.ndim])

    @partial(shard_map, mesh=sharding.mesh, in_specs=(spec,), out_specs=spec,
             check_rep=False)
    def _unpad(y):
        return _slice_unpad_local(y, hx, hy)

    return _unpad(x)


# ---------------------------------------------------------------------------
# Local CIC/TSC/PCS kernels on a padded mesh
# ---------------------------------------------------------------------------

def _local_scatter_padded(disp_grid, halo: Tuple[int, int], worder: int,
                          dtype, factor: int = 1):
    """Local scatter onto a padded PM mesh, no inter-rank communication.

    The displacement field is sized to the PARTICLE grid; the painted mesh
    is sized to the PM grid (= particle grid times ``factor`` per axis).
    ``factor = 1`` recovers the legacy "one particle per PM cell" mode.

    Parameters
    ----------
    disp_grid : (Lx_p, Ly_p, Lz_p, 3) displacement in PM-GRID UNITS
                (= Psi / cell_size_pm). Lx_p,Ly_p,Lz_p are particles per shard.
    halo      : (Hx, Hy) -- ghost-zone depth on the X and Y axes, IN PM CELLS.
    worder    : 2 (CIC), 3 (TSC), or 4 (PCS).
    factor    : ``res_pm // res_part`` -- PM oversampling factor.

    Returns
    -------
    mesh_padded : (Lx_p*factor + 2*Hx, Ly_p*factor + 2*Hy, Lz_p*factor)
                  raw mass count (NOT delta).
    """
    Lx_p, Ly_p, Lz_p, _ = disp_grid.shape
    Lx_m = Lx_p * factor
    Ly_m = Ly_p * factor
    Lz_m = Lz_p * factor
    Hx, Hy = halo

    # Particle position in LOCAL PADDED PM-grid coordinates:
    #   pos = (i*factor + Hx + disp_x, j*factor + Hy + disp_y, k*factor + disp_z)
    # The Lagrangian PM-cell index of particle (i,j,k) within the shard is
    # i*factor (and similarly for j, k); the +Hx/+Hy offset places it in the
    # padded mesh interior.
    pmid_x = (jnp.arange(Lx_p, dtype=disp_grid.dtype) * factor) + Hx
    pmid_y = (jnp.arange(Ly_p, dtype=disp_grid.dtype) * factor) + Hy
    pmid_z = jnp.arange(Lz_p, dtype=disp_grid.dtype) * factor
    pos_x = pmid_x[:, None, None] + disp_grid[..., 0]
    pos_y = pmid_y[None, :, None] + disp_grid[..., 1]
    pos_z = pmid_z[None, None, :] + disp_grid[..., 2]
    pos = jnp.stack([pos_x, pos_y, pos_z], axis=-1).reshape(-1, 3)
    n_local = pos.shape[0]

    kernel_func, _ = kernel_functions[worder]
    stencil_arr, n_ngbs = get_stencil(3, dtype=dtype, worder=worder)
    int_dtype = jnp.int32
    stencil_arr = jnp.asarray(stencil_arr, dtype=int_dtype)

    pos_e = pos[:, None, :]  # (N, 1, 3)
    if worder % 2 == 0:
        int_coord = jnp.floor(pos_e).astype(int_dtype)
    else:
        int_coord = jnp.round(pos_e).astype(int_dtype)
    neighbor = int_coord + stencil_arr  # (N, n_ngbs, 3)
    abs_diff = jnp.abs(pos_e - neighbor.astype(pos_e.dtype))
    weights = jnp.prod(kernel_func(abs_diff), axis=-1)  # (N, n_ngbs)

    # Wrap each axis when it is fully local (no halo padding): then the
    # globally-periodic boundary is internal to this shard and must be
    # wrapped with mod. When an axis IS sharded (Hx>0 / Hy>0), we rely on
    # the padded mesh + halo_exchange and DROP out-of-bounds deposits
    # (the JaxPM Lagrangian trick).
    nx = jnp.mod(neighbor[..., 0], Lx_m + 2 * Hx) if Hx == 0 else neighbor[..., 0]
    ny = jnp.mod(neighbor[..., 1], Ly_m + 2 * Hy) if Hy == 0 else neighbor[..., 1]
    nz = jnp.mod(neighbor[..., 2], Lz_m)
    neighbor = jnp.stack([nx, ny, nz], axis=-1)

    mesh = jnp.zeros((Lx_m + 2 * Hx, Ly_m + 2 * Hy, Lz_m), dtype=dtype)
    dnums = lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0, 1, 2),
        scatter_dims_to_operand_dims=(0, 1, 2),
    )
    mesh = lax.scatter_add(
        mesh,
        neighbor.reshape(n_local, n_ngbs, 3),
        weights.reshape(n_local, n_ngbs),
        dnums,
        mode="drop",
    )
    return mesh


def _local_gather_padded(mesh_padded, disp_grid, halo: Tuple[int, int],
                         worder: int, factor: int = 1):
    """Local gather from a (already halo-filled) padded PM mesh.

    The padded mesh is sized to the PM grid; the output (one value per
    particle) is sized to the PARTICLE grid. ``factor = 1`` recovers the
    legacy "one particle per PM cell" mode.

    Parameters
    ----------
    mesh_padded : (Lx_p*factor + 2*Hx, Ly_p*factor + 2*Hy, Lz_p*factor)
    disp_grid   : (Lx_p, Ly_p, Lz_p, 3) displacement in PM-GRID UNITS.
    halo        : (Hx, Hy) -- the per-side padding actually applied (note: for
                  the gather pattern this is ``halo_size // 2``, not ``halo_size``),
                  IN PM CELLS.
    factor      : ``res_pm // res_part`` -- PM oversampling factor.

    Returns
    -------
    field_at_particles : (Lx_p, Ly_p, Lz_p)
    """
    Lx_p, Ly_p, Lz_p, _ = disp_grid.shape
    Lz_m = Lz_p * factor
    Hx, Hy = halo
    dtype = mesh_padded.dtype

    pmid_x = (jnp.arange(Lx_p, dtype=disp_grid.dtype) * factor) + Hx
    pmid_y = (jnp.arange(Ly_p, dtype=disp_grid.dtype) * factor) + Hy
    pmid_z = jnp.arange(Lz_p, dtype=disp_grid.dtype) * factor
    pos_x = pmid_x[:, None, None] + disp_grid[..., 0]
    pos_y = pmid_y[None, :, None] + disp_grid[..., 1]
    pos_z = pmid_z[None, None, :] + disp_grid[..., 2]
    pos = jnp.stack([pos_x, pos_y, pos_z], axis=-1).reshape(-1, 3)

    kernel_func, _ = kernel_functions[worder]
    stencil_arr, n_ngbs = get_stencil(3, dtype=dtype, worder=worder)
    int_dtype = jnp.int32
    stencil_arr = jnp.asarray(stencil_arr, dtype=int_dtype)

    pos_e = pos[:, None, :]
    if worder % 2 == 0:
        int_coord = jnp.floor(pos_e).astype(int_dtype)
    else:
        int_coord = jnp.round(pos_e).astype(int_dtype)
    neighbor = int_coord + stencil_arr  # (N, n_ngbs, 3)
    abs_diff = jnp.abs(pos_e - neighbor.astype(pos_e.dtype))
    weights = jnp.prod(kernel_func(abs_diff), axis=-1)  # (N, n_ngbs)

    # Wrap fully-local axes (no halo padding) modulo the padded length. For
    # sharded axes (Hx>0 / Hy>0) we rely on the halo-filled padded mesh and
    # use ``mode='fill', fill_value=0`` to gate any reads outside the padded
    # extent (note: NaN-on-OOB is JAX's default for .get(mode='drop'), so we
    # use 'fill' explicitly to keep the gather safe).
    Lx_padded = mesh_padded.shape[0]
    Ly_padded = mesh_padded.shape[1]
    nx = jnp.mod(neighbor[..., 0], Lx_padded) if Hx == 0 else neighbor[..., 0]
    ny = jnp.mod(neighbor[..., 1], Ly_padded) if Hy == 0 else neighbor[..., 1]
    nz = jnp.mod(neighbor[..., 2], Lz_m)

    vals = mesh_padded.at[nx, ny, nz].get(
        mode='fill', fill_value=0.0,
    )  # (N, n_ngbs)
    out = jnp.sum(vals * weights, axis=-1)
    return out.reshape(Lx_p, Ly_p, Lz_p)


# ---------------------------------------------------------------------------
# Pattern A: distributed CIC/TSC/PCS density assignment (paint)
# ---------------------------------------------------------------------------

def scatter_dx_distributed(disp_grid, halo_size: int, worder: int,
                           n_part_tot: int, res_pm: int, boxsize: float,
                           dtype_num: int, sharding: NamedSharding):
    """Distributed density assignment with ghost-zone halo exchange.

    Pipeline (JaxPM Pattern A):
      1. Local scatter onto an ``H``-padded PM mesh (no comm).
      2. ``halo_exchange`` swaps the OUTER ``H/2`` ghost cells with neighbors.
      3. ``slice_unpad`` folds the received outer ghost into the first/last
         ``H/2`` interior cells and trims off the full padding.

    Supports ``res_pm = factor * res_part`` with integer ``factor >= 1``: the
    displacement field is at PARTICLE resolution; the painted/returned mesh
    is at PM resolution. ``factor`` is inferred from ``disp_grid``'s global
    shape and ``res_pm``.

    Parameters
    ----------
    disp_grid : (Lx_p, Ly_p, Lz_p, 3) displacement in PM-GRID UNITS
        = ``psi / cell_size_pm``, sharded ``P('x','y',None,None)``.
        Globally this is ``(res_part, res_part, res_part, 3)``.
    halo_size : ``H`` -- ghost-zone depth (PM cells per side). Must satisfy
        ``H/2 >= max_disp_in_pm_cells + worder//2`` to avoid silently dropping
        boundary deposits.
    worder    : 2 (CIC), 3 (TSC), 4 (PCS).
    n_part_tot, res_pm : global particle count and PM resolution.
    boxsize   : physical box size (Mpc/h). Currently unused inside the function
        (disp is already in PM-grid units) but kept in the signature for
        symmetry with the legacy API.
    dtype_num : 32 or 64.

    Returns
    -------
    delta : (Lx_m, Ly_m, Lz_m) density contrast ``rho/rho_mean - 1`` at PM
            resolution, sharded ``P('x','y',None)``. Globally
            ``(res_pm, res_pm, res_pm)``.
    """
    del boxsize  # unused (disp already in grid units)
    px, py = get_local_pdims(sharding)
    Hx = halo_size if px > 1 else 0
    Hy = halo_size if py > 1 else 0

    # Infer res_part and factor from the displacement's global shape.
    # disp_grid sharded P('x','y',None,None); axis 2 (Z) is unsharded so its
    # local size equals the global res_part.
    res_part = int(disp_grid.shape[2])
    if res_pm % res_part != 0:
        raise ValueError(
            f"res_pm={res_pm} must be an integer multiple of "
            f"res_part={res_part} (got factor={res_pm/res_part})."
        )
    factor = res_pm // res_part
    if res_part % px != 0 or res_part % py != 0:
        raise ValueError(
            f"res_part={res_part} must be divisible by pdims={px, py}."
        )

    pad_width = ((Hx, Hx), (Hy, Hy), (0, 0))
    halo_extents = (Hx // 2, Hy // 2)
    dtype = jnp.float64 if dtype_num == 64 else jnp.float32

    # Step 1: local scatter onto padded mesh (per-rank).
    mesh = sharding.mesh
    axis_x, axis_y = get_mesh_axis_names(sharding)
    in_spec = P(axis_x, axis_y, None, None)
    out_spec = P(axis_x, axis_y, None)

    @partial(shard_map, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec,
             check_rep=False)
    def _scatter_step(d):
        return _local_scatter_padded(d, halo=(Hx, Hy), worder=worder,
                                     dtype=dtype, factor=factor)

    rho_padded = _scatter_step(disp_grid)

    # Step 2: ghost swap of OUTER H/2 cells with neighbors.
    if halo_extents[0] > 0 or halo_extents[1] > 0:
        rho_padded = jaxdecomp.halo_exchange(
            rho_padded,
            halo_extents=halo_extents,
            halo_periods=(True, True),
        )

    # Step 3: fold received ghost into interior, trim.
    rho_count = slice_unpad(rho_padded, pad_width, sharding)

    # Convert raw count to density contrast (matches DISCO-DJ scatter convention).
    rhomean = n_part_tot / (res_pm ** 3)
    delta = rho_count / rhomean - 1.0
    return delta


# ---------------------------------------------------------------------------
# Pattern B: distributed CIC/TSC/PCS force interpolation (gather)
# ---------------------------------------------------------------------------

def gather_dx_distributed(acc_field, disp_grid, halo_size: int, worder: int,
                          sharding: NamedSharding):
    """Distributed force interpolation with halo fill.

    Pipeline (JaxPM Pattern B):
      1. Pad the PM field with ``H/2`` zeros per side on the sharded axes.
      2. ``halo_exchange`` fills the entire ``H/2`` ghost zone with neighbors'
         INTERIOR cells (since the field has no ghost zone of its own).
      3. Local gather from the (now correctly-filled) padded field at the
         particle positions.

    Supports ``res_pm = factor * res_part`` with integer ``factor >= 1``: the
    PM field is at PM resolution; the displacement (and the returned per-
    particle field) are at PARTICLE resolution. ``factor`` is inferred from
    the global shapes of ``acc_field`` and ``disp_grid``.

    Parameters
    ----------
    acc_field : (Lx_m, Ly_m, Lz_m) force component at PM resolution,
                sharded ``P('x','y',None)``. Globally ``(res_pm,)*3``.
    disp_grid : (Lx_p, Ly_p, Lz_p, 3) displacement in PM-GRID UNITS,
                sharded ``P('x','y',None,None)``. Globally
                ``(res_part, res_part, res_part, 3)``.
    halo_size : same ``H`` used in ``scatter_dx_distributed``. The gather
                halves it internally (the read pattern only needs ``H/2``).
    worder    : 2/3/4.

    Returns
    -------
    acc_at_particles : (Lx_p, Ly_p, Lz_p) at PARTICLE resolution,
                        sharded ``P('x','y',None)``.
    """
    px, py = get_local_pdims(sharding)
    # Halve the halo for the gather pattern.
    Hx = (halo_size // 2) if px > 1 else 0
    Hy = (halo_size // 2) if py > 1 else 0

    # Infer factor from the global shapes (Z is unsharded for both arrays,
    # so the local Lz equals the global Lz).
    res_pm = int(acc_field.shape[2])
    res_part = int(disp_grid.shape[2])
    if res_pm % res_part != 0:
        raise ValueError(
            f"acc_field global size ({res_pm}) must be an integer multiple of "
            f"disp_grid global size ({res_part}); got factor={res_pm/res_part}."
        )
    factor = res_pm // res_part

    pad_width = ((Hx, Hx), (Hy, Hy), (0, 0))
    halo_extents = (Hx, Hy)
    mesh = sharding.mesh
    axis_x, axis_y = get_mesh_axis_names(sharding)

    field_spec = P(axis_x, axis_y, None)
    disp_spec = P(axis_x, axis_y, None, None)

    # Step 1: pad force field.
    @partial(shard_map, mesh=mesh, in_specs=(field_spec,),
             out_specs=field_spec, check_rep=False)
    def _pad_step(f):
        return jnp.pad(f, pad_width, mode='constant')

    acc_padded = _pad_step(acc_field)

    # Step 2: halo fill (ghost = neighbor's interior values).
    if halo_extents[0] > 0 or halo_extents[1] > 0:
        acc_padded = jaxdecomp.halo_exchange(
            acc_padded,
            halo_extents=halo_extents,
            halo_periods=(True, True),
        )

    # Step 3: local gather. Output is at PARTICLE resolution.
    @partial(shard_map, mesh=mesh, in_specs=(field_spec, disp_spec),
             out_specs=field_spec, check_rep=False)
    def _gather_step(m, d):
        return _local_gather_padded(m, d, halo=(Hx, Hy), worder=worder,
                                    factor=factor)

    return _gather_step(acc_padded, disp_grid)


# ---------------------------------------------------------------------------
# gradient_kernel for the full-complex (pfft3d) FFT layout
# ---------------------------------------------------------------------------

def gradient_kernel_dist(k_vecs, axis: int, order: int = 0):
    """Gradient kernel matching DISCO-DJ's semantics on the full-FFT (pfft3d)
    output layout.

    DISCO-DJ's ``gradient_kernel(order=0)`` zeros the Nyquist mode at index
    ``-1`` for ``axis=dim-1`` (rfft convention). After ``pfft3d`` the output
    is full-complex, with Nyquist always at index ``N//2``. This wrapper:
      * delegates to the original ``gradient_kernel`` for ``axis < dim-1`` and
        ``order != 0`` (where the rfft assumption never fires);
      * substitutes a corrected ``1j*k`` with ``at[N//2].set(0)`` for the
        ``axis = dim-1, order = 0`` case.

    The original ``gradient_kernel`` source is **not** modified.
    """
    if order != 0 or axis != len(k_vecs) - 1:
        return _disco_gradient_kernel(k_vecs, axis=axis, order=order)

    # axis == dim-1, order == 0: fix the Nyquist position.
    kernel = 1j * k_vecs[axis]
    shape = kernel.shape
    # Find the single non-broadcast axis of k_vecs[axis].
    nax = next(i for i, s in enumerate(shape) if s > 1)
    N = shape[nax]
    idx = [slice(None)] * len(shape)
    idx[nax] = N // 2
    return kernel.at[tuple(idx)].set(0.0)

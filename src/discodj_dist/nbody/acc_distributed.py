"""Distributed (multi-GPU) version of DISCO-DJ's PM force computation.

This file is *additive*: the original ``calc_acc_PM_or_Tree_PM`` /
``calc_acc_PM_`` in ``acc.py`` is untouched. ``calc_acc_PM_distributed`` here
is a stand-alone refactored version that:

* keeps the DKD-leapfrog control flow identical (same scan over ``dim`` force
  components, same kernels, same dtype handling);
* replaces ``jnp.fft.rfftn`` / ``jnp.fft.irfftn`` with
  ``jaxdecomp.pfft3d`` / ``jaxdecomp.pifft3d`` (full-complex distributed FFT);
* rebuilds the k-vectors dynamically from ``jaxdecomp.fftfreq3d`` (no static
  ``k_vecs_pm`` precompute is needed any more) and reorders them to
  ``[k_X, k_Y, k_Z]`` so DISCO-DJ's existing ``inv_laplace_kernel`` works
  unchanged;
* replaces ``scatter`` / ``gather`` with the new
  ``scatter_dx_distributed`` / ``gather_dx_distributed`` from
  ``core/distributed_pm.py``.

The function works on a single GPU (with ``pdims=(1,1)``) as a degenerate
case -- this is how the unit test in ``tests/test_distributed_pm.py``
validates that the new code reproduces the legacy result bit-for-bit
(modulo the small Nyquist-mode handling shift, which is exactly compensated
in the test).
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

import jaxdecomp

from ..core.distributed_pm import (
    scatter_dx_distributed,
    gather_dx_distributed,
    gradient_kernel_dist,
    get_local_pdims,
)
from ..core.kernels import inv_laplace_kernel, inv_mak_kernel
from ..core.utils import set_0_to_val


__all__ = ["calc_acc_PM_distributed"]


def _build_k_vecs_dist(fdelta_k, boxsize: float, res_pm: int):
    """Construct the angular-frequency k-vectors for DISCO-DJ's kernels from
    the output of ``pfft3d``.

    Returns ``[k_X, k_Y, k_Z]`` with broadcast shapes:
        k_X : (1, 1, res_pm)   broadcasts along array axis 2 (physical X)
        k_Y : (res_pm, 1, 1)   broadcasts along array axis 0 (physical Y)
        k_Z : (1, res_pm, 1)   broadcasts along array axis 1 (physical Z)

    The ordering matches DISCO-DJ's ``gradient_kernel(k_vecs, axis=d)``
    convention: ``k_vecs[d]`` is the k-vector along physical direction ``d``,
    so ``axis=0`` corresponds to the physical X direction, etc.

    ``jaxdecomp.fftfreq3d`` already returns the k-vectors in physical
    (X, Y, Z) order, multiplied by 2π (angular frequency, not cyclic),
    with shapes broadcast-compatible with the post-pfft3d (Y, Z, X) array
    layout. So we just take its outputs verbatim.
    """
    cell_size = boxsize / res_pm
    k_X, k_Y, k_Z = jaxdecomp.fftfreq3d(fdelta_k, d=cell_size)
    dtype = jnp.float32 if fdelta_k.dtype == jnp.complex64 else jnp.float64
    return [k_X.astype(dtype), k_Y.astype(dtype), k_Z.astype(dtype)]


def calc_acc_PM_distributed(
    psi_grid,
    *,
    dim: int = 3,
    res_pm: int,
    boxsize: float,
    halo_size: int,
    sharding: NamedSharding,
    grad_order: int = 0,
    lap_order: int = 0,
    dtype_num: int = 32,
    worder: int = 2,
    deconvolve: bool = False,
):
    """SPMD-aware refactor of ``calc_acc_PM_`` for distributed PM.

    Supports ``res_pm = factor * res_part`` with integer ``factor >= 1``: the
    displacement field is at PARTICLE resolution; the FFT mesh is at PM
    resolution; the returned per-particle acceleration is at PARTICLE
    resolution. ``factor`` is inferred from ``psi_grid``'s global shape and
    ``res_pm``.

    Parameters
    ----------
    psi_grid : (Lx_p, Ly_p, Lz_p, 3) displacement field at PARTICLE resolution
        in **Mpc/h** (NOT grid units), sharded ``P('x','y',None,None)``.
        Globally this represents ``(res_part, res_part, res_part, 3)``.
    dim : must be 3 (multi-GPU is only implemented for 3-D).
    res_pm : global PM mesh resolution per dimension. Must satisfy
        ``res_pm % res_part == 0``.
    boxsize : physical box size (Mpc/h).
    halo_size : ``H`` -- ghost-zone depth (in PM cells) used by both scatter
        and gather. Must satisfy
        ``H/2 >= worder//2 + max_disp_in_pm_cells_per_step``.
    sharding : ``NamedSharding(jax.make_mesh(pdims, ('x','y')), P('x','y',...))``.
    grad_order, lap_order : as in DISCO-DJ's ``calc_acc_PM``.
        ``grad_order=0`` uses the exact ``ik`` Fourier derivative (with the
        Nyquist mode at ``N//2`` zeroed for the full-FFT layout).
    dtype_num : 32 or 64.
    worder : 2 (CIC), 3 (TSC), 4 (PCS).
    deconvolve : if True, apply ``inv_mak_kernel`` (MAK deconvolution) in k-space.

    Returns
    -------
    acc : (Lx_p, Ly_p, Lz_p, 3) acceleration at PARTICLE resolution,
          sharded ``P('x','y',None,None)``.
    """
    if dim != 3:
        raise NotImplementedError(
            "calc_acc_PM_distributed currently only supports dim=3 "
            "(jaxDecomp's pfft3d is 3-D)."
        )
    px, py = get_local_pdims(sharding)

    # Particle resolution comes from psi_grid's global shape (Z is unsharded,
    # so its local size equals the global size).
    res_part = int(psi_grid.shape[2])
    if res_part % px != 0 or res_part % py != 0:
        raise ValueError(
            f"res_part={res_part} must be divisible by pdims={px, py}."
        )
    if res_pm % res_part != 0:
        raise ValueError(
            f"res_pm={res_pm} must be an integer multiple of "
            f"res_part={res_part}."
        )
    if res_pm % px != 0 or res_pm % py != 0:
        raise ValueError(
            f"res_pm={res_pm} must be divisible by pdims={px, py}."
        )

    cell_size = boxsize / res_pm  # PM cell size — matches displacement units
    n_part_tot = res_part ** dim
    dtype = jnp.float64 if dtype_num == 64 else jnp.float32
    dtype_c = jnp.complex128 if dtype_num == 64 else jnp.complex64

    # ----- 1. displacement (Mpc/h) -> PM-grid units -----------------------
    # disp_grid is at PARTICLE resolution but uses PM-cell units (so the
    # scatter/gather kernels can index directly into the PM mesh).
    disp_grid = (psi_grid / cell_size).astype(dtype)

    # ----- 2. distributed density assignment ------------------------------
    delta = scatter_dx_distributed(
        disp_grid,
        halo_size=halo_size,
        worder=worder,
        n_part_tot=n_part_tot,
        res_pm=res_pm,
        boxsize=boxsize,
        dtype_num=dtype_num,
        sharding=sharding,
    )  # (Lx_m, Ly_m, Lz_m) at PM res, sharded P('x','y',None)

    # ----- 3. forward distributed FFT (full complex) ----------------------
    # pfft3d requires complex input; cast and rely on XLA to optimise the
    # zero-imaginary part. The output is complex in the cyclic-permuted
    # (Y, Z, X) layout, with the SAME per-shard shape for cubic meshes.
    fdelta_raw = jaxdecomp.pfft3d(delta.astype(dtype_c), norm='backward')

    # ----- 4. build angular k-vectors from fftfreq3d, then optional MAK ---
    k_vecs_dist = _build_k_vecs_dist(fdelta_raw, boxsize=boxsize,
                                     res_pm=res_pm)

    if deconvolve:
        # NOTE: inv_mak_kernel is a pointwise function of k_vecs; it is
        # broadcast-compatible with the (Y,Z,X) layout since multiplication
        # is independent of axis-name semantics.
        fdelta_raw = fdelta_raw * inv_mak_kernel(
            k_vecs_dist, account_for_shotnoise=False, double_exponent=True,
            worder=worder, with_jax=True,
        )

    fdelta = set_0_to_val(dim, fdelta_raw, 0.0)

    # ----- 5. Poisson + gradient + IFFT + gather, per force component -----
    inv_lap = inv_laplace_kernel(k_vecs_dist, order=lap_order, with_jax=True)
    pot_k = inv_lap * fdelta  # PM-resolution complex k-space potential

    # Per-axis force components. Use a Python loop (not lax.scan) so that
    # `axis=d` is a static int -- gradient_kernel_dist branches on axis at
    # trace time, which fails if d is a tracer. dim is small (1/2/3), so
    # full unrolling is fine.
    accs = []
    for d in range(dim):
        gradk = gradient_kernel_dist(k_vecs_dist, axis=d, order=grad_order)
        facc = -gradk * pot_k  # PM-res complex
        # Inverse distributed FFT: complex (Y,Z,X) layout -> real (X,Y,Z) layout
        acc_field_d = jaxdecomp.pifft3d(facc, norm='backward').real.astype(dtype)
        # Distributed gather: PM-res field -> per-particle (PARTICLE-res) values.
        acc_d = gather_dx_distributed(
            acc_field_d,
            disp_grid,
            halo_size=halo_size,
            worder=worder,
            sharding=sharding,
        )  # (Lx_p, Ly_p, Lz_p) at PARTICLE res
        accs.append(acc_d)

    # Stack along last axis to match psi_grid layout (PARTICLE-res).
    return jnp.stack(accs, axis=-1)  # (Lx_p, Ly_p, Lz_p, 3)

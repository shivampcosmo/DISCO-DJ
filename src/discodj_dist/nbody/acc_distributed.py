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
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

import jaxdecomp

from ..core.distributed_pm import (
    scatter_dx_distributed,
    gather_dx_distributed,
    gradient_kernel_dist,
    get_local_pdims,
    get_mesh_axis_names,
)
from ..core.kernels import gradient_kernel as _disco_gradient_kernel
from ..core.kernels import inv_laplace_kernel, inv_mak_kernel


__all__ = ["calc_acc_PM_distributed", "kick_PM_distributed"]


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


def _is_slab_rfft_backend(fft_backend: str) -> bool:
    return fft_backend.strip().lower() in {"jax_rfft", "rfft", "slab_rfft"}


def _require_slab_rfft_sharding(sharding: NamedSharding, res_pm: int) -> tuple[int, int]:
    px, py = get_local_pdims(sharding)
    if py != 1:
        raise ValueError(
            "DISCO_PM_FFT_BACKEND=JAX_RFFT currently requires a slab mesh "
            f"with pdims=(n,1), got pdims={(px, py)}.  This avoids splitting "
            f"the real-FFT half-spectrum of length {res_pm // 2 + 1}."
        )
    if res_pm % px != 0:
        raise ValueError(f"res_pm={res_pm} must be divisible by slab pdims[0]={px}.")
    return px, py


def _slab_prfft3d_x(field, *, sharding: NamedSharding):
    """Distributed real FFT for an X-slab input layout.

    Input is sharded as ``P('x','y',None)`` with ``mesh.shape['y'] == 1`` and
    global shape ``(N,N,N)``.  Output has global shape ``(N,N,N//2+1)`` and is
    sharded as ``P(None,'x',None)``.  The transform is algebraically the same
    as ``jnp.fft.rfftn(field)`` in backward normalization.
    """
    mesh = sharding.mesh
    axis_x, axis_y = get_mesh_axis_names(sharding)

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(axis_x, axis_y, None),),
        out_specs=P(None, axis_x, None),
        check_rep=False,
    )
    def _fft(local_field):
        local_k = jnp.fft.rfft2(local_field, axes=(1, 2))
        local_k = lax.all_to_all(
            local_k, axis_x, split_axis=1, concat_axis=0, tiled=True
        )
        return jnp.fft.fft(local_k, axis=0)

    return _fft(field)


def _slab_pirfft3d_x(field_k, *, sharding: NamedSharding, res_pm: int, dtype):
    """Inverse of ``_slab_prfft3d_x``."""
    mesh = sharding.mesh
    axis_x, axis_y = get_mesh_axis_names(sharding)

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(None, axis_x, None),),
        out_specs=P(axis_x, axis_y, None),
        check_rep=False,
    )
    def _ifft(local_field_k):
        local_field = jnp.fft.ifft(local_field_k, axis=0)
        local_field = lax.all_to_all(
            local_field, axis_x, split_axis=0, concat_axis=1, tiled=True
        )
        return jnp.fft.irfft2(local_field, s=(res_pm, res_pm), axes=(1, 2)).astype(dtype)

    return _ifft(field_k)


def _build_k_vecs_slab_rfft(fdelta_k, boxsize: float, res_pm: int):
    """Angular k-vectors for ``_slab_prfft3d_x`` layout.

    The spectral layout is the standard real-FFT layout with global shape
    ``(kx, ky, kz_rfft)`` and sharding ``P(None,'x',None)``.
    """
    cell_size = boxsize / res_pm
    dtype = jnp.float32 if fdelta_k.dtype == jnp.complex64 else jnp.float64
    kx = jnp.fft.fftfreq(res_pm, d=cell_size, dtype=dtype) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(res_pm, d=cell_size, dtype=dtype) * 2 * jnp.pi
    kz = jnp.fft.rfftfreq(res_pm, d=cell_size, dtype=dtype) * 2 * jnp.pi
    return [
        kx.reshape(-1, 1, 1),
        ky.reshape(1, -1, 1),
        kz.reshape(1, 1, -1),
    ]


def _gradient_kernel_pm(k_vecs, axis: int, order: int, *, real_fft: bool):
    if real_fft:
        return _disco_gradient_kernel(k_vecs, axis=axis, order=order, with_jax=True)
    return gradient_kernel_dist(k_vecs, axis=axis, order=order)


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
    fft_backend: str = "JAX",
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
    fft_backend : backend passed to ``jaxdecomp.pfft3d`` / ``pifft3d``.
        Use ``"JAX"`` for the large PM mesh unless cuDecomp has been validated
        for the target resolution; cuFFT workspace can be the dominant peak.

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
    real_fft = _is_slab_rfft_backend(fft_backend)
    if real_fft:
        _require_slab_rfft_sharding(sharding, res_pm)
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

    # ----- 3. forward distributed FFT -------------------------------------
    if real_fft:
        fdelta_raw = _slab_prfft3d_x(delta.astype(dtype), sharding=sharding)
        k_vecs_dist = _build_k_vecs_slab_rfft(fdelta_raw, boxsize=boxsize,
                                              res_pm=res_pm)
    else:
        # pfft3d requires complex input; cast and rely on XLA to optimise the
        # zero-imaginary part. The output is complex in the cyclic-permuted
        # (Y, Z, X) layout, with the SAME per-shard shape for cubic meshes.
        fdelta_raw = jaxdecomp.pfft3d(delta.astype(dtype_c), norm='backward', backend=fft_backend)

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

    # ----- 5. Poisson + gradient + IFFT + gather, per force component -----
    inv_lap = None
    if lap_order != 0:
        inv_lap = inv_laplace_kernel(k_vecs_dist, order=lap_order, with_jax=True)

    # Per-axis force components. Use a Python loop (not lax.scan) so that
    # `axis=d` is a static int -- gradient_kernel_dist branches on axis at
    # trace time, which fails if d is a tracer. dim is small (1/2/3), so
    # full unrolling is fine.
    accs = []
    for d in range(dim):
        acc_d = _pm_force_component_at_particles(
            fdelta_raw,
            disp_grid,
            k_vecs_dist,
            axis=d,
            inv_lap=inv_lap,
            halo_size=halo_size,
            grad_order=grad_order,
            lap_order=lap_order,
            worder=worder,
            sharding=sharding,
            dtype=dtype,
            fft_backend=fft_backend,
            real_fft=real_fft,
            res_pm=res_pm,
        )
        accs.append(acc_d)

    # Stack along last axis to match psi_grid layout (PARTICLE-res).
    return jnp.stack(accs, axis=-1)  # (Lx_p, Ly_p, Lz_p, 3)


def _pm_force_component_at_particles(
    fdelta,
    disp_grid,
    k_vecs_dist,
    *,
    axis: int,
    inv_lap,
    halo_size: int,
    grad_order: int,
    lap_order: int,
    worder: int,
    sharding: NamedSharding,
    dtype,
    fft_backend: str,
    real_fft: bool = False,
    res_pm: int | None = None,
):
    gradk = _gradient_kernel_pm(k_vecs_dist, axis=axis, order=grad_order, real_fft=real_fft)
    if lap_order == 0:
        # Exact Poisson kernel, fused into the force multiply. This avoids
        # materializing a persistent full PM-resolution inverse-Laplace field.
        ksquare = sum(ki ** 2 for ki in k_vecs_dist)
        mask = ksquare != 0
        safe_ksquare = jnp.where(mask, ksquare, 1.0)
        kernel = (gradk / safe_ksquare) * mask.astype(safe_ksquare.dtype)
        facc = kernel * fdelta
    else:
        facc = -gradk * inv_lap * fdelta

    if real_fft:
        if res_pm is None:
            raise ValueError("res_pm must be provided for real-FFT PM inverse.")
        acc_field_d = _slab_pirfft3d_x(facc, sharding=sharding,
                                       res_pm=res_pm, dtype=dtype)
    else:
        acc_field_d = jaxdecomp.pifft3d(facc, norm='backward', backend=fft_backend).real.astype(dtype)
    return gather_dx_distributed(
        acc_field_d,
        disp_grid,
        halo_size=halo_size,
        worder=worder,
        sharding=sharding,
    )


def kick_PM_distributed(
    psi_grid,
    mom_grid,
    *,
    alpha,
    beta,
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
    fft_backend: str = "JAX",
):
    """Apply the PM kick ``alpha*mom + beta*acc_PM(psi)`` without returning acc.

    This is algebraically equivalent to calling ``calc_acc_PM_distributed`` and
    then forming the momentum update, but it avoids materializing a full
    particle-resolution acceleration vector in the DKD step.
    """
    if dim != 3:
        raise NotImplementedError(
            "kick_PM_distributed currently only supports dim=3 "
            "(jaxDecomp's pfft3d is 3-D)."
        )
    px, py = get_local_pdims(sharding)
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

    cell_size = boxsize / res_pm
    n_part_tot = res_part ** dim
    dtype = jnp.float64 if dtype_num == 64 else jnp.float32
    real_fft = _is_slab_rfft_backend(fft_backend)
    if real_fft:
        _require_slab_rfft_sharding(sharding, res_pm)
    dtype_c = jnp.complex128 if dtype_num == 64 else jnp.complex64
    disp_grid = (psi_grid / cell_size).astype(dtype)

    delta = scatter_dx_distributed(
        disp_grid,
        halo_size=halo_size,
        worder=worder,
        n_part_tot=n_part_tot,
        res_pm=res_pm,
        boxsize=boxsize,
        dtype_num=dtype_num,
        sharding=sharding,
    )
    if real_fft:
        fdelta_raw = _slab_prfft3d_x(delta.astype(dtype), sharding=sharding)
        k_vecs_dist = _build_k_vecs_slab_rfft(fdelta_raw, boxsize=boxsize,
                                              res_pm=res_pm)
    else:
        fdelta_raw = jaxdecomp.pfft3d(delta.astype(dtype_c), norm='backward', backend=fft_backend)
        k_vecs_dist = _build_k_vecs_dist(fdelta_raw, boxsize=boxsize, res_pm=res_pm)

    if deconvolve:
        fdelta_raw = fdelta_raw * inv_mak_kernel(
            k_vecs_dist, account_for_shotnoise=False, double_exponent=True,
            worder=worder, with_jax=True,
        )

    inv_lap = None
    if lap_order != 0:
        inv_lap = inv_laplace_kernel(k_vecs_dist, order=lap_order, with_jax=True)

    mom_components = []
    for d in range(dim):
        acc_d = _pm_force_component_at_particles(
            fdelta_raw,
            disp_grid,
            k_vecs_dist,
            axis=d,
            inv_lap=inv_lap,
            halo_size=halo_size,
            grad_order=grad_order,
            lap_order=lap_order,
            worder=worder,
            sharding=sharding,
            dtype=dtype,
            fft_backend=fft_backend,
            real_fft=real_fft,
            res_pm=res_pm,
        )
        mom_d = alpha * mom_grid[..., d] + beta * acc_d.astype(mom_grid.dtype)
        mom_components.append(mom_d)
    return jnp.stack(mom_components, axis=-1)

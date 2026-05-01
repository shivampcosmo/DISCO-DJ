"""Unit test: distributed PM (multi-GPU code path) vs original single-GPU PM.

The test runs a small-scale PM force computation TWICE on the same problem:

  1. Reference path: DISCO-DJ's existing ``calc_acc_PM`` (single-GPU,
     ``jnp.fft.rfftn`` + scatter/gather with ``np.mod`` periodic wrap).

  2. Distributed path: ``calc_acc_PM_distributed`` (the new code in
     ``acc_distributed.py``), built on ``jaxdecomp.pfft3d`` + the new
     ``scatter_dx_distributed`` / ``gather_dx_distributed`` halo-exchange
     primitives in ``core/distributed_pm.py``.

Both paths SHOULD produce the same accelerations (up to floating-point
rounding). On a single device with ``pdims=(1,1)`` this exercises the new
code path end-to-end (FFT swap, k-vec rebuild, Lagrangian-painting scatter
with the padded mesh and slice_unpad fold) without any actual halo
communication. On a multi-device CPU emulation (set the env var BEFORE
importing JAX), the same comparison validates that the halo exchange is
correct end-to-end.

Usage
-----
Single-device:
    python -m discodj.tests.test_distributed_pm

Multi-device CPU emulation (e.g. 2 'devices'):
    XLA_FLAGS="--xla_force_host_platform_device_count=2" \
      DISCODJ_PDIMS=1,2 \
      python -m discodj.tests.test_distributed_pm

Multi-GPU:
    python -m discodj.tests.test_distributed_pm  # will pick up all visible GPUs
"""
from __future__ import annotations

import os
import sys

import numpy as onp


def _maybe_emulate_devices():
    """Read DISCODJ_PDIMS env var and request that many CPU devices BEFORE
    JAX is imported. Idempotent; no-op if XLA_FLAGS already set or if running
    on real GPUs."""
    pdims_str = os.environ.get("DISCODJ_PDIMS", "1,1")
    pdims = tuple(int(s) for s in pdims_str.split(","))
    n_dev = pdims[0] * pdims[1]
    if n_dev > 1 and "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = (
            f"--xla_force_host_platform_device_count={n_dev}"
        )
    return pdims


_PDIMS = _maybe_emulate_devices()

# JAX must be imported AFTER the XLA_FLAGS env var is set.
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import mesh_utils  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

import jaxdecomp  # noqa: E402

# Make the parallel `discodj_dist` package importable when running this test
# directly from a checkout (the legacy `discodj` package is shipped via pip).
import sys as _sys  # noqa: E402
_DISCODJ_SRC = '/mnt/ceph/users/spandey/quijote_v2_gotham/DISCO-DJ/src'
if _DISCODJ_SRC not in _sys.path:
    _sys.path.append(_DISCODJ_SRC)

from discodj.core.grids import get_fourier_grid  # noqa: E402
from discodj.nbody.acc import calc_acc_PM  # noqa: E402
from discodj_dist.nbody.acc_distributed import calc_acc_PM_distributed  # noqa: E402


def _make_test_problem(res: int, boxsize: float, dtype, seed: int = 0):
    """Generate small random Lagrangian displacements (Mpc/h) on a (res^3, 3)
    array. Displacements are kept to a small fraction of a cell to ensure no
    particle exits the local subdomain even with tight halos.
    """
    cell_size = boxsize / res
    n = res ** 3
    key = jax.random.PRNGKey(seed)
    # Max displacement ~ 0.3 cells in each direction.
    psi_flat = 0.3 * cell_size * jax.random.normal(key, (n, 3), dtype=dtype)
    return psi_flat


def _run_reference(psi_flat, res, res_pm, boxsize, dtype_num, worder=2):
    """DISCO-DJ's original single-GPU calc_acc_PM."""
    k_dict = get_fourier_grid(
        (res_pm,) * 3, boxsize=boxsize, sparse_k_vecs=True,
        full=False, dtype_num=dtype_num,
    )
    k_pm = k_dict["|k|"]
    k_vecs_pm = k_dict["k_vecs"]
    return calc_acc_PM(
        psi=psi_flat, dim=3, res_pm=res_pm, n_part=res,
        k=k_pm, k_vecs=k_vecs_pm,
        boxsize=boxsize, antialias=0,
        grad_order=0, lap_order=0, dtype_num=dtype_num,
        worder=worder, with_jax=True,
    )


def _run_distributed(psi_flat, res, res_pm, boxsize, dtype_num, sharding,
                     halo_size, worder=2):
    """The new distributed calc_acc_PM_distributed."""
    psi_grid = psi_flat.reshape(res, res, res, 3)
    psi_grid = jax.device_put(psi_grid, sharding)

    @jax.jit
    def _go(p):
        return calc_acc_PM_distributed(
            p,
            dim=3,
            res_pm=res_pm,
            boxsize=boxsize,
            halo_size=halo_size,
            sharding=sharding,
            grad_order=0,
            lap_order=0,
            dtype_num=dtype_num,
            worder=worder,
        )

    acc_grid = _go(psi_grid)
    # acc_grid: (Lx, Ly, Lz, 3) sharded across (x, y); collect to host.
    acc_host = jax.device_get(acc_grid)
    return onp.asarray(acc_host).reshape(-1, 3)


def main():
    pdims = _PDIMS
    n_dev_required = pdims[0] * pdims[1]
    n_dev_avail = jax.device_count()
    print(f"[test_distributed_pm] devices available: {n_dev_avail}, "
          f"requested pdims: {pdims} ({n_dev_required} ranks)")
    if n_dev_avail < n_dev_required:
        print(
            f"  Skipping multi-device test (need {n_dev_required}, have "
            f"{n_dev_avail}). Falling back to pdims=(1,1)."
        )
        pdims = (1, 1)

    # NOTE: jaxdecomp.init() is a no-op for the JAX backend on a single
    # process (and the JAX backend is the default for pfft3d / halo_exchange).
    # If running on real GPUs across multiple ranks via SLURM, the user must
    # call jax.distributed.initialize() and jaxdecomp.init() in their entry
    # script BEFORE this test runs.

    # ---- problem size --------------------------------------------------
    res = 16        # particles per dim
    res_pm = 16     # PM mesh resolution per dim (must equal res for default)
    boxsize = 100.0
    worder = 2      # CIC

    # Use float64 for a stricter comparison.
    dtype_num = 64
    dtype = jnp.float64
    jax.config.update("jax_enable_x64", True)

    # ---- mesh / sharding -----------------------------------------------
    devices = mesh_utils.create_device_mesh(pdims)
    mesh = Mesh(devices, axis_names=('x', 'y'))
    sharding_disp = NamedSharding(mesh, P('x', 'y', None, None))

    # halo_size: must be even (we exchange halo_size//2). For pdims=(1,1)
    # the halo is internally set to 0 -- value here doesn't matter.
    # For larger pdims, choose H so H/2 > worder//2 + max_disp_cells (~0.3).
    halo_size = 4

    # ---- generate problem ----------------------------------------------
    psi_flat = _make_test_problem(res, boxsize, dtype=dtype, seed=0)

    # ---- reference ------------------------------------------------------
    print("[test_distributed_pm] running single-GPU reference ...")
    acc_ref = _run_reference(psi_flat, res, res_pm, boxsize, dtype_num,
                             worder=worder)
    acc_ref = onp.asarray(acc_ref)
    print(f"  acc_ref shape  : {acc_ref.shape}")
    print(f"  |acc_ref|_max  : {onp.abs(acc_ref).max():.6e}")

    # ---- distributed ----------------------------------------------------
    print(f"[test_distributed_pm] running distributed (pdims={pdims}) ...")
    acc_dist = _run_distributed(
        psi_flat, res, res_pm, boxsize, dtype_num,
        sharding=sharding_disp, halo_size=halo_size, worder=worder,
    )
    print(f"  acc_dist shape : {acc_dist.shape}")
    print(f"  |acc_dist|_max : {onp.abs(acc_dist).max():.6e}")

    # ---- compare --------------------------------------------------------
    abs_err = onp.abs(acc_dist - acc_ref)
    max_abs = abs_err.max()
    max_ref = onp.abs(acc_ref).max()
    rel_err = max_abs / max_ref
    rmse = onp.sqrt((abs_err ** 2).mean())
    print(f"[test_distributed_pm] max abs err = {max_abs:.3e}, "
          f"rel = {rel_err:.3e}, rmse = {rmse:.3e}")

    # Tolerance:
    # - The two paths use DIFFERENT FFTs (rfftn vs full pfft3d) but the same
    #   underlying Poisson math.
    # - The Nyquist mode is treated correctly in both (rfft has Nyquist at -1
    #   in the last axis; gradient_kernel_dist explicitly zeros it at N//2
    #   for the full-FFT layout).
    # - At float64 we expect agreement to a few * 1e-13 in relative terms,
    #   modulo Nyquist-mode artefacts of order 1/R^3.
    tol = 1e-10 if dtype_num == 64 else 1e-5
    if rel_err < tol:
        print(f"[test_distributed_pm] PASS (rel err {rel_err:.3e} < {tol:.0e})")
        return 0
    else:
        print(f"[test_distributed_pm] FAIL (rel err {rel_err:.3e} >= {tol:.0e})")
        return 1


if __name__ == "__main__":
    sys.exit(main())

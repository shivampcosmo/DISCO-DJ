"""Unit test: distributed full-complex nLPT vs DISCO-DJ's single-device nLPT.

Usage
-----
Single device:
    python -m tests.test_distributed_lpt

CPU multi-device emulation:
    DISCODJ_PDIMS=1,2 JAX_PLATFORMS=cpu python -m tests.test_distributed_lpt
"""
from __future__ import annotations

import os
import sys

import numpy as onp


def _maybe_emulate_devices():
    pdims_str = os.environ.get("DISCODJ_PDIMS", "1,1")
    pdims = tuple(int(s) for s in pdims_str.split(","))
    n_dev = pdims[0] * pdims[1]
    if n_dev > 1 and "XLA_FLAGS" not in os.environ:
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n_dev}"
    return pdims


_PDIMS = _maybe_emulate_devices()

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.experimental import mesh_utils  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

import jaxdecomp  # noqa: E402

_DISCODJ_SRC = "/mnt/ceph/users/spandey/quijote_v2_gotham/DISCO-DJ/src"
if _DISCODJ_SRC not in sys.path:
    sys.path.append(_DISCODJ_SRC)

from discodj_dist import DiscoDJ  # noqa: E402
from discodj_dist.lpt.nlpt_distributed import (  # noqa: E402
    pad_fourier_full,
    crop_fourier_full,
    conv2_fourier_distributed,
    compute_2lpt_initial_state_distributed,
)


def _make_mesh(pdims):
    n_dev_required = pdims[0] * pdims[1]
    n_dev_avail = jax.device_count()
    if n_dev_avail < n_dev_required:
        print(
            f"[test_distributed_lpt] requested {n_dev_required} devices but only "
            f"{n_dev_avail} available; falling back to pdims=(1,1)."
        )
        pdims = (1, 1)
    devices = mesh_utils.create_device_mesh(pdims)
    mesh = Mesh(devices, axis_names=("x", "y"))
    return pdims, mesh


def _zero_removed_modes(fk, res: int):
    out = fk.at[0, 0, 0].set(0.0)
    for axis in range(3):
        idx = [slice(None)] * 3
        idx[axis] = res // 2
        out = out.at[tuple(idx)].set(0.0)
    return out


def _relative_rmse(a, b):
    a = onp.asarray(a)
    b = onp.asarray(b)
    diff = a - b
    denom = onp.sqrt(onp.mean(onp.abs(b) ** 2)) + 1e-30
    return float(onp.sqrt(onp.mean(onp.abs(diff) ** 2)) / denom), float(onp.max(onp.abs(diff)))


def _test_pad_crop_and_conv(sharding_field):
    print("[test_distributed_lpt] testing pad/crop and distributed convolution ...")
    res = 8
    ext_res = 12
    dtype = jnp.float64
    dtype_c = jnp.complex128
    key_a, key_b = jax.random.split(jax.random.PRNGKey(11))
    a = jax.random.normal(key_a, (res, res, res), dtype=dtype)
    b = jax.random.normal(key_b, (res, res, res), dtype=dtype)

    a_g = jax.device_put(a, sharding_field)
    b_g = jax.device_put(b, sharding_field)
    fa = jaxdecomp.pfft3d(a_g.astype(dtype_c), norm="backward")
    fb = jaxdecomp.pfft3d(b_g.astype(dtype_c), norm="backward")
    fft_sharding = fa.sharding

    fa_ext = pad_fourier_full(
        fa, orig_res=res, ext_res=ext_res, dtype_c=dtype_c, fft_sharding=fft_sharding
    )
    fa_round = crop_fourier_full(
        fa_ext, orig_res=res, ext_res=ext_res, dtype_c=dtype_c, fft_sharding=fft_sharding
    )
    expected = _zero_removed_modes(fa, res)
    rel, max_abs = _relative_rmse(jax.device_get(fa_round), jax.device_get(expected))
    print(f"  pad/crop rel_rmse={rel:.3e}, max_abs={max_abs:.3e}")
    if rel > 1e-12 or max_abs > 1e-10:
        raise AssertionError("pad/crop round trip failed")

    fb_ext = pad_fourier_full(
        fb, orig_res=res, ext_res=ext_res, dtype_c=dtype_c, fft_sharding=fft_sharding
    )
    conv_ext = conv2_fourier_distributed(
        fa_ext,
        fb_ext,
        orig_res=None,
        ext_res=ext_res,
        dtype_c=dtype_c,
        field_sharding=sharding_field,
        fft_sharding=fft_sharding,
        do_crop=False,
    )
    product_real = (
        jaxdecomp.pifft3d(fa_ext, norm="backward").real
        * jaxdecomp.pifft3d(fb_ext, norm="backward").real
    )
    expected_conv_ext = jaxdecomp.pfft3d(product_real.astype(dtype_c), norm="backward")
    rel, max_abs = _relative_rmse(jax.device_get(conv_ext), jax.device_get(expected_conv_ext))
    print(f"  conv ext rel_rmse={rel:.3e}, max_abs={max_abs:.3e}")
    if rel > 1e-12 or max_abs > 1e-10:
        raise AssertionError("distributed convolution failed")


def _compare_lpt_against_single(
    sharding_field,
    sharding_disp,
    *,
    n_order: int,
    grad_kernel_order: int,
    try_to_jit: bool,
    precision: str = "double",
):
    print(
        f"[test_distributed_lpt] testing {n_order}LPT against single-device "
        f"reference (precision={precision}, grad_order={grad_kernel_order}, jit={try_to_jit}) ..."
    )
    res = 8
    boxsize = 100.0
    dtype = onp.float64 if precision == "double" else onp.float32
    jdtype = jnp.float64 if precision == "double" else jnp.float32
    key = jax.random.PRNGKey(7)
    delta = onp.array(0.05 * jax.random.normal(key, (res, res, res), dtype=jdtype), dtype=dtype)
    delta -= delta.mean(dtype=dtype)

    dj_ref = DiscoDJ(dim=3, res=res, precision=precision, boxsize=boxsize).with_timetables()
    dj_ref = dj_ref.with_external_ics(delta=delta)
    dj_ref = dj_ref.with_lpt(n_order=n_order, grad_kernel_order=grad_kernel_order, try_to_jit=try_to_jit)

    dj_dist = DiscoDJ(dim=3, res=res, precision=precision, boxsize=boxsize).with_timetables()
    dj_dist = dj_dist.with_external_ics(delta=delta, sharding=sharding_field)
    dj_dist = dj_dist.with_lpt(
        n_order=n_order,
        grad_kernel_order=grad_kernel_order,
        sharding=sharding_disp,
        try_to_jit=try_to_jit,
    )

    if precision == "double":
        tolerances = {"psi_1": 2e-12, "psi_2": 2e-11, "psi_3": 2e-8}
        eval_tol = 2e-8
    else:
        tolerances = {"psi_1": 2e-5, "psi_2": 2e-5, "psi_3": 2e-5}
        eval_tol = 2e-5
    for key_name, tol in list(tolerances.items())[:n_order]:
        dist = jax.device_get(dj_dist._lpt.psi[key_name])
        ref = onp.asarray(dj_ref._lpt.psi[key_name])
        rel, max_abs = _relative_rmse(dist, ref)
        print(f"  {key_name}: rel_rmse={rel:.3e}, max_abs={max_abs:.3e}")
        if rel > tol:
            raise AssertionError(f"{key_name} mismatch: rel_rmse={rel:.3e} > {tol:.1e}")

    psi_ref = onp.asarray(dj_ref.evaluate_lpt_psi_at_a(1.0 / 32.0, n_order=n_order))
    psi_dist = jax.device_get(dj_dist.evaluate_lpt_psi_at_a(1.0 / 32.0, n_order=n_order))
    rel, max_abs = _relative_rmse(psi_dist, psi_ref)
    print(f"  evaluated psi: rel_rmse={rel:.3e}, max_abs={max_abs:.3e}")
    if rel > eval_tol:
        raise AssertionError(f"evaluated LPT mismatch: rel_rmse={rel:.3e}")

    if n_order == 2:
        a_eval = 1.0 / 32.0
        D_eval = float(dj_dist.cosmo.Dplus(a_eval))
        psi_fast, mom_fast = compute_2lpt_initial_state_distributed(
            dj_dist._ics["fphi_full"],
            res=res,
            boxsize=boxsize,
            Dplus=D_eval,
            grad_kernel_order=grad_kernel_order,
            dtype_num=dj_dist.dtype_num,
            dtype_c_num=dj_dist.dtype_c_num,
            no_factors=False,
            field_sharding=sharding_field,
            disp_sharding=sharding_disp,
            fft_sharding=getattr(dj_dist._ics["fphi_full"], "sharding", None),
        )
        psi_fast = jax.device_get(psi_fast)
        mom_fast = jax.device_get(mom_fast)
        mom_ref = onp.asarray(
            dj_ref._evaluate_lpt_property_at_a(
                a=a_eval,
                n_order=n_order,
                include_psi_0=False,
                D_derivative=True,
            )
        )
        rel, max_abs = _relative_rmse(psi_fast, psi_ref)
        print(f"  optimized 2LPT psi: rel_rmse={rel:.3e}, max_abs={max_abs:.3e}")
        if rel > eval_tol:
            raise AssertionError(f"optimized 2LPT psi mismatch: rel_rmse={rel:.3e}")
        rel, max_abs = _relative_rmse(mom_fast, mom_ref)
        print(f"  optimized 2LPT dpsi/dD: rel_rmse={rel:.3e}, max_abs={max_abs:.3e}")
        if rel > eval_tol:
            raise AssertionError(f"optimized 2LPT dpsi/dD mismatch: rel_rmse={rel:.3e}")


def _test_lpt_against_single(sharding_field, sharding_disp):
    _compare_lpt_against_single(
        sharding_field, sharding_disp, n_order=3, grad_kernel_order=0, try_to_jit=False
    )
    _compare_lpt_against_single(
        sharding_field, sharding_disp, n_order=2, grad_kernel_order=4, try_to_jit=True
    )
    _compare_lpt_against_single(
        sharding_field, sharding_disp, n_order=2, grad_kernel_order=4, try_to_jit=True,
        precision="single",
    )


def main():
    pdims, mesh = _make_mesh(_PDIMS)
    print(
        f"[test_distributed_lpt] devices available={jax.device_count()}, "
        f"using pdims={pdims}"
    )
    sharding_field = NamedSharding(mesh, P("x", "y", None))
    sharding_disp = NamedSharding(mesh, P("x", "y", None, None))

    _test_pad_crop_and_conv(sharding_field)
    _test_lpt_against_single(sharding_field, sharding_disp)
    print("[test_distributed_lpt] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Distributed full-complex 3D nLPT helpers.

This module mirrors the 3D algebra in :mod:`nlpt_3d_jax`, but keeps all
Fourier-space fields in ``jaxdecomp.pfft3d`` layout.  ``pfft3d`` maps physical
``(X, Y, Z)`` arrays to full-complex ``(Y, Z, X)`` arrays, so all k-space
helpers here treat the three Fourier axes symmetrically and use
``jaxdecomp.fftfreq3d`` to recover physical ``(k_X, k_Y, k_Z)`` kernels.
"""

from __future__ import annotations

from functools import partial
import time
from typing import Iterable

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.sharding import NamedSharding, PartitionSpec as P

import jaxdecomp

from ..core.distributed_pm import get_local_pdims, gradient_kernel_dist
from ..core.kernels import inv_laplace_kernel
from ..core.utils import set_0_to_val

__all__ = [
    "build_k_vecs_dist",
    "validate_distributed_lpt_setup",
    "pad_fourier_full",
    "crop_fourier_full",
    "conv2_fourier_distributed",
    "fmu2_sym_distributed",
    "fmu2_and_C_distributed",
    "fmu3_distributed",
    "compute_2lpt_initial_state_distributed",
    "compute_lpt_distributed",
]


def _with_sharding(x: Array, sharding: NamedSharding | None) -> Array:
    if sharding is None:
        return x
    return lax.with_sharding_constraint(x, sharding)


def _spec_with_component_axis(sharding: NamedSharding | None) -> NamedSharding | None:
    if sharding is None:
        return None
    return NamedSharding(sharding.mesh, P(*tuple(sharding.spec), None))


def _fourier_blocks(orig_res: int, ext_res: int):
    """Return source and destination low/negative Fourier blocks.

    The Nyquist plane at ``orig_res // 2`` is intentionally dropped, matching
    DISCO-DJ's existing dealiased padding convention.
    """
    if ext_res < orig_res:
        raise ValueError(f"ext_res={ext_res} must be >= orig_res={orig_res}.")
    if orig_res % 2 != 0:
        raise ValueError("Full-complex distributed nLPT requires an even resolution.")
    if ext_res == orig_res:
        blocks = ((0, orig_res),)
        return blocks, blocks

    half = orig_res // 2
    src = ((0, half), (half + 1, orig_res))
    dst = ((0, half), (ext_res - half + 1, ext_res))
    return src, dst


def _set_blocks(out: Array, src_arr: Array, dst_blocks, src_blocks) -> Array:
    for bx, (dx0, dx1) in enumerate(dst_blocks):
        sx0, sx1 = src_blocks[bx]
        for by, (dy0, dy1) in enumerate(dst_blocks):
            sy0, sy1 = src_blocks[by]
            for bz, (dz0, dz1) in enumerate(dst_blocks):
                sz0, sz1 = src_blocks[bz]
                out = out.at[dx0:dx1, dy0:dy1, dz0:dz1].set(
                    src_arr[sx0:sx1, sy0:sy1, sz0:sz1]
                )
    return out


def validate_distributed_lpt_setup(
    *,
    res: int,
    ext_res: int,
    field_sharding: NamedSharding,
    disp_sharding: NamedSharding,
):
    """Validate the shape/sharding assumptions used by distributed nLPT."""
    field_spec = tuple(field_sharding.spec)
    disp_spec = tuple(disp_sharding.spec)
    if len(field_spec) != 3:
        raise ValueError(
            f"field_sharding must be rank-3 P('x','y',None); got {field_sharding.spec}."
        )
    if len(disp_spec) != 4:
        raise ValueError(
            f"disp_sharding must be rank-4 P('x','y',None,None); got {disp_sharding.spec}."
        )
    if field_sharding.mesh is not disp_sharding.mesh:
        raise ValueError("field_sharding and disp_sharding must use the same Mesh object.")

    px, py = get_local_pdims(field_sharding)
    if res % px != 0 or res % py != 0:
        raise ValueError(f"res={res} must be divisible by pdims={(px, py)}.")
    if ext_res % px != 0 or ext_res % py != 0:
        raise ValueError(
            f"ext_res={ext_res} must be divisible by pdims={(px, py)}. "
            "Choose a resolution whose 3/2 dealiased grid partitions cleanly."
        )


def build_k_vecs_dist(fk: Array, *, boxsize: float, res: int):
    """Build angular-frequency k-vectors for a ``pfft3d`` output array."""
    cell_size = boxsize / res
    k_X, k_Y, k_Z = jaxdecomp.fftfreq3d(fk, d=cell_size)
    dtype = jnp.float32 if fk.dtype == jnp.complex64 else jnp.float64
    return [k_X.astype(dtype), k_Y.astype(dtype), k_Z.astype(dtype)]


def pad_fourier_full(
    ff: Array,
    *,
    orig_res: int,
    ext_res: int,
    dtype_c,
    fft_sharding: NamedSharding | None = None,
) -> Array:
    """Pad a full-complex ``pfft3d``-layout field to ``ext_res``."""
    ff = _with_sharding(ff, fft_sharding)
    if ext_res == orig_res:
        return ff

    src_blocks, dst_blocks = _fourier_blocks(orig_res, ext_res)
    out = jnp.zeros((ext_res, ext_res, ext_res), dtype=dtype_c)
    out = _with_sharding(out, fft_sharding)
    out = _set_blocks(out, ff, dst_blocks, src_blocks)
    out = set_0_to_val(3, out, 0.0)
    out = out * (ext_res / orig_res) ** 3
    return _with_sharding(out, fft_sharding)


def crop_fourier_full(
    ff: Array,
    *,
    orig_res: int,
    ext_res: int,
    dtype_c,
    fft_sharding: NamedSharding | None = None,
) -> Array:
    """Crop a full-complex ``pfft3d``-layout field from ``ext_res`` to ``orig_res``."""
    ff = _with_sharding(ff, fft_sharding)
    if ext_res == orig_res:
        return ff

    dst_blocks, src_blocks = _fourier_blocks(orig_res, ext_res)
    out = jnp.zeros((orig_res, orig_res, orig_res), dtype=dtype_c)
    out = _with_sharding(out, fft_sharding)
    out = _set_blocks(out, ff, dst_blocks, src_blocks)
    out = out * (orig_res / ext_res) ** 3
    return _with_sharding(out, fft_sharding)


def conv2_fourier_distributed(
    ff1: Array,
    ff2: Array,
    *,
    orig_res: int | None,
    ext_res: int,
    dtype_c,
    field_sharding: NamedSharding,
    fft_sharding: NamedSharding | None,
    fft_backend: str = "JAX",
    do_crop: bool = True,
) -> Array:
    """Convolve two full-complex Fourier fields using distributed FFTs."""
    ff1 = _with_sharding(ff1, fft_sharding)
    ff2 = _with_sharding(ff2, fft_sharding)

    real1 = jaxdecomp.pifft3d(ff1, norm="backward", backend=fft_backend).real
    real2 = jaxdecomp.pifft3d(ff2, norm="backward", backend=fft_backend).real
    real1 = _with_sharding(real1, field_sharding)
    real2 = _with_sharding(real2, field_sharding)

    product = _with_sharding((real1 * real2).astype(dtype_c), field_sharding)
    out = jaxdecomp.pfft3d(product, norm="backward", backend=fft_backend)
    out = _with_sharding(out, fft_sharding)
    if do_crop:
        if orig_res is None:
            raise ValueError("orig_res must be provided when do_crop=True.")
        out = crop_fourier_full(
            out,
            orig_res=orig_res,
            ext_res=ext_res,
            dtype_c=dtype_c,
            fft_sharding=fft_sharding,
        )
    return out


def _deriv_component(
    field: Array,
    *,
    component: int,
    axis: int,
    orig_res: int,
    ext_res: int,
    derivs_ext,
    dtype_c,
    fft_sharding: NamedSharding | None,
) -> Array:
    padded = pad_fourier_full(
        field[..., component],
        orig_res=orig_res,
        ext_res=ext_res,
        dtype_c=dtype_c,
        fft_sharding=fft_sharding,
    )
    return _with_sharding(derivs_ext[axis] * padded, fft_sharding)


def _accumulate_terms(
    init: Array,
    terms: Iterable[tuple[int, int, int, int, int]],
    field_a: Array,
    field_b: Array,
    *,
    orig_res: int,
    ext_res: int,
    derivs_ext,
    dtype_c,
    field_sharding: NamedSharding,
    fft_sharding: NamedSharding | None,
    fft_backend: str,
) -> Array:
    out = init
    for comp_a, comp_b, axis_a, axis_b, sign in terms:
        term_a = _deriv_component(
            field_a,
            component=comp_a,
            axis=axis_a,
            orig_res=orig_res,
            ext_res=ext_res,
            derivs_ext=derivs_ext,
            dtype_c=dtype_c,
            fft_sharding=fft_sharding,
        )
        term_b = _deriv_component(
            field_b,
            component=comp_b,
            axis=axis_b,
            orig_res=orig_res,
            ext_res=ext_res,
            derivs_ext=derivs_ext,
            dtype_c=dtype_c,
            fft_sharding=fft_sharding,
        )
        conv = conv2_fourier_distributed(
            term_a,
            term_b,
            orig_res=orig_res,
            ext_res=ext_res,
            dtype_c=dtype_c,
            field_sharding=field_sharding,
            fft_sharding=fft_sharding,
            fft_backend=fft_backend,
        )
        out = out + jnp.asarray(sign, dtype=dtype_c) * conv
    return _with_sharding(out, fft_sharding)


def fmu2_sym_distributed(
    f1: Array,
    *,
    orig_res: int,
    ext_res: int,
    derivs_ext,
    dtype_c,
    field_sharding: NamedSharding,
    fft_sharding: NamedSharding | None,
    fft_backend: str = "JAX",
) -> Array:
    """Compute the symmetric ``mu2`` term for ``j == i - j``."""
    terms = (
        (0, 1, 0, 1, 1),
        (0, 2, 0, 2, 1),
        (1, 2, 1, 2, 1),
        (0, 1, 1, 0, -1),
        (0, 2, 2, 0, -1),
        (1, 2, 2, 1, -1),
    )
    out = jnp.zeros((orig_res, orig_res, orig_res), dtype=dtype_c)
    out = _with_sharding(out, fft_sharding)
    return _accumulate_terms(
        out,
        terms,
        f1,
        f1,
        orig_res=orig_res,
        ext_res=ext_res,
        derivs_ext=derivs_ext,
        dtype_c=dtype_c,
        field_sharding=field_sharding,
        fft_sharding=fft_sharding,
        fft_backend=fft_backend,
    )


def fmu2_and_C_distributed(
    f1: Array,
    f2: Array,
    *,
    orig_res: int,
    ext_res: int,
    derivs_ext,
    dtype_c,
    field_sharding: NamedSharding,
    fft_sharding: NamedSharding | None,
    fft_backend: str = "JAX",
) -> tuple[Array, Array]:
    """Compute asymmetric ``mu2`` and transverse ``C`` terms."""
    mu2_terms = (
        (0, 1, 0, 1, 1),
        (0, 2, 0, 2, 1),
        (1, 0, 0, 1, -1),
        (2, 0, 0, 2, -1),
        (1, 2, 1, 2, 1),
        (1, 0, 1, 0, 1),
        (2, 1, 1, 2, -1),
        (0, 1, 1, 0, -1),
        (2, 0, 2, 0, 1),
        (2, 1, 2, 1, 1),
        (0, 2, 2, 0, -1),
        (1, 2, 2, 1, -1),
    )
    scalar_shape = (orig_res, orig_res, orig_res)
    mu2 = _with_sharding(jnp.zeros(scalar_shape, dtype=dtype_c), fft_sharding)
    mu2 = _accumulate_terms(
        mu2,
        mu2_terms,
        f1,
        f2,
        orig_res=orig_res,
        ext_res=ext_res,
        derivs_ext=derivs_ext,
        dtype_c=dtype_c,
        field_sharding=field_sharding,
        fft_sharding=fft_sharding,
        fft_backend=fft_backend,
    )

    c_base = (
        (0, 0, 1, 2, 1),
        (0, 0, 2, 1, -1),
        (1, 1, 1, 2, 1),
        (1, 1, 2, 1, -1),
        (2, 2, 1, 2, 1),
        (2, 2, 2, 1, -1),
    )
    c_components = []
    for shift in range(3):
        shifted = tuple(
            (
                (comp_a + shift) % 3,
                (comp_b + shift) % 3,
                (axis_a + shift) % 3,
                (axis_b + shift) % 3,
                sign,
            )
            for comp_a, comp_b, axis_a, axis_b, sign in c_base
        )
        c_i = _with_sharding(jnp.zeros(scalar_shape, dtype=dtype_c), fft_sharding)
        c_i = _accumulate_terms(
            c_i,
            shifted,
            f1,
            f2,
            orig_res=orig_res,
            ext_res=ext_res,
            derivs_ext=derivs_ext,
            dtype_c=dtype_c,
            field_sharding=field_sharding,
            fft_sharding=fft_sharding,
            fft_backend=fft_backend,
        )
        c_components.append(c_i)

    c_sharding = _spec_with_component_axis(fft_sharding)
    C = _with_sharding(jnp.stack(c_components, axis=-1), c_sharding)
    return mu2, C


def fmu3_distributed(
    f1: Array,
    f2: Array,
    f3: Array,
    *,
    orig_res: int,
    ext_res: int,
    derivs_ext,
    dtype_c,
    field_sharding: NamedSharding,
    fft_sharding: NamedSharding | None,
    fft_backend: str = "JAX",
) -> Array:
    """Compute the 3D ``mu3`` term."""
    terms = (
        (0, 1, 2, 1),
        (0, 2, 1, -1),
        (1, 2, 0, 1),
        (1, 0, 2, -1),
        (2, 0, 1, 1),
        (2, 1, 0, -1),
    )
    out = _with_sharding(jnp.zeros((orig_res, orig_res, orig_res), dtype=dtype_c), fft_sharding)
    for axis_a, axis_b, axis_c, sign in terms:
        term_a = _deriv_component(
            f1,
            component=0,
            axis=axis_a,
            orig_res=orig_res,
            ext_res=ext_res,
            derivs_ext=derivs_ext,
            dtype_c=dtype_c,
            fft_sharding=fft_sharding,
        )
        term_b = _deriv_component(
            f2,
            component=1,
            axis=axis_b,
            orig_res=orig_res,
            ext_res=ext_res,
            derivs_ext=derivs_ext,
            dtype_c=dtype_c,
            fft_sharding=fft_sharding,
        )
        term_c = _deriv_component(
            f3,
            component=2,
            axis=axis_c,
            orig_res=orig_res,
            ext_res=ext_res,
            derivs_ext=derivs_ext,
            dtype_c=dtype_c,
            fft_sharding=fft_sharding,
        )
        inner = conv2_fourier_distributed(
            term_b,
            term_c,
            orig_res=None,
            ext_res=ext_res,
            dtype_c=dtype_c,
            field_sharding=field_sharding,
            fft_sharding=fft_sharding,
            fft_backend=fft_backend,
            do_crop=False,
        )
        conv = conv2_fourier_distributed(
            term_a,
            inner,
            orig_res=orig_res,
            ext_res=ext_res,
            dtype_c=dtype_c,
            field_sharding=field_sharding,
            fft_sharding=fft_sharding,
            fft_backend=fft_backend,
        )
        out = out + jnp.asarray(sign, dtype=dtype_c) * conv
    return _with_sharding(out, fft_sharding)


def _real_psi_from_fourier(
    psi_k_orders: list[Array],
    *,
    dtype,
    field_sharding: NamedSharding,
    disp_sharding: NamedSharding,
    fft_sharding: NamedSharding | None,
    fft_backend: str,
) -> dict[str, Array]:
    out = {}
    for n, psi_k in enumerate(psi_k_orders, start=1):
        components = []
        for axis in range(3):
            comp_k = _with_sharding(psi_k[..., axis], fft_sharding)
            comp = jaxdecomp.pifft3d(comp_k, norm="backward", backend=fft_backend).real.astype(dtype)
            components.append(_with_sharding(comp, field_sharding))
        out[f"psi_{n}"] = _with_sharding(jnp.stack(components, axis=-1), disp_sharding)
    return out


def compute_2lpt_initial_state_distributed(
    fphi_ini: Array,
    *,
    res: int,
    boxsize: float,
    Dplus: float | Array,
    grad_kernel_order: int,
    dtype_num: int,
    dtype_c_num: int,
    no_factors: bool,
    field_sharding: NamedSharding,
    disp_sharding: NamedSharding,
    fft_sharding: NamedSharding | None = None,
    fft_backend: str = "JAX",
    mu2_fft_backend: str | None = None,
    progress: bool = False,
    progress_prefix: str = "",
    sync_progress: bool = True,
) -> tuple[Array, Array]:
    """Compute 2LPT ``psi`` and ``dpsi/dD`` directly with low peak memory.

    This is a production-oriented specialization of ``compute_lpt_distributed``
    for the common ``n_order=2`` IC case.  It preserves the same EdS 2LPT
    factor, 3/2 dealiased convolution, and distributed full-complex FFT layout,
    but avoids materializing persistent ``psi_1``/``psi_2`` dictionaries or a
    dealiased complex vector field.
    """
    ext_res = 3 * res // 2
    validate_distributed_lpt_setup(
        res=res,
        ext_res=ext_res,
        field_sharding=field_sharding,
        disp_sharding=disp_sharding,
    )
    if fft_sharding is None:
        fft_sharding = getattr(fphi_ini, "sharding", None)

    dtype = jnp.float64 if dtype_num == 64 else jnp.float32
    dtype_c = jnp.complex128 if dtype_c_num == 128 else jnp.complex64
    if mu2_fft_backend is None:
        mu2_fft_backend = "JAX" if fft_backend.lower() == "cudecomp" else fft_backend
    if mu2_fft_backend.lower() not in {"jax", "cudecomp"}:
        raise ValueError(f"mu2_fft_backend must be 'JAX' or 'cudecomp', got {mu2_fft_backend!r}.")
    mu2_fft_backend = "cudecomp" if mu2_fft_backend.lower() == "cudecomp" else "JAX"

    def _progress(message: str, arr: Array | None = None) -> None:
        if not progress:
            return
        if sync_progress and arr is not None:
            arr.block_until_ready()
        prefix = f"{progress_prefix} " if progress_prefix else ""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {prefix}{message}", flush=True)

    _progress(
        "2LPT setup: "
        f"res={res}, ext_res={ext_res}, dtype={dtype}, dtype_c={dtype_c}, "
        f"fft_backend={fft_backend}, mu2_fft_backend={mu2_fft_backend}"
    )

    # The input is expected to come from the distributed fphi builder or
    # with_external_ics, both of which zero the DC mode. Avoid another
    # full-field prepared copy here; it materially raises the 2LPT peak.
    fphi = _with_sharding(fphi_ini, fft_sharding)
    if fphi.dtype != dtype_c:
        fphi = _with_sharding(fphi.astype(dtype_c), fft_sharding)

    k_vecs = build_k_vecs_dist(fphi, boxsize=boxsize, res=res)
    derivs = tuple(gradient_kernel_dist(k_vecs, axis=d, order=grad_kernel_order) for d in range(3))

    @jax.jit
    def build_inv_lap(template):
        local_k_vecs = build_k_vecs_dist(template, boxsize=boxsize, res=res)
        ksquare = sum(ki ** 2 for ki in local_k_vecs)
        mask = (ksquare != 0).astype(ksquare.dtype)
        ksquare = set_0_to_val(3, ksquare, 1.0)
        kernel = -1.0 / ksquare
        kernel = kernel * mask
        return _with_sharding(kernel, fft_sharding)

    D = jnp.asarray(Dplus, dtype=dtype)
    D2 = D * D

    def hessian_real_ext(component: int, axis: int) -> Array:
        @jax.jit
        def _compute(phi, deriv_component):
            # Matches _deriv_component(psi_1, component, axis):
            # psi_1_component(k) = -grad_component(k) * phi(k).
            psi1_component = _with_sharding(-deriv_component * phi, fft_sharding)
            padded = pad_fourier_full(
                psi1_component,
                orig_res=res,
                ext_res=ext_res,
                dtype_c=dtype_c,
                fft_sharding=fft_sharding,
            )
            k_vecs_ext = build_k_vecs_dist(padded, boxsize=boxsize, res=ext_res)
            deriv_ext = gradient_kernel_dist(k_vecs_ext, axis=axis, order=grad_kernel_order)
            hess_k = _with_sharding(deriv_ext * padded, fft_sharding)
            hess = jaxdecomp.pifft3d(hess_k, norm="backward", backend=fft_backend).real.astype(dtype)
            return _with_sharding(hess, field_sharding)

        return _compute(fphi, derivs[component])

    @partial(jax.jit, donate_argnums=(0,))
    def accumulate_product(acc, term_a, term_b, sign):
        out = acc + sign * term_a * term_b
        return _with_sharding(out.astype(dtype), field_sharding)

    @jax.jit
    def first_product(term_a, term_b, sign):
        return _with_sharding((sign * term_a * term_b).astype(dtype), field_sharding)

    # Same six terms as fmu2_sym_distributed, accumulated in real space so the
    # expensive forward FFT/crop is performed once by linearity.
    terms = (
        (0, 1, 0, 1, 1),
        (0, 2, 0, 2, 1),
        (1, 2, 1, 2, 1),
        (0, 1, 1, 0, -1),
        (0, 2, 2, 0, -1),
        (1, 2, 2, 1, -1),
    )
    mu2_real_ext = None
    for term_idx, (comp_a, comp_b, axis_a, axis_b, sign) in enumerate(terms, start=1):
        _progress(
            f"2LPT mu2 term {term_idx}/6: "
            f"hessian psi[{comp_a}],d{axis_a} start"
        )
        term_a = hessian_real_ext(comp_a, axis_a)
        _progress(
            f"2LPT mu2 term {term_idx}/6: "
            f"hessian psi[{comp_a}],d{axis_a} done",
            term_a,
        )
        _progress(
            f"2LPT mu2 term {term_idx}/6: "
            f"hessian psi[{comp_b}],d{axis_b} start"
        )
        term_b = hessian_real_ext(comp_b, axis_b)
        _progress(
            f"2LPT mu2 term {term_idx}/6: "
            f"hessian psi[{comp_b}],d{axis_b} done",
            term_b,
        )
        sign = jnp.asarray(sign, dtype=dtype)
        _progress(f"2LPT mu2 term {term_idx}/6: product accumulation start")
        if mu2_real_ext is None:
            mu2_real_ext = first_product(term_a, term_b, sign)
        else:
            mu2_real_ext = accumulate_product(mu2_real_ext, term_a, term_b, sign)
        _progress(f"2LPT mu2 term {term_idx}/6: product accumulation done", mu2_real_ext)
        del term_a, term_b

    @partial(jax.jit, donate_argnums=(0,))
    def mu2_real_to_fL(mu2_real):
        mu2_complex = _with_sharding(mu2_real.astype(dtype_c), field_sharding)
        # The dealiased forward FFT is the peak-workspace step. cuDecomp's
        # pencil forward plan can require an H100-sized scratch buffer at
        # 2304^3, while JAX's distributed FFT avoids that cuFFT workspace.
        mu2_k_ext = jaxdecomp.pfft3d(mu2_complex, norm="backward", backend=mu2_fft_backend)
        mu2_k_ext = _with_sharding(mu2_k_ext, fft_sharding)
        mu2_k = crop_fourier_full(
            mu2_k_ext,
            orig_res=res,
            ext_res=ext_res,
            dtype_c=dtype_c,
            fft_sharding=fft_sharding,
        )
        if no_factors:
            fac_sym = jnp.asarray(1.0, dtype=dtype_c)
        else:
            fac_sym = jnp.asarray(-3.0 / 7.0, dtype=dtype_c)
        return _with_sharding(fac_sym * mu2_k, fft_sharding)

    _progress("2LPT mu2 real-to-Fourier/crop start")
    fL2 = mu2_real_to_fL(mu2_real_ext)
    _progress("2LPT mu2 real-to-Fourier/crop done", fL2)
    del mu2_real_ext
    _progress("2LPT inverse-Laplace kernel build start")
    inv_lap = build_inv_lap(fphi)
    _progress("2LPT inverse-Laplace kernel build done", inv_lap)

    def lpt_component(axis: int) -> tuple[Array, Array]:
        @jax.jit
        def _compute(phi, fL, deriv_axis, inv_lap_kernel):
            psi1_k = _with_sharding(-deriv_axis * phi, fft_sharding)
            psi2_k = _with_sharding(inv_lap_kernel * (deriv_axis * fL), fft_sharding)
            psi1 = jaxdecomp.pifft3d(psi1_k, norm="backward", backend=fft_backend).real.astype(dtype)
            psi2 = jaxdecomp.pifft3d(psi2_k, norm="backward", backend=fft_backend).real.astype(dtype)
            psi = D * psi1 + D2 * psi2
            mom = psi1 + (2.0 * D) * psi2
            return _with_sharding(psi, field_sharding), _with_sharding(mom, field_sharding)

        return _compute(fphi, fL2, derivs[axis], inv_lap)

    psi_components = []
    mom_components = []
    for axis in range(3):
        _progress(f"2LPT final component axis={axis}: inverse FFTs start")
        psi_axis, mom_axis = lpt_component(axis)
        _progress(f"2LPT final component axis={axis}: psi done", psi_axis)
        _progress(f"2LPT final component axis={axis}: momentum done", mom_axis)
        psi_components.append(psi_axis)
        mom_components.append(mom_axis)

    # These Fourier-space inputs and kernels are no longer needed once all
    # component fields have been materialized. Drop Python references before
    # stacking to lower the peak live buffer set in the final 2LPT stage.
    del fphi, fL2, inv_lap, derivs

    @jax.jit
    def stack_components(c0, c1, c2):
        return _with_sharding(jnp.stack([c0, c1, c2], axis=-1), disp_sharding)

    _progress("2LPT stack psi components start")
    psi = stack_components(*psi_components)
    _progress("2LPT stack psi components done", psi)
    del psi_components
    _progress("2LPT stack momentum components start")
    mom = stack_components(*mom_components)
    _progress("2LPT stack momentum components done", mom)
    del mom_components
    _progress("2LPT initialization complete")
    return psi, mom


def compute_lpt_distributed(
    fphi_ini: Array,
    *,
    res: int,
    boxsize: float,
    n_order: int,
    grad_kernel_order: int,
    dtype_num: int,
    dtype_c_num: int,
    no_transverse: bool,
    no_factors: bool,
    field_sharding: NamedSharding,
    disp_sharding: NamedSharding,
    fft_sharding: NamedSharding | None,
    fft_backend: str = "JAX",
) -> dict[str, Array]:
    """Compute 1/2/3LPT in distributed full-complex Fourier space."""
    if n_order < 1:
        raise ValueError("n_order must be >= 1.")
    if n_order > 3:
        raise NotImplementedError("Distributed nLPT currently supports n_order <= 3.")

    ext_res = 3 * res // 2
    validate_distributed_lpt_setup(
        res=res,
        ext_res=ext_res,
        field_sharding=field_sharding,
        disp_sharding=disp_sharding,
    )

    dtype = jnp.float64 if dtype_num == 64 else jnp.float32
    dtype_c = jnp.complex128 if dtype_c_num == 128 else jnp.complex64
    vector_fft_sharding = _spec_with_component_axis(fft_sharding)

    fphi = _with_sharding(fphi_ini.astype(dtype_c), fft_sharding)
    fphi = set_0_to_val(3, fphi, 0.0)

    k_vecs = build_k_vecs_dist(fphi, boxsize=boxsize, res=res)
    derivs = tuple(gradient_kernel_dist(k_vecs, axis=d, order=grad_kernel_order) for d in range(3))
    inv_lap = inv_laplace_kernel(k_vecs, with_jax=True)

    fphi_ext = pad_fourier_full(
        fphi,
        orig_res=res,
        ext_res=ext_res,
        dtype_c=dtype_c,
        fft_sharding=fft_sharding,
    )
    k_vecs_ext = build_k_vecs_dist(fphi_ext, boxsize=boxsize, res=ext_res)
    derivs_ext = tuple(gradient_kernel_dist(k_vecs_ext, axis=d, order=grad_kernel_order) for d in range(3))

    psi_orders = [
        _with_sharding(
            -jnp.stack([deriv * fphi for deriv in derivs], axis=-1),
            vector_fft_sharding,
        )
    ]

    scalar_shape = (res, res, res)
    for order in range(2, n_order + 1):
        fL = _with_sharding(jnp.zeros(scalar_shape, dtype=dtype_c), fft_sharding)
        fT = _with_sharding(jnp.zeros(scalar_shape + (3,), dtype=dtype_c), vector_fft_sharding)

        if order % 2 == 0:
            if no_factors:
                fac_sym = jnp.asarray(1.0, dtype=dtype_c)
            else:
                j_mid = order // 2
                fac_sym = jnp.asarray(
                    ((3 - order) / 2 - j_mid**2 - j_mid**2) / ((order + 3 / 2) * (order - 1)),
                    dtype=dtype_c,
                )
            fL = fL + fac_sym * fmu2_sym_distributed(
                psi_orders[order // 2 - 1],
                orig_res=res,
                ext_res=ext_res,
                derivs_ext=derivs_ext,
                dtype_c=dtype_c,
                field_sharding=field_sharding,
                fft_sharding=fft_sharding,
                fft_backend=fft_backend,
            )

        if order > 2:
            c_multiplier = jnp.asarray(0.0 if no_transverse else 1.0, dtype=dtype_c)
            for j_order in range(1, (order + 1) // 2):
                if no_factors:
                    fac_mu2 = jnp.asarray(1.0, dtype=dtype_c)
                    fac_C = jnp.asarray(1.0, dtype=dtype_c)
                else:
                    fac_mu2 = jnp.asarray(
                        ((3 - order) / 2 - j_order**2 - (order - j_order) ** 2)
                        / ((order + 3 / 2) * (order - 1)),
                        dtype=dtype_c,
                    )
                    fac_C = jnp.asarray(1.0 - 2 * j_order / order, dtype=dtype_c)
                mu2, C = fmu2_and_C_distributed(
                    psi_orders[j_order - 1],
                    psi_orders[order - j_order - 1],
                    orig_res=res,
                    ext_res=ext_res,
                    derivs_ext=derivs_ext,
                    dtype_c=dtype_c,
                    field_sharding=field_sharding,
                    fft_sharding=fft_sharding,
                    fft_backend=fft_backend,
                )
                fL = fL + fac_mu2 * mu2
                fT = fT + fac_C * c_multiplier * C

            for k_order in range(1, order - 1):
                for l_order in range(1, order - k_order):
                    if no_factors:
                        fac_mu3 = jnp.asarray(1.0, dtype=dtype_c)
                    else:
                        fac_mu3 = jnp.asarray(
                            (
                                (3 - order) / 2
                                - k_order**2
                                - l_order**2
                                - (order - k_order - l_order) ** 2
                            )
                            / ((order + 3 / 2) * (order - 1)),
                            dtype=dtype_c,
                        )
                    fL = fL + fac_mu3 * fmu3_distributed(
                        psi_orders[k_order - 1],
                        psi_orders[l_order - 1],
                        psi_orders[order - k_order - l_order - 1],
                        orig_res=res,
                        ext_res=ext_res,
                        derivs_ext=derivs_ext,
                        dtype_c=dtype_c,
                        field_sharding=field_sharding,
                        fft_sharding=fft_sharding,
                        fft_backend=fft_backend,
                    )

        d_dx, d_dy, d_dz = derivs
        psi_i_x = inv_lap * (d_dx * fL - (d_dy * fT[..., 2] - d_dz * fT[..., 1]))
        psi_i_y = inv_lap * (d_dy * fL - (d_dz * fT[..., 0] - d_dx * fT[..., 2]))
        psi_i_z = inv_lap * (d_dz * fL - (d_dx * fT[..., 1] - d_dy * fT[..., 0]))
        psi_orders.append(
            _with_sharding(jnp.stack([psi_i_x, psi_i_y, psi_i_z], axis=-1), vector_fft_sharding)
        )

    return _real_psi_from_fourier(
        psi_orders,
        dtype=dtype,
        field_sharding=field_sharding,
        disp_sharding=disp_sharding,
        fft_sharding=fft_sharding,
        fft_backend=fft_backend,
    )

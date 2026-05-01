"""Reproducer: single-GPU vs multi-GPU PM force comparison with factor=2.

Runs on 2 emulated CPU "devices" so we can debug locally. Compares:
  - single-GPU `calc_acc_PM` (rfftn + scatter/gather)
  - distributed `calc_acc_PM_distributed` (pfft3d + halo exchange)

with `res_part=8`, `res_pm=16` (factor=2). Tiny problem so we can iterate
quickly. Uses fp64 for tight tolerance.

Goal: figure out why the multi-GPU code under-evolves at factor=2 in the
notebook (single-GPU at full precision should agree with multi-GPU to ~1e-12).
"""
from __future__ import annotations
import os, sys

# Force 2 emulated CPU devices BEFORE importing JAX.
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as onp
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import jaxdecomp

# Bypass discodj_dist/__init__.py (needs einops + native deps); import directly
import importlib.util
SRC = '/mnt/ceph/users/spandey/quijote_v2_gotham/DISCO-DJ/src'

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Need to inject empty stubs for the discodj_dist packages so the relative
# imports inside these modules work.
import types
discodj_dist = types.ModuleType('discodj_dist')
discodj_dist.__path__ = [SRC + '/discodj_dist']
sys.modules['discodj_dist'] = discodj_dist
for sub in ['core', 'nbody']:
    m = types.ModuleType(f'discodj_dist.{sub}')
    m.__path__ = [SRC + f'/discodj_dist/{sub}']
    sys.modules[f'discodj_dist.{sub}'] = m

# Now load the actual modules we need
utils      = _load('discodj_dist.core.utils',                  SRC + '/discodj_dist/core/utils.py')
kernels    = _load('discodj_dist.core.kernels',                SRC + '/discodj_dist/core/kernels.py')
grids      = _load('discodj_dist.core.grids',                  SRC + '/discodj_dist/core/grids.py')
sg         = _load('discodj_dist.core.scatter_and_gather',     SRC + '/discodj_dist/core/scatter_and_gather.py')
dist_pm    = _load('discodj_dist.core.distributed_pm',         SRC + '/discodj_dist/core/distributed_pm.py')
acc_dist   = _load('discodj_dist.nbody.acc_distributed',       SRC + '/discodj_dist/nbody/acc_distributed.py')

# ----- Reproducer parameters --------------------------------------------------

res_part = int(os.environ.get('RES', 32))
factor   = int(os.environ.get('FACTOR', 2))
res_pm   = factor * res_part
boxsize  = 100.0
worder   = 2
dtype_num = int(os.environ.get('DTYPE_NUM', 64))
dtype = jnp.float64 if dtype_num == 64 else jnp.float32
seed = 0
# Maximum displacement in PM cells — set this high to exercise the halo.
MAX_DISP_PM = float(os.environ.get('MAX_DISP', 0.3))

cell_pm    = boxsize / res_pm
n_part_tot = res_part ** 3
print(f'res_part={res_part}, res_pm={res_pm}, factor={res_pm//res_part}, '
      f'cell_pm={cell_pm:.4f}, max_disp={MAX_DISP_PM} PM cells')

# ----- Random displacements --------------------------------------------------

key = jax.random.PRNGKey(seed)
psi_flat = MAX_DISP_PM * cell_pm * jax.random.normal(key, (n_part_tot, 3), dtype=dtype)
print(f'psi range: [{float(psi_flat.min()):.4e}, {float(psi_flat.max()):.4e}] Mpc/h')

# ----- SINGLE-GPU REFERENCE (factor=2) ---------------------------------------
# Re-implement calc_acc_PM directly here using sg.scatter / sg.gather and
# np.fft.rfftn — same as discodj_dist/nbody/acc.py:calc_acc_PM_

def calc_acc_PM_single(psi_flat, n_part, res_pm, boxsize, dtype_num,
                       worder=2, grad_order=0, lap_order=0):
    np = jnp
    dtype = np.float64 if dtype_num == 64 else np.float32

    # Build q (Lagrangian grid in Mpc/h)
    dx_for_q = boxsize / n_part
    q_vec = np.tile(np.arange(n_part) * dx_for_q, [3, 1]).astype(dtype)
    q = np.moveaxis(np.asarray(np.meshgrid(*q_vec, indexing='ij')), 0, -1)
    X = psi_flat + q.reshape(-1, 3)

    # Build k_vecs at PM resolution
    k_dict = grids.get_fourier_grid((res_pm,)*3, boxsize=boxsize,
                                    sparse_k_vecs=True, full=False,
                                    dtype_num=dtype_num, with_jax=True)
    k_vecs = k_dict['k_vecs']

    # Scatter
    delta = sg.scatter(np.zeros((res_pm,)*3, dtype=dtype), X,
                       n_part_tot=n_part**3, res=res_pm, boxsize=boxsize,
                       dtype_num=dtype_num, worder=worder, with_jax=True)

    # FFT
    fdelta_raw = np.fft.rfftn(delta)
    fdelta = utils.set_0_to_val(3, fdelta_raw, 0.0)

    # Poisson + grad + IFFT + gather, per axis
    inv_lap = kernels.inv_laplace_kernel(k_vecs, order=lap_order, with_jax=True)
    pot_k = inv_lap * fdelta

    accs = []
    for d in range(3):
        gradk = kernels.gradient_kernel(k_vecs, axis=d, order=grad_order,
                                        with_jax=True)
        # broadcast gradk over the field's full shape
        shape_for_x = [1, res_pm, res_pm//2 + 1]
        shape_for_y = [res_pm, 1, res_pm//2 + 1]
        shape_for_z = [res_pm, res_pm, 1]
        shapes = [shape_for_x, shape_for_y, shape_for_z]
        g = np.tile(gradk, shapes[d])
        facc = -g * pot_k
        acc_real = np.fft.irfftn(facc)
        acc_d = sg.gather(acc_real, X, n_part_tot=n_part**3, res=res_pm,
                          boxsize=boxsize, dtype_num=dtype_num, worder=worder,
                          with_jax=True)
        accs.append(acc_d.reshape(-1))

    return np.stack(accs, axis=-1)  # (N_part, 3)


print('\n=== Single-GPU reference (factor=2) ===')
acc_ref = onp.asarray(calc_acc_PM_single(psi_flat, res_part, res_pm, boxsize,
                                          dtype_num, worder=worder))
print(f'acc_ref.shape = {acc_ref.shape}')
print(f'|acc_ref|_max = {onp.abs(acc_ref).max():.6e}')
print(f'acc_ref[0]    = {acc_ref[0]}')


# ----- MULTI-GPU DISTRIBUTED (factor=2, pdims=(1,2)) -------------------------

n_dev = jax.device_count()
print(f'\n=== Distributed (jax.device_count()={n_dev}) ===')

pdims_str = os.environ.get('PDIMS', '1,2')
pdims = tuple(int(s) for s in pdims_str.split(','))
assert pdims[0] * pdims[1] == n_dev, f'pdims {pdims} != n_dev {n_dev}'
for _ in range(1):
    print(f'\n--- pdims = {pdims} ---')
    devices = mesh_utils.create_device_mesh(pdims)
    mesh = Mesh(devices, axis_names=('x', 'y'))
    sharding_disp = NamedSharding(mesh, P('x', 'y', None, None))

    # Halo size sized for the largest expected displacement.
    H_min = int(2 * (int(MAX_DISP_PM * 4) + worder // 2 + 4))
    halo_size = max(8, H_min + (H_min % 2))
    print(f'halo_size = {halo_size}')

    psi_grid = psi_flat.reshape(res_part, res_part, res_part, 3)
    psi_grid = jax.device_put(psi_grid, sharding_disp)

    @jax.jit
    def go(p):
        return acc_dist.calc_acc_PM_distributed(
            p, dim=3, res_pm=res_pm, boxsize=boxsize,
            halo_size=halo_size, sharding=sharding_disp,
            grad_order=0, lap_order=0, dtype_num=dtype_num, worder=worder,
        )
    acc_dist_grid = jax.device_get(go(psi_grid))
    acc_d = onp.asarray(acc_dist_grid).reshape(-1, 3)

    abs_err = onp.abs(acc_d - acc_ref)
    rel = abs_err.max() / max(onp.abs(acc_ref).max(), 1e-30)
    rmse = onp.sqrt((abs_err**2).mean())
    print(f'|acc_dist|_max = {onp.abs(acc_d).max():.6e}')
    print(f'acc_dist[0]    = {acc_d[0]}')
    print(f'acc_dist / acc_ref ratio at element 0: '
          f'{(acc_d[0] / (acc_ref[0] + 1e-30))}')
    print(f'max abs err = {abs_err.max():.3e}, rel = {rel:.3e}, rmse = {rmse:.3e}')

    # Find the WORST offending particles and report their (i,j,k) indices
    err_norm = onp.linalg.norm(acc_d - acc_ref, axis=-1)
    bad = onp.argsort(-err_norm)[:8]
    print(f'\nTop offending particles (worst err first):')
    print(f'  Particle layout: (i, j, k) at PARTICLE resolution {res_part}')
    for p in bad:
        i, j, k = onp.unravel_index(p, (res_part, res_part, res_part))
        d = psi_flat[p]
        print(f'  particle {p:6d} (i,j,k)=({i:2d},{j:2d},{k:2d})  '
              f'disp_pm=({float(d[0])/cell_pm:+.2f}, {float(d[1])/cell_pm:+.2f}, {float(d[2])/cell_pm:+.2f})  '
              f'acc_ref={acc_ref[p]}  acc_dist={acc_d[p]}  err={err_norm[p]:.3e}')

print('\nDone.')

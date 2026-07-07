#
# su2_hb_validate.py — validation gate for the Metal SU(2) heatbath generator.
#
# 1. layout:  GPU plaquette on a random gpt config == g.qcd.gauge.plaquette
# 2. staple:  GPU staple quaternions == gpt action.staple / (beta/2), float32
# 3. physics: GPU heatbath chain reproduces the CPU (gpt) chain's mean
#             plaquette on 8^4 at beta=2.4 within errors
#
# Usage: PYTHONPATH=gpt/lib/cgpt/build:gpt/lib python3.12 gpu-metal-wip/su2_hb_validate.py
#
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpt as g
from su2_hb_gpu import su2_gpu_generator, mat_to_quat

L = [8, 8, 8, 8]
beta = 2.4
grid = g.grid(L, g.single)
rng = g.random("su2-gpu-validate")

U = [g.lattice(grid, g.ot_matrix_su_n_fundamental_group(2)) for _ in range(4)]
rng.element(U)

gen = su2_gpu_generator(U, seed=424242)
g.message(f"Metal device: {gen.device_name}")

# ---- 1. layout / plaquette parity ----
gen.pack(U)
p_gpu = gen.plaquette()
p_gpt = float(g.qcd.gauge.plaquette(U))
err = abs(p_gpu - p_gpt)
g.message(f"[1] plaquette  gpu={p_gpu:.8f}  gpt={p_gpt:.8f}  |diff|={err:.2e}")
assert err < 5e-6, "plaquette parity FAILED (layout/convention bug)"

# round-trip pack/unpack must be lossless (single precision)
U2 = [g.copy(u) for u in U]
gen.unpack(U2)
rt = max(float(g.norm2(g.eval(U2[mu] - U[mu]))) for mu in range(4))
g.message(f"[1b] pack/unpack round-trip max norm2 diff = {rt:.2e}")
assert rt < 1e-10, "round-trip FAILED"

# ---- 2. staple parity ----
action = g.qcd.gauge.action.wilson(beta)
scale = beta / 2.0
max_err = 0.0
for mu in range(4):
    s_gpu = gen.staple_dump(mu)  # (nsites,4) raw staple sum
    s_gpt_lat = action.staple(U, mu)
    arr = s_gpt_lat[gen.coords].reshape(gen.nsites, 2, 2)
    s_gpt = np.zeros((gen.nsites, 4), dtype=np.float64)
    s_gpt[gen.idx, :] = mat_to_quat(arr).astype(np.float64) / scale
    e = float(np.max(np.abs(s_gpu.astype(np.float64) - s_gpt)))
    max_err = max(max_err, e)
    g.message(f"[2] staple mu={mu}  max|gpu-gpt| = {e:.2e}")
assert max_err < 2e-5, "staple parity FAILED"

# ---- 3. statistical physics test ----
ntherm = int(g.default.get_int("--ntherm", 150))
nmeas = int(g.default.get_int("--nmeas", 300))
run_cpu = g.default.get_int("--cpu", 1) != 0


def chain_stats(plaqs):
    a = np.array(plaqs, dtype=np.float64)
    n = len(a)
    mean = a.mean()
    # crude integrated-autocorrelation-aware error via binning
    nb = max(4, n // 20)
    bins = a[: (n // nb) * nb].reshape(-1, nb).mean(axis=1)
    err = bins.std(ddof=1) / np.sqrt(len(bins))
    return mean, err


# GPU chain
rng.element(U)  # fresh random start
gen.pack(U)
t0 = time.time()
gen.sweep_gpu_only(ntherm, beta)
plaqs_gpu = []
for i in range(nmeas):
    gen.sweep_gpu_only(1, beta)
    plaqs_gpu.append(gen.plaquette())
t_gpu = time.time() - t0
m_gpu, e_gpu = chain_stats(plaqs_gpu)
sw_gpu = (ntherm + nmeas) / t_gpu
g.message(
    f"[3] GPU chain 8^4 beta={beta}: <P>={m_gpu:.6f} +/- {e_gpu:.6f} "
    f"({ntherm}+{nmeas} sweeps in {t_gpu:.1f}s = {sw_gpu:.1f} sweeps/s)"
)

# inline CPU reference chain (mirrors the driver's one_sweep)
if run_cpu:
    hb = g.algorithms.markov.su2_heat_bath(rng)
    mask_rb = g.complex(grid.checkerboarded(g.redblack))
    mask_rb[:] = 1
    mask = g.complex(grid)

    def one_sweep_ref(Uf):
        for cb in [g.even, g.odd]:
            mask[:] = 0
            mask_rb.checkerboard(cb)
            g.set_checkerboard(mask, mask_rb)
            for mu in range(4):
                hb(Uf[mu], action.staple(Uf, mu), mask)

    rng.element(U)
    t0 = time.time()
    plaqs_cpu = []
    ntherm_cpu = min(ntherm, 100)
    nmeas_cpu = min(nmeas, 200)
    for i in range(ntherm_cpu):
        one_sweep_ref(U)
    for i in range(nmeas_cpu):
        one_sweep_ref(U)
        plaqs_cpu.append(float(g.qcd.gauge.plaquette(U)))
    t_cpu = time.time() - t0
    m_cpu, e_cpu = chain_stats(plaqs_cpu)
    sw_cpu = (ntherm_cpu + nmeas_cpu) / t_cpu
    g.message(
        f"[3] CPU chain 8^4 beta={beta}: <P>={m_cpu:.6f} +/- {e_cpu:.6f} "
        f"({ntherm_cpu}+{nmeas_cpu} sweeps in {t_cpu:.1f}s = {sw_cpu:.2f} sweeps/s)"
    )
    sigma = abs(m_gpu - m_cpu) / max(1e-12, np.hypot(e_gpu, e_cpu))
    g.message(f"[3] GPU vs CPU: |dP| = {abs(m_gpu-m_cpu):.6f} = {sigma:.2f} sigma")
    g.message(f"[3] speedup at 8^4: {sw_gpu/sw_cpu:.0f}x sweeps/s (GPU vs gpt CPU)")
    assert sigma < 4.0, "statistical plaquette parity FAILED (>4 sigma)"

g.message("ALL VALIDATIONS PASSED")

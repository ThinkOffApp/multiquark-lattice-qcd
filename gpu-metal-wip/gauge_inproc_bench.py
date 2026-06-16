#!/usr/bin/env python3
"""In-process GPU gauge measurement: correctness + timing vs CPU.

Loads libcwgauge.dylib (Swift Metal dispatcher) into gpt's own python process
via ctypes (no subprocess, no /tmp file roundtrip), computes the full 6-plane
SU(3) plaquette on the GPU in-process, checks it against numpy double and
gpt-native g.qcd.gauge.plaquette, and times GPU vs CPU on the same Cshift-aligned
fields.

Run: PYTHONPATH=gpt/lib/cgpt/build:gpt/lib python3.12 gauge_inproc_bench.py
"""
import os, ctypes, time, numpy as np, gpt as g

HERE = os.path.dirname(os.path.abspath(__file__))
METAL = f"{HERE}/su3_wilson_loop.metal"
DYLIB = "/tmp/libcwgauge.dylib"
L = int(os.environ.get("CW_L", "16"))
PLANES = [(mu, nu) for mu in range(4) for nu in range(mu+1, 4)]

grid = g.grid([L, L, L, L], g.single)
U = g.qcd.gauge.unit(grid)
def rand_su3(n, seed):
    rs = np.random.RandomState(seed)
    M = rs.randn(n, 3, 3) + 1j*rs.randn(n, 3, 3)
    Q, _ = np.linalg.qr(M); d = np.linalg.det(Q)
    return (Q / d[:, None, None]**(1/3)).astype(np.complex64)
for mu in range(4):
    a = U[mu][:]; n = a.reshape(-1, 3, 3).shape[0]
    U[mu][:] = rand_su3(n, seed=10+mu).reshape(a.shape)
P_gpt = g.qcd.gauge.plaquette(U)

mat = lambda x: x.reshape(-1, 3, 3).astype(np.complex64)
nsite = mat(U[0][:]).shape[0]
nV = nsite // 2

def pack(F):                        # vectorized: 4 assignments, no 9-element loop
    out = np.empty((nV, 9, 4), dtype=np.float32)
    s0 = F[0::2].reshape(nV, 9); s1 = F[1::2].reshape(nV, 9)
    out[:, :, 0] = s0.real; out[:, :, 1] = s0.imag
    out[:, :, 2] = s1.real; out[:, :, 3] = s1.imag
    return out

# Pre-Cshift the 4 plaquette links for every plane (common to GPU and CPU paths).
planes = []
for (mu, nu) in PLANES:
    A = mat(U[mu][:]); D = mat(U[nu][:])
    B = mat(g.cshift(U[nu], mu, 1)[:]); C = mat(g.cshift(U[mu], nu, 1)[:])
    planes.append((A, B, C, D))

# ---- in-process Metal dispatcher via ctypes ----
lib = ctypes.CDLL(DYLIB)
lib.cw_gauge_init.argtypes = [ctypes.c_char_p]; lib.cw_gauge_init.restype = ctypes.c_int
lib.cw_wilson_dispatch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                   ctypes.c_int, ctypes.c_void_p]
lib.cw_wilson_dispatch.restype = ctypes.c_int
assert lib.cw_gauge_init(METAL.encode()) == 0, "dispatcher init failed"
dag = np.array([0, 0, 1, 1], dtype=np.uint8)   # plaquette dagger pattern (A B C^dag D^dag)
out = np.empty((nV, 2), dtype=np.float32)

def gpu_plaq():
    tot = 0.0
    for (A, B, C, D) in planes:
        packed = np.ascontiguousarray(np.stack([pack(A), pack(B), pack(C), pack(D)]))  # (4,nV,9,4)
        lib.cw_wilson_dispatch(packed.ctypes.data, 4, dag.ctypes.data, nV, out.ctypes.data)
        retr = np.empty(nsite); retr[0::2] = out[:, 0]; retr[1::2] = out[:, 1]
        tot += retr.mean() / 3.0
    return tot / len(PLANES)

def cpu_plaq():
    tot = 0.0
    for (A, B, C, D) in planes:
        P = (A.astype(np.complex128) @ B.astype(np.complex128)
             @ C.conj().transpose(0, 2, 1).astype(np.complex128)
             @ D.conj().transpose(0, 2, 1).astype(np.complex128))
        tot += np.real(np.trace(P, axis1=1, axis2=2)).mean() / 3.0
    return tot / len(PLANES)

# correctness
pg = gpu_plaq(); pc = cpu_plaq()
print("=============================================")
print(f"In-process GPU gauge measurement on {grid.fdimensions}, {nsite} sites")
print(f"  GPU in-process plaquette = {pg:.10f}")
print(f"  CPU numpy plaquette      = {pc:.10f}")
print(f"  gpt-native plaquette     = {P_gpt:.10f}")
print(f"  |GPU - gpt| = {abs(pg - P_gpt):.3e}   |GPU - CPU| = {abs(pg - pc):.3e}")

# timing (warm both, then N reps)
N = 20
gpu_plaq(); cpu_plaq()
t = time.perf_counter()
for _ in range(N): gpu_plaq()
t_gpu = (time.perf_counter() - t) / N
t = time.perf_counter()
for _ in range(N): cpu_plaq()
t_cpu = (time.perf_counter() - t) / N
print(f"  in-process GPU: {t_gpu*1e3:8.2f} ms/plaquette")
print(f"  CPU numpy     : {t_cpu*1e3:8.2f} ms/plaquette")
print(f"  speedup       : {t_cpu/t_gpu:6.2f}x  (pack overhead included in GPU time)")
ok = abs(pg - P_gpt) < 1e-5
print("RESULT:", "IN-PROCESS GPU MEASUREMENT OK" if ok else "MISMATCH")
print("=============================================")

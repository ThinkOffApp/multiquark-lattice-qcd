#!/usr/bin/env python3
"""One-plane SU(3) plaquette parity: gpt CPU reference vs Metal GPU kernel, on a
real gpt gauge config. Ladder step 1 (ether's sequencing).

  gpt config -> pick (mu,nu) plane -> CPU Cshift align A,B,C,D ->
  numpy ReTr reference  AND  pack -> Metal su3_plaquette.metal -> GPU ReTr ->
  compare GPU vs numpy vs g.qcd.gauge.plaquette.

Run: PYTHONPATH=... python3 plaq_parity.py
"""
import os, subprocess, numpy as np, gpt as g

HERE = os.path.dirname(os.path.abspath(__file__))
PLANES = [(mu, nu) for mu in range(4) for nu in range(mu+1, 4)]  # all 6
TMP = "/tmp"

grid = g.grid([4, 4, 4, 4], g.single)
U = g.qcd.gauge.unit(grid)
def rand_su3(n, seed):
    rs = np.random.RandomState(seed)
    M = rs.randn(n, 3, 3) + 1j*rs.randn(n, 3, 3)
    Q, _ = np.linalg.qr(M); d = np.linalg.det(Q)
    return (Q / d[:, None, None]**(1/3)).astype(np.complex64)
for mu in range(4):
    a = U[mu][:]; n = a.reshape(-1, 3, 3).shape[0]
    U[mu][:] = rand_su3(n, seed=10+mu).reshape(a.shape)

P_gpt_global = g.qcd.gauge.plaquette(U)

mat = lambda x: x.reshape(-1, 3, 3).astype(np.complex64)
nsite = mat(U[0][:]).shape[0]
assert nsite % 2 == 0
nV = nsite // 2

def pack(F):  # (nsite,3,3) -> (nV,9,4) float32; float4=(s0_re,s0_im,s1_re,s1_im)
    out = np.zeros((nV, 9, 4), dtype=np.float32)
    s0 = F[0::2]; s1 = F[1::2]                       # even/odd sites -> 2 lanes
    for r in range(3):
        for c in range(3):
            k = r*3 + c
            out[:, k, 0] = s0[:, r, c].real; out[:, k, 1] = s0[:, r, c].imag
            out[:, k, 2] = s1[:, r, c].real; out[:, k, 3] = s1[:, r, c].imag
    return out

subprocess.run(["swiftc","-O", f"{HERE}/plaq_parity_dispatch.swift","-o","/tmp/ppd"], check=True)

print("=============================================")
print(f"SU(3) plaquette parity on {grid.fdimensions}, {nsite} sites")
gpu_plane_sum = 0.0; cpu_plane_sum = 0.0; worst = 0.0
for (MU, NU) in PLANES:
    A = mat(U[MU][:]); D = mat(U[NU][:])
    B = mat(g.cshift(U[NU], MU, 1)[:]); C = mat(g.cshift(U[MU], NU, 1)[:])
    # numpy CPU reference
    P = (A.astype(np.complex128) @ B.astype(np.complex128)
         @ C.conj().transpose(0,2,1).astype(np.complex128)
         @ D.conj().transpose(0,2,1).astype(np.complex128))
    retr_cpu = np.real(np.trace(P, axis1=1, axis2=2))
    plane_cpu = retr_cpu.mean() / 3.0
    # Metal GPU
    for name, F in (("A",A),("B",B),("C",C),("D",D)):
        pack(F).tofile(f"{TMP}/parity_{name}.bin")
    subprocess.run(["/tmp/ppd", str(nV), f"{TMP}/parity_A.bin", f"{TMP}/parity_B.bin",
                    f"{TMP}/parity_C.bin", f"{TMP}/parity_D.bin", f"{TMP}/parity_out.bin"],
                   check=True, stdout=subprocess.DEVNULL)
    gpu = np.fromfile(f"{TMP}/parity_out.bin", dtype=np.float32).reshape(nV, 2)
    retr_gpu = np.empty(nsite); retr_gpu[0::2] = gpu[:,0]; retr_gpu[1::2] = gpu[:,1]
    plane_gpu = retr_gpu.mean() / 3.0
    err = np.max(np.abs(retr_gpu - retr_cpu)); worst = max(worst, err)
    cpu_plane_sum += plane_cpu; gpu_plane_sum += plane_gpu
    print(f"  plane ({MU},{NU}): CPU {plane_cpu:+.8f}  GPU {plane_gpu:+.8f}  max|d| {err:.2e}")

P_gpu_global = gpu_plane_sum / len(PLANES)
P_cpu_global = cpu_plane_sum / len(PLANES)
print("  -------------------------------------------")
print(f"  GPU full plaquette   = {P_gpu_global:.10f}")
print(f"  numpy full plaquette = {P_cpu_global:.10f}")
print(f"  gpt  full plaquette  = {P_gpt_global:.10f}")
print(f"  |GPU - gpt|          = {abs(P_gpu_global - P_gpt_global):.3e}")
print(f"  worst per-site |GPU-CPU| ReTr = {worst:.3e} (float32)")
ok = worst < 1e-3 and abs(P_gpu_global - P_gpt_global) < 1e-5
print("RESULT:", "FULL PLAQUETTE PARITY PASS" if ok else "PARITY FAIL")
print("=============================================")

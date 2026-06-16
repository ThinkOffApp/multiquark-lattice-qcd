#!/usr/bin/env python3
"""Rectangular SU(3) Wilson-loop parity: numpy CPU reference vs Metal GPU kernel,
on a real gpt gauge config. Ladder step 3 (physical Cshift/cgpt parity).

  gpt config -> for each (R,T) in a plane: CPU Cshift-align the 2R+2T loop links
  -> numpy double-precision ordered product (reference) AND pack -> Metal
  su3_wilson_loop.metal WilsonLoopPath -> GPU ReTr -> compare per-site.

Anchor: the 1x1 loop must equal the plaquette, which is already validated against
gpt-native g.qcd.gauge.plaquette to 6.9e-12 (claudemm), tying this to gpt truth.

Run: PYTHONPATH=gpt/lib/cgpt/build:gpt/lib python3.12 wilson_parity.py
"""
import os, subprocess, numpy as np, gpt as g

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = "/tmp"
MU, NU = 0, 1                       # one plane is enough to prove the path product
LOOPS = [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 4)]

L = 8                               # 8^4: room for loops up to ~4 without heavy wraparound
grid = g.grid([L, L, L, L], g.single)
U = g.qcd.gauge.unit(grid)

def rand_su3(n, seed):              # g.qcd.gauge.random is broken here; use numpy QR
    rs = np.random.RandomState(seed)
    M = rs.randn(n, 3, 3) + 1j*rs.randn(n, 3, 3)
    Q, _ = np.linalg.qr(M); d = np.linalg.det(Q)
    return (Q / d[:, None, None]**(1/3)).astype(np.complex64)
for mu in range(4):
    a = U[mu][:]; n = a.reshape(-1, 3, 3).shape[0]
    U[mu][:] = rand_su3(n, seed=10+mu).reshape(a.shape)

# Plaquette anchor: GPU 1x1 loop should reproduce this gpt-native value.
P_gpt = g.qcd.gauge.plaquette(U)

mat = lambda x: x.reshape(-1, 3, 3).astype(np.complex64)
nsite = mat(U[0][:]).shape[0]
assert nsite % 2 == 0
nV = nsite // 2

def shift(field, a, b, mu, nu):     # field(x + a*e_mu + b*e_nu) aligned to x
    f = field
    if a: f = g.cshift(f, mu, a)
    if b: f = g.cshift(f, nu, b)
    return f

def pack(F):                        # (nsite,3,3)->(nV,9,4) f32; float4=(s0re,s0im,s1re,s1im)
    out = np.zeros((nV, 9, 4), dtype=np.float32)
    s0 = F[0::2]; s1 = F[1::2]
    for r in range(3):
        for c in range(3):
            k = r*3 + c
            out[:, k, 0] = s0[:, r, c].real; out[:, k, 1] = s0[:, r, c].imag
            out[:, k, 2] = s1[:, r, c].real; out[:, k, 3] = s1[:, r, c].imag
    return out

def loop_links(R, T, mu, nu):
    """Aligned link fields + dagger flags in path order around the R x T loop:
    +mu (R) , +nu (T) , -mu (R, dag) , -nu (T, dag)."""
    fields, dag = [], []
    for i in range(R):              # bottom, +mu
        fields.append(shift(U[mu], i, 0, mu, nu)); dag.append(0)
    for j in range(T):              # right, +nu  at x+R*mu
        fields.append(shift(U[nu], R, j, mu, nu)); dag.append(0)
    for i in range(R-1, -1, -1):    # top, -mu (dag) at nu=T
        fields.append(shift(U[mu], i, T, mu, nu)); dag.append(1)
    for j in range(T-1, -1, -1):    # left, -nu (dag) at mu=0
        fields.append(shift(U[nu], 0, j, mu, nu)); dag.append(1)
    return fields, dag

def cpu_retr(fields, dag):          # numpy double-precision ordered product reference
    acc = np.broadcast_to(np.eye(3, dtype=np.complex128), (nsite, 3, 3)).copy()
    for f, d in zip(fields, dag):
        Uk = mat(f[:]).astype(np.complex128)
        acc = acc @ (Uk.conj().transpose(0, 2, 1) if d else Uk)
    return np.real(np.trace(acc, axis1=1, axis2=2))

def gpu_retr(fields, dag):          # dispatch su3_wilson_loop.metal WilsonLoopPath
    nLink = len(fields)
    for idx, f in enumerate(fields):
        pack(mat(f[:])).tofile(f"{TMP}/wl_{idx}.bin")
    dbits = "".join(str(x) for x in dag)
    cmd = ["/tmp/wpd", str(nV), str(nLink), dbits, f"{TMP}/wl_out.bin"] + \
          [f"{TMP}/wl_{idx}.bin" for idx in range(nLink)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    gpu = np.fromfile(f"{TMP}/wl_out.bin", dtype=np.float32).reshape(nV, 2)
    retr = np.empty(nsite); retr[0::2] = gpu[:, 0]; retr[1::2] = gpu[:, 1]
    return retr

subprocess.run(["swiftc", "-O", f"{HERE}/wilson_parity_dispatch.swift", "-o", "/tmp/wpd"], check=True)

print("=============================================")
print(f"SU(3) Wilson-loop parity on {grid.fdimensions}, {nsite} sites, plane ({MU},{NU})")
worst_all = 0.0
for (R, T) in LOOPS:
    fields, dag = loop_links(R, T, MU, NU)
    rc = cpu_retr(fields, dag); rg = gpu_retr(fields, dag)
    err = np.max(np.abs(rg - rc)); worst_all = max(worst_all, err)
    print(f"  W({R},{T}) N={len(fields):2d}: CPU {rc.mean()/3.0:+.8f}  GPU {rg.mean()/3.0:+.8f}  max|d|ReTr {err:.2e}")

# gpt-native anchor: build the full plaquette from 1x1 GPU loops over all 6 planes
# and compare to g.qcd.gauge.plaquette. Ties the Wilson kernel to gpt truth.
PLANES = [(mu, nu) for mu in range(4) for nu in range(mu+1, 4)]
plaq_gpu = 0.0
for (mu, nu) in PLANES:
    f, d = loop_links(1, 1, mu, nu)
    plaq_gpu += gpu_retr(f, d).mean() / 3.0
plaq_gpu /= len(PLANES)

print("  -------------------------------------------")
print(f"  GPU plaquette (1x1 loop, 6 planes) = {plaq_gpu:.10f}")
print(f"  gpt-native g.qcd.gauge.plaquette   = {P_gpt:.10f}")
print(f"  |GPU - gpt| plaquette anchor       = {abs(plaq_gpu - P_gpt):.3e}")
print(f"  worst per-site |GPU-CPU| ReTr over all loops = {worst_all:.3e} (float32)")
ok = worst_all < 1e-3 and abs(plaq_gpu - P_gpt) < 1e-5
print("RESULT:", "WILSON-LOOP PARITY PASS" if ok else "PARITY FAIL")
print("=============================================")

import time
import gpt as g

# Genuine SU(2) (explicit gauge group, per PR #18 lesson)
L = [8, 8, 8, 8]
beta = 2.4
grid = g.grid(L, g.double)
rng = g.random("mb-bench")

otype = g.ot_matrix_su_n_fundamental_group(2)
U = [g.lattice(grid, otype) for _ in range(4)]
for u in U:
    u @= g.identity(u)
action = g.qcd.gauge.action.wilson(beta)
hb = g.algorithms.markov.su2_heat_bath(rng)
g.default.push_verbose("su2_heat_bath", False)

mask_rb = g.complex(g.grid(L, g.double, g.redblack))
mask_rb[:] = 1
mask = g.complex(grid)

def sweep():
    for cb in [g.even, g.odd]:
        mask[:] = 0
        mask_rb.checkerboard(cb)
        g.set_checkerboard(mask, mask_rb)
        for mu in range(4):
            hb(U[mu], action.staple(U, mu), mask)

# warm up / thermalize a little
for _ in range(10):
    sweep()

n = 20
t0 = time.time()
for _ in range(n):
    sweep()
dt = time.time() - t0

# plaquette (normalized average)
p = g.qcd.gauge.plaquette(U)
mi = g.mem_info()
print(f"RESULT L={L} beta={beta}")
print(f"RESULT sweeps={n} total={dt:.3f}s per_sweep={dt/n*1000:.1f}ms")
print(f"RESULT plaquette={p:.6f}  (SU(2) Wilson beta=2.4 literature ~0.63)")
print(f"RESULT accelerator_total={mi.get('accelerator_total')} backend={'gpu' if (mi.get('accelerator_total') or 0)>0 else 'cpu(GCD)'}")

#!/usr/bin/env python3
"""Parity + invariance gate for the off-axis 2Q implementation (spec P1).

Checks, on a random 8^4 SU(2) config (built via numpy quaternions —
g.qcd.gauge.random is broken on the Mini, see project memory):

1. orientation-count unit tests (pure python, no lattice);
2. on-axis consistency: a single-leg off-axis orientation must reproduce
   su2_2q_signal_scan.wilson_loop_trace (reverse traversal => complex
   conjugate trace; identical for SU(2), whose fundamental traces are real);
3. production offaxis_loop_traces vs an independent allocation-naive
   reference built from plain g() expressions (planar + 3D classes);
4. gauge invariance: random SU(2) gauge transform leaves every off-axis
   loop unchanged;
5. numpy assignment round-trip + unitarity of the constructed links.

Run (mini):
    PYTHONPATH=/Users/petrus/grid-gpt-upstream/lib/cgpt/build:/Users/petrus/grid-gpt-upstream/lib \
        python3 tools/test_offaxis_parity.py
Threshold: worst |delta| < 1e-11 (double precision), as in
tools/test_loop_workspace_parity.py (PR #24).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpt", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpt", "applications", "hmc"))

import gpt as g
import su2_offaxis as oa
import su2_2q_signal_scan as scan

TOL = 1e-11


def as_complex(x):
    if hasattr(x, "real"):
        return complex(float(x.real), float(x.imag))
    return complex(float(x), 0.0)


def random_su2_field(grid, rng, nd=4):
    """SU(2) links from normalized quaternions: U = a0*I + i*(a . sigma)."""
    otype = g.ot_matrix_su_n_fundamental_group(2)
    U = []
    n = int(grid.gsites)
    for _ in range(nd):
        a = rng.normal(size=(n, 4))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        m = np.empty((n, 2, 2), dtype=np.complex128)
        m[:, 0, 0] = a[:, 0] + 1j * a[:, 3]
        m[:, 0, 1] = a[:, 2] + 1j * a[:, 1]
        m[:, 1, 0] = -a[:, 2] + 1j * a[:, 1]
        m[:, 1, 1] = a[:, 0] - 1j * a[:, 3]
        lat = g.lattice(grid, otype)
        lat[:] = np.ascontiguousarray(m)
        U.append(lat)
    return U


def ref_shift(f, vec):
    out = f
    for d, s in enumerate(vec):
        if s != 0:
            out = g.cshift(out, d, s)
    return out


def _straight(axis, n):
    sgn = 1 if n > 0 else -1
    return [(axis, sgn)] * abs(n)


def ref_route(U, steps):
    nd = len(U)
    pos = [0] * nd
    acc = None
    for ax, sgn in steps:
        if sgn > 0:
            f = ref_shift(U[ax], pos)
            pos[ax] += 1
        else:
            pos[ax] -= 1
            f = ref_shift(g(g.adj(U[ax])), pos)
        acc = g.copy(f) if acc is None else g(acc * f)
    return acc


def ref_connector(U, orient):
    nd = len(U)
    axes = sorted(orient)
    if len(axes) == 1:
        return ref_route(U, _straight(axes[0], orient[axes[0]]))
    if len(axes) == 2:
        a1, a2 = axes
        pa = ref_route(U, _straight(a1, orient[a1]) + _straight(a2, orient[a2]))
        pb = ref_route(U, _straight(a2, orient[a2]) + _straight(a1, orient[a1]))
        return g(0.5 * pa + 0.5 * pb)
    assert len(axes) == 3
    terms = []
    for lead in axes:
        rest = [a for a in axes if a != lead]
        leg = ref_route(U, _straight(lead, orient[lead]))
        plan = ref_connector(U, {rest[0]: orient[rest[0]], rest[1]: orient[rest[1]]})
        shift = [0] * nd
        shift[lead] = orient[lead]
        terms.append(g(leg * ref_shift(plan, shift)))
    s = g(terms[0] + terms[1])
    s = g(s + terms[2])
    return g((1.0 / 3.0) * s)


def ref_offaxis_trace(U, tdir, orient, T):
    nd = len(U)
    S = ref_connector(U, orient)
    Lt = g.copy(U[tdir])
    for i in range(1, T):
        Lt = g(Lt * g.cshift(U[tdir], tdir, i))
    rvec = [0] * nd
    for ax, val in orient.items():
        rvec[ax] = val
    tvec = [0] * nd
    tvec[tdir] = T
    W = g(S * ref_shift(Lt, rvec))
    W = g(W * g.adj(ref_shift(S, tvec)))
    W = g(W * g.adj(Lt))
    ndim = U[0].otype.shape[0]
    tr = g.sum(g(g.trace(W))) / W.grid.gsites / ndim
    return as_complex(tr)


def main():
    worst = 0.0
    n_checks = 0
    failures = []

    def check(label, delta, tol=TOL):
        nonlocal worst, n_checks
        worst = max(worst, delta)
        n_checks += 1
        status = "ok" if delta < tol else "FAIL"
        print(f"{status:4s} {label}: |delta| = {delta:.3e}")
        if delta >= tol:
            failures.append(label)

    # ---- 1. orientation counts (no lattice needed) ----
    for cls, n_ap, n_full in [
        ((1, 1, 0), 3, 6),
        ((2, 1, 0), 6, 12),
        ((1, 1, 1), 1, 4),
        ((2, 2, 1), 3, 12),
        ((3, 2, 1), 6, 24),
    ]:
        got_ap = len(oa.offaxis_orientations(cls, [0, 1, 2], mode="axis-perm"))
        got_full = len(oa.offaxis_orientations(cls, [0, 1, 2], mode="full"))
        check(f"orient-count axis-perm {cls} ({got_ap} vs {n_ap})", abs(got_ap - n_ap), tol=0.5)
        check(f"orient-count full {cls} ({got_full} vs {n_full})", abs(got_full - n_full), tol=0.5)

    grid = g.grid([8, 8, 8, 8], g.double)
    rng = np.random.default_rng(13)
    U = random_su2_field(grid, rng)

    # ---- 5. assignment round-trip + unitarity ----
    arr = U[0][:]
    arr = np.asarray(arr).reshape(-1, 2, 2)
    uudag = np.einsum("sij,skj->sik", arr, arr.conj())
    eye = np.broadcast_to(np.eye(2), uudag.shape)
    check("links unitary (U Udag = 1)", float(np.abs(uudag - eye).max()))
    det = np.linalg.det(arr)
    check("links special (det = 1)", float(np.abs(det - 1.0).max()))

    tdir = 3
    Ts = [1, 2, 3]

    # ---- 2. on-axis consistency vs production wilson_loop_trace ----
    for n in (1, 2, 3):
        for sdir in (0, 1):
            mine = oa.offaxis_loop_traces(U, tdir, {sdir: n}, Ts)
            for T in Ts:
                ref = as_complex(scan.wilson_loop_trace(U, tdir, T, sdir, n))
                a = complex(*mine[T])
                # reverse traversal => conjugate trace (equal for SU(2))
                check(f"on-axis R={n} sdir={sdir} T={T}", abs(a - ref.conjugate()))

    # ---- 3. production vs naive reference, planar + 3D ----
    test_orients = [
        {0: 1, 1: 1},
        {0: 2, 1: 1},
        {1: 3, 2: 2},
        {0: 1, 1: 1, 2: 1},
        {0: 2, 1: 1, 2: 2},
        {0: 2, 1: 3, 2: 2},
    ]
    for orient in test_orients:
        mine = oa.offaxis_loop_traces(U, tdir, orient, Ts)
        for T in Ts:
            ref = ref_offaxis_trace(U, tdir, orient, T)
            a = complex(*mine[T])
            check(f"ref parity {sorted(orient.items())} T={T}", abs(a - ref))

    # ---- 4. gauge invariance ----
    om_field = random_su2_field(grid, np.random.default_rng(99), nd=1)[0]
    Up = []
    for mu in range(4):
        om_sh = g.cshift(om_field, mu, 1)
        Up.append(g(om_field * U[mu] * g.adj(om_sh)))
    oa.clear_workspaces()  # fresh buffers for the transformed field
    for orient in [{0: 2, 1: 1}, {0: 1, 1: 1, 2: 1}]:
        mine = oa.offaxis_loop_traces(U, tdir, orient, [2])
        oa.clear_workspaces()
        trans = oa.offaxis_loop_traces(Up, tdir, orient, [2])
        oa.clear_workspaces()
        check(
            f"gauge invariance {sorted(orient.items())} T=2",
            abs(complex(*mine[2]) - complex(*trans[2])),
        )

    print()
    if not failures and worst < TOL:
        print(f"OFF-AXIS PARITY PASS: {n_checks} checks, worst |delta| = {worst:.3e}")
        return 0
    print(f"OFF-AXIS PARITY FAIL: {len(failures)} failing checks, worst |delta| = {worst:.3e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Numerical parity test for the workspace-based Wilson-loop rewrite.

Compares wilson_loop_trace / wilson_loop_field / the flux shift+product path
between the workspace implementation (current su2_2q_signal_scan.py) and an
independent, allocation-naive reference implemented here, on a small random
gauge field. Must PASS (max |delta| < 1e-11) before the leak patch merges.

Run on a machine with a working cgpt build (the mb):
    source gpt/lib/cgpt/build/source.sh
    python3 tools/test_loop_workspace_parity.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpt", "lib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gpt", "applications", "hmc"))

import gpt as g
import su2_2q_signal_scan as scan


def ref_loop_matrix(U, mu, L_mu, nu, L_nu):
    """Allocation-naive reference: the pre-workspace algorithm, verbatim."""
    nd = len(U)
    W = g.copy(U[mu])
    for i in range(1, L_mu):
        W = g(W * g.cshift(U[mu], mu, i))
    for j in range(L_nu):
        tmp = U[nu]
        for d in range(nd):
            s = (L_mu if d == mu else 0) + (j if d == nu else 0)
            if s != 0:
                tmp = g.cshift(tmp, d, s)
        W = g(W * tmp)
    for i in range(L_mu - 1, -1, -1):
        tmp = g.adj(U[mu])
        for d in range(nd):
            s = (i if d == mu else 0) + (L_nu if d == nu else 0)
            if s != 0:
                tmp = g.cshift(tmp, d, s)
        W = g(W * tmp)
    for j in range(L_nu - 1, 0, -1):
        W = g(W * g.cshift(g.adj(U[nu]), nu, j))
    W = g(W * g.adj(U[nu]))
    return W


def ref_trace(U, mu, L_mu, nu, L_nu):
    W = ref_loop_matrix(U, mu, L_mu, nu, L_nu)
    ndim = U[0].otype.shape[0]
    return g.sum(g.trace(W)) / W.grid.gsites / ndim


def ref_field(U, mu, L_mu, nu, L_nu):
    W = ref_loop_matrix(U, mu, L_mu, nu, L_nu)
    ndim = U[0].otype.shape[0]
    return g(g.trace(W) / ndim)


def main():
    grid = g.grid([8, 8, 8, 8], g.double)
    rng = g.random("parity-seed-13")
    U = g.qcd.gauge.random(grid, rng)

    worst = 0.0
    n_checks = 0

    # 1. wilson_loop_trace over a spread of (R,T) and directions
    for (mu, nu) in [(3, 0), (3, 1), (0, 2)]:
        for (L_mu, L_nu) in [(1, 1), (2, 3), (4, 2), (3, 4)]:
            a = scan.wilson_loop_trace(U, mu, L_mu, nu, L_nu)
            b = ref_trace(U, mu, L_mu, nu, L_nu)
            d = abs(a - b)
            worst = max(worst, d)
            n_checks += 1
            print(f"trace mu={mu} nu={nu} L=({L_mu},{L_nu}): new={a:.15f} ref={b:.15f} |d|={d:.3e}")

    # 2. wilson_loop_field: site-level parity (norm of difference field)
    for (mu, nu, L_mu, L_nu) in [(3, 0, 2, 2), (3, 1, 3, 2)]:
        f_new = g.copy(scan.wilson_loop_field(U, mu, L_mu, nu, L_nu))
        f_ref = ref_field(U, mu, L_mu, nu, L_nu)
        diff = g(f_new - f_ref)
        d = g.norm2(diff) ** 0.5
        worst = max(worst, d)
        n_checks += 1
        print(f"field mu={mu} nu={nu} L=({L_mu},{L_nu}): |new-ref|_2 = {d:.3e}")

    # 3. flux shift path: workspace chained-cshift of a complex field vs naive
    p = g.complex(grid)
    rng.cnormal(p)
    ws = scan._workspace(U[0])
    for shift_vec in [(1, 0, 2, 0), (0, -1, 0, 3), (2, 2, 0, -1)]:
        cur, other = ws.Ca, ws.Cb
        src = p
        for mu_s, s in enumerate(shift_vec):
            if s != 0:
                g.cshift(cur, src, mu_s, s)
                src = cur
                cur, other = other, cur
        naive = p
        for mu_s, s in enumerate(shift_vec):
            if s != 0:
                naive = g.cshift(naive, mu_s, s)
        d = g.norm2(g(src - naive)) ** 0.5
        worst = max(worst, d)
        n_checks += 1
        print(f"shift {shift_vec}: |ws-naive|_2 = {d:.3e}")

    print()
    if worst < 1e-11:
        print(f"PARITY PASS: {n_checks} checks, worst |delta| = {worst:.3e}")
        return 0
    print(f"PARITY FAIL: worst |delta| = {worst:.3e} (threshold 1e-11)")
    return 1


if __name__ == "__main__":
    sys.exit(main())

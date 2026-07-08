"""Off-axis 2Q Wilson loops with PGM symmetrized paths.

P1 of docs/4q_measurement_spec.md: the canonical class list (spec §8,
extracted verbatim from the PGM papers — docs/pgm_offaxis_extraction.md) and
the symmetrized path construction of hep-lat/9404004 §3.1:

- planar (x,y,0): the spatial connector is the equal-weight average of the
  two L-shaped routes, S = 1/2 [P(a) + P(b)];
- 3D (x,y,z): the symmetric cuboid-edge combination
  S = 1/3 [P(x) S(yz) + P(y) S(xz) + P(z) S(xy)], each planar factor itself
  the 1/2 (a+b) L-average.

The averaged connector S is still gauge-covariant (every route connects the
same endpoints), so W(Rvec, T) = < Tr[ S(x) Lt(x+Rvec) S(x+T that)^dag
Lt(x)^dag ] > is a proper Wilson loop; symmetrization happens at the
OPERATOR level exactly as in the papers, not on closed loops.

All lattice arithmetic goes through gpt/Grid expressions (g.cshift, g.eval,
g.trace, g.sum) into preallocated workspace buffers — the same zero-alloc
discipline and the same execution engine (Metal-accelerated Grid where
built) as the on-axis loops in su2_2q_signal_scan.py (spec §7.4).

Loop keys: "Rv{a}_{b}_{c}_T{t}" with (a,b,c) the canonical class (sorted
descending), e.g. Rv2_1_0_T3. On-axis classes are NOT re-measured here.
"""

import itertools

import gpt as g


# Canonical off-axis class list — spec §8, one representative per cubic
# class, components sorted descending. Provenance per group in
# docs/pgm_offaxis_extraction.md.
OFF_AXIS_CLASSES = [
    # planar L-shaped grid: hep-lat/9301006 Table 1 (beta=2.4, MC-measured)
    (1, 1, 0), (2, 1, 0), (2, 2, 0), (3, 1, 0), (3, 2, 0), (3, 3, 0),
    (4, 1, 0), (4, 2, 0), (4, 3, 0), (5, 1, 0), (5, 2, 0), (5, 3, 0),
    (6, 1, 0), (6, 2, 0), (6, 3, 0),
    # diagonal series: hep-lat/9508002 Tables 4-5, hep-lat/9804004 flux
    # diagonals; (7,7,0) is ours (completes the square-d=7 / tetra anchor)
    (4, 4, 0), (5, 5, 0), (6, 6, 0), (7, 7, 0), (8, 8, 0),
    # near-diagonal: hep-lat/9508002 Tables 4-5
    (5, 4, 0), (6, 5, 0),
    # 3D vectors: hep-lat/9404004 Tables 5-6
    (1, 1, 1), (1, 2, 1), (1, 3, 1), (1, 4, 1), (2, 1, 2), (2, 3, 2),
]


def class_key(cls_vec):
    a, b, c = sorted((abs(x) for x in cls_vec), reverse=True)
    return f"Rv{a}_{b}_{c}"


def class_r(cls_vec):
    return sum(x * x for x in cls_vec) ** 0.5


def offaxis_orientations(cls_vec, spatial_dirs, mode="axis-perm"):
    """Distinct orientation vectors of a cubic class over the given spatial
    directions. Returns a list of dicts {axis: component != 0}.

    axis-perm (default): distinct assignments of the class components to the
      spatial axes, all components positive. Reflections are omitted: the
      ensemble average is invariant under them (parity/cubic symmetry of the
      action), so positive-octant permutations give an unbiased mean; the
      omitted mirror copies would only add statistics.
    full: additionally all sign choices, modulo overall Rvec -> -Rvec
      (path reversal conjugates the trace; Re W is unchanged).
    """
    comps = sorted((abs(int(x)) for x in cls_vec), reverse=True)
    if len(spatial_dirs) != 3 or len(comps) != 3:
        raise ValueError("off-axis classes are 3-component over 3 spatial dirs")
    out = []
    seen = set()
    for perm in set(itertools.permutations(comps)):
        base = {d: n for d, n in zip(spatial_dirs, perm) if n != 0}
        if mode == "axis-perm":
            key = tuple(sorted(base.items()))
            if key not in seen:
                seen.add(key)
                out.append(base)
            continue
        if mode != "full":
            raise ValueError(f"unknown orientation mode {mode!r}")
        axes = sorted(base)
        for signs in itertools.product((1, -1), repeat=len(axes)):
            # modulo overall reversal: first component positive
            if signs[0] < 0:
                continue
            vec = {ax: s * base[ax] for ax, s in zip(axes, signs)}
            key = tuple(sorted(vec.items()))
            if key not in seen:
                seen.add(key)
                out.append(vec)
    out.sort(key=lambda v: tuple(sorted(v.items())))
    return out


# ---------------------------------------------------------------------------
# Workspace: fixed buffer set per grid, mirroring su2_2q_signal_scan's
# _LatticeWorkspace (PR #24). All off-axis lattice temporaries live here so
# steady-state per-measurement allocation is zero.
# ---------------------------------------------------------------------------

class _OffaxisWorkspace:
    def __init__(self, template):
        # route/product ping-pong + shift chains
        self.Ra = g.lattice(template)
        self.Rb = g.lattice(template)
        self.Ca = g.lattice(template)
        self.Cb = g.lattice(template)
        # symmetrized-connector assembly
        self.Ssum = g.lattice(template)   # accumulates route/term average
        self.Sleg = g.lattice(template)   # straight-leg hold (3D terms)
        self.Splan = g.lattice(template)  # planar sub-connector hold (3D)
        self.S = g.lattice(template)      # finished connector for the orientation
        # temporal line and assembly
        self.Lt = g.lattice(template)     # cumulative time-line product
        self.LtR = g.lattice(template)    # Lt shifted by Rvec
        self.St = g.lattice(template)     # S shifted by T*that (incremental)
        self.Adj = g.lattice(template)    # adjoint hold
        self.Wa = g.lattice(template)     # final product ping
        self.Wb = g.lattice(template)     # final product pong
        self.tr = g.complex(template.grid)


_WORKSPACES = {}


def _workspace(template):
    key = id(template.grid)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = _OffaxisWorkspace(template)
        _WORKSPACES[key] = ws
    return ws


def clear_workspaces():
    """Drop workspace buffers (call with the scan's cache clears if needed)."""
    _WORKSPACES.clear()


def _shift_into(dst_pair, src, shift):
    """Chained cshift of src by integer vector `shift` using the (a, b)
    buffer pair; returns the buffer holding the result (or src if no-op)."""
    cur, other = dst_pair
    out = src
    for d, s in enumerate(shift):
        if s != 0:
            g.cshift(cur, out, d, s)
            out = cur
            cur, other = other, cur
    return out


def _route_product(U, steps, ws):
    """Product of links along `steps` (sequence of (axis, sign) unit steps)
    starting at every site x. Result lands in ws.Ra or ws.Rb; valid until the
    next _route_product call on the same workspace."""
    nd = len(U)
    pos = [0] * nd
    acc = None          # buffer holding the running product
    spare = None
    for axis, sgn in steps:
        if sgn > 0:
            src, shift = U[axis], list(pos)
            pos[axis] += 1
        else:
            pos[axis] -= 1
            g.eval(ws.Adj, g.adj(U[axis]))
            src, shift = ws.Adj, list(pos)
        factor = _shift_into((ws.Ca, ws.Cb), src, shift)
        if acc is None:
            acc, spare = ws.Ra, ws.Rb
            g.copy(acc, factor)
        else:
            g.eval(spare, acc * factor)
            acc, spare = spare, acc
    return acc


def _straight_steps(axis, n):
    sgn = 1 if n > 0 else -1
    return [(axis, sgn)] * abs(n)


def _planar_connector_into(dst, U, ax1, n1, ax2, n2, ws):
    """Symmetrized planar connector 1/2 [P(ax1-first) + P(ax2-first)] -> dst."""
    p = _route_product(U, _straight_steps(ax1, n1) + _straight_steps(ax2, n2), ws)
    g.copy(dst, p)
    p = _route_product(U, _straight_steps(ax2, n2) + _straight_steps(ax1, n1), ws)
    g.eval(ws.Ssum, 0.5 * dst + 0.5 * p)
    g.copy(dst, ws.Ssum)


def build_connector(U, orient, ws):
    """Symmetrized spatial connector for orientation dict {axis: comp}.
    Result -> ws.S. 1 leg: straight; 2 legs: 1/2 L-pair; 3 legs: 1/3 cuboid
    combination (hep-lat/9404004 sec 3.1)."""
    axes = sorted(orient)
    if len(axes) == 1:
        p = _route_product(U, _straight_steps(axes[0], orient[axes[0]]), ws)
        g.copy(ws.S, p)
        return ws.S
    if len(axes) == 2:
        (a1, a2) = axes
        _planar_connector_into(ws.S, U, a1, orient[a1], a2, orient[a2], ws)
        return ws.S
    if len(axes) != 3:
        raise ValueError("connector supports 1-3 legs")
    first = True
    for lead in axes:
        rest = [ax for ax in axes if ax != lead]
        n_lead = orient[lead]
        # straight leg along `lead`
        p = _route_product(U, _straight_steps(lead, n_lead), ws)
        g.copy(ws.Sleg, p)
        # planar connector for the remaining two axes, then shift it to the
        # end of the lead leg
        _planar_connector_into(ws.Splan, U, rest[0], orient[rest[0]], rest[1], orient[rest[1]], ws)
        shift = [0] * len(U)
        shift[lead] = n_lead
        planar_shifted = _shift_into((ws.Ca, ws.Cb), ws.Splan, shift)
        if first:
            g.eval(ws.S, ws.Sleg * planar_shifted)
            first = False
        else:
            g.eval(ws.Wa, ws.Sleg * planar_shifted)
            g.eval(ws.Wb, ws.S + ws.Wa)
            g.copy(ws.S, ws.Wb)
    g.eval(ws.Wa, (1.0 / 3.0) * ws.S)
    g.copy(ws.S, ws.Wa)
    return ws.S


def offaxis_loop_traces(U, tdir, orient, Ts, ws=None):
    """Re/Im volume-averaged Tr W(Rvec, T)/N for one orientation and all T in
    Ts (ascending). Returns {T: (re, im)}.

    W(x) = S(x) * Lt(x+Rvec) * adj(S(x+T that)) * adj(Lt(x)); the time line
    Lt and the shifted connector St are extended incrementally in T.
    """
    if ws is None:
        ws = _workspace(U[0])
    nd = len(U)
    ndim = U[0].otype.shape[0]
    Ts = sorted(set(int(t) for t in Ts))
    if not Ts or Ts[0] < 1:
        raise ValueError("Ts must be positive")

    build_connector(U, orient, ws)      # -> ws.S (stable: no later
                                        # _route_product calls this orientation)
    rshift = [0] * nd
    for ax, n in orient.items():
        rshift[ax] = n

    out = {}
    t_cur = 0
    # ws.Lt holds the product of t_cur time links; ws.St holds S shifted by
    # t_cur in tdir.
    for T in Ts:
        while t_cur < T:
            if t_cur == 0:
                g.copy(ws.Lt, U[tdir])
                g.copy(ws.St, ws.S)
                g.cshift(ws.Wa, ws.St, tdir, 1)
                g.copy(ws.St, ws.Wa)
            else:
                g.cshift(ws.Wa, U[tdir], tdir, t_cur)
                g.eval(ws.Wb, ws.Lt * ws.Wa)
                g.copy(ws.Lt, ws.Wb)
                g.cshift(ws.Wa, ws.St, tdir, 1)
                g.copy(ws.St, ws.Wa)
            t_cur += 1
        lt_r = _shift_into((ws.Ca, ws.Cb), ws.Lt, rshift)
        g.copy(ws.LtR, lt_r)
        g.eval(ws.Adj, g.adj(ws.St))
        g.eval(ws.Wa, ws.S * ws.LtR)
        g.eval(ws.Wb, ws.Wa * ws.Adj)
        g.eval(ws.Adj, g.adj(ws.Lt))
        g.eval(ws.Wa, ws.Wb * ws.Adj)
        g.eval(ws.tr, g.trace(ws.Wa))
        tr = g.sum(ws.tr) / ws.tr.grid.gsites / ndim
        if hasattr(tr, "real"):
            out[T] = (float(tr.real), float(tr.imag))
        else:
            out[T] = (float(tr), 0.0)
    return out


def measure_offaxis_loops_for_tdir(
    U_use,
    tdir,
    classes,
    Ts,
    loops_acc,
    progress_cb=None,
    orient_mode="axis-perm",
):
    """Accumulate off-axis loop samples into loops_acc under
    "Rv{a}_{b}_{c}_T{t}" keys, one (re, im) sample per orientation — the
    same accumulate-then-mean contract as measure_loops_for_tdir, so
    orientation, tdir, multihit and multilevel averaging all apply
    unchanged."""
    nd = len(U_use)
    spatial_dirs = [mu for mu in range(nd) if mu != tdir]
    ws = _workspace(U_use[0])
    for cls_vec in classes:
        ckey = class_key(cls_vec)
        for orient in offaxis_orientations(cls_vec, spatial_dirs, mode=orient_mode):
            traces = offaxis_loop_traces(U_use, tdir, orient, Ts, ws=ws)
            for T, val in traces.items():
                key = f"{ckey}_T{T}"
                loops_acc.setdefault(key, []).append(val)
            if progress_cb is not None:
                progress_cb(
                    "measure_offaxis",
                    1,
                    cursor={
                        "kind": "offaxis",
                        "tdir": int(tdir),
                        "class": ckey,
                        "orient": {int(k): int(v) for k, v in orient.items()},
                    },
                )

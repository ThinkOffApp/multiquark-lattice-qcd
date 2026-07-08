# 4Q + Off-Axis Measurement Extension — Specification

Status: ACCEPTED by Petrus 2026-07-07 (decisions in §7); technical review by
claudeMB pending on this PR. Gauge group: **SU(2)** throughout —
same as the existing production pipeline and the PGM program this extends.
Target lattice: 24⁴ at β=2.4 (dev runs on 16⁴/8⁴), on the existing
generation → measurement → live-dashboard chain.

## 1. Physics goals

1. **Off-axis 2Q potential** V(R⃗) on non-axis separations — fills the gaps
   between integer R, tests restoration of rotational invariance (lattice
   artifacts show as scatter off the Cornell curve at small R), and anchors
   the same R values the tetrahedron uses (R = d√2).
2. **4Q binding energies** B(geometry, d) = E₄Q − E₂Q,pair for
   **square** (d = 2…8) and **tetrahedron** (d = 2…6, edge R = d√2 ≈ 2.8…8.5)
   geometries — the multiquark question proper: is the 4-body ground state
   below two independent flux tubes, and by how much.
3. (Later phase) 4Q flux distributions on the symmetry-reduced wedges — where
   does the field rearrange when binding turns on.

## 2. Geometries and quark positions (lattice sites)

| Geometry | Positions | Scan | Point group | Reduction |
|---|---|---|---|---|
| off-axis 2Q | 0 → R⃗ for the canonical PGM class list (§8) | \|R\| ≤ ~8.5 | axial C₂ᵥ-type | orbit-avg |
| square | (0,0,0),(d,0,0),(0,d,0),(d,d,0) | d = 2…8 | D₄h (order 16) | 1/16 |
| tetra | (0,0,0),(d,d,0),(d,0,d),(0,d,d) | d = 2…6 | T_d (order 24) | 1/24 |

All tetra vertices are lattice sites for every integer d (verified). The
fundamental-domain conditions in `tools/geometry_viewer.html` are verified
exact (20k-orbit brute-force test, 2026-07-07) and are reused verbatim for
flux sampling.

**Weights (hard requirement):** flux/observable sums over a reduced wedge must
weight each site by `|G| / |stab(site)|` — NOT flat ×16/×24. On-wall sites are
14–50% of the wedge; flat weighting overcounts sums by ~56% at d=6. The
reduction helper must return `(site, weight)` pairs and a unit test must check
`Σ weights == full-volume count` for every geometry and d (both parities).

## 3. Operators

### 3.1 Off-axis 2Q
Wilson loop with spatial legs built as **PGM symmetrized paths** (corrected
2026-07-08 — the earlier draft said "canonical staircase, fixed lexicographic
order"; the PGM papers did not use a single staircase):

- **Planar (x,y,0)**: equal-weight average of the two L-shaped routes,
  P(x,y) = ½[P(a) + P(b)] (hep-lat/9404004 §3.1 fig. 2; identical statement
  in hep-lat/9508002 §2).
- **3D (x,y,z)**: symmetric cuboid-edge combination
  P(x,y,z) = ⅓[P(x)P(yz) + P(y)P(xz) + P(z)P(xy)], each planar factor itself
  the ½(a+b) L-average — effectively the 6 symmetric routes along cuboid
  edges (hep-lat/9404004 §3.1; UKQCD hep-lat/9209007 used the same 2-/6-route
  symmetric sums).

Symmetrization keeps the operator invariant under the coordinate
interchanges that fix R⃗ and improves ground-state overlap over any single
staircase. Same smearing (12-step spatial), same T set {1…6}, same V_eff
plateau extraction as on-axis. Cubic-orbit averaging on top: measure all
distinct orientations of each R⃗ class per time direction and average before
the T-fit — same statistical role that time-direction averaging plays today.

### 3.2 4Q correlation matrix (PGM formalism)
Four static quarks admit two independent pairings into SU(2) singlets:
A = (Q₁Q₂)(Q₃Q₄), B = (Q₁Q₃)(Q₂Q₄). Measure the 2×2 matrix

```
W_ij(T) = ⟨ P_i(0) · [4 temporal lines] · P_j(T)† ⟩,  i,j ∈ {A,B}
```

- **Diagonal W_AA, W_BB**: products of two rectangular (or staircase) Wilson
  loops — existing loop machinery, run twice per pairing.
- **Off-diagonal W_AB**: one closed contour — spatial connectors of pairing A
  at t=0, the four temporal lines, spatial connectors of pairing B at t=T.
  New but small: the path builder already composes arbitrary link products;
  this is a single trace over a longer path.
- Extract E₄Q via the generalized eigenvalue problem on W(T)/W(T₀) (2×2 GEVP,
  closed-form for 2×2 — no numerics dependency), then the same V_eff plateau
  fit per eigenvalue. Binding: B(d) = E₄Q^(0)(d) − 2·V(R_pair) with V from the
  SAME configs (correlated errors partially cancel — jackknife over configs).

Note the square has a third pairing (diagonals) that mixes only weakly; PGM
used the 2×2 basis and so do we (documented limitation).

## 4. Estimator reuse (nothing new invented)

- 12-step spatial smearing on connectors (ground-state overlap).
- Multilevel blocks × multihit on temporal lines — identical mechanics;
  the 4Q observable is linear in each temporal line's sub-average exactly like
  2Q, so blocks×hits factorization carries over unchanged.
- Full-volume translation averaging via the existing sampler.
- Time-direction (×4) orientation averaging.
- Same live JSON/JSONL schema: loop keys extended to
  `G<geom>_d<d>_P<ij>_T<t>` (e.g. `Gsq_d4_PAB_T3`); off-axis 2Q keys
  `Rv<class>_<n>_T<t>` (e.g. `Rv110_3_T4`). Dashboard plateau/loop panels key
  off these transparently (R-selector becomes a geometry+size selector).

## 5. Implementation plan (phased, each lands runnable)

- **P1 — off-axis 2Q** (smallest diff, validates the path builder):
  symmetrized L/cuboid spatial paths (§3.1) over the §8 class list + orbit
  averaging + dashboard keys. Gate: off-axis V(R) points lie on the Cornell
  fit through on-axis points within 2σ.
- **P2 — square 4Q**: pairing connectors, 2×2 matrix, GEVP, binding output.
  Gate: at large pair separation the ground state → 2·V(d) within errors;
  W_AB → 0 as separation grows.
- **P3 — tetra 4Q**: same machinery, tetra positions/pairings.
- **P4 — 4Q flux profiles** on the verified wedges with orbit weights.

Each phase: parity test at 8⁴ against a slow reference implementation
(direct link-product evaluation, no multilevel) before production, same
discipline as PR #24's bit-identical check.

## 6. Cost estimate (24⁴, per config)

Off-axis 2Q, measured on 8⁴ (PR #31): full 28-class list ≈ 2.7× current 2Q
loop cost (124 symmetrized passes vs 108 rectangles per tdir); the
production diagonal subset (§7.5) is 24 passes ≈ 0.5×. (The original 1.5×
guess for the full list was wrong.)
4Q square/tetra: 3 matrix elements × 7 (5) sizes × 6 T × 4 tdirs ≈ 2–3× the
2Q loop budget at equal multilevel settings — well inside the current
per-measurement envelope; flux phases dominate cost and get the 16×/24×
wedge reduction. Run tiering follows the leak situation (2x1 now, 8x2 when
the reuse-pool/census-v3 or process-split lands).

## 7. Decisions (Petrus, 2026-07-07, thinkoff-development)

1. **Off-axis classes: use the exact PGM set.** DONE 2026-07-08 — class list
   extracted verbatim from the PGM/UKQCD paper tables (full provenance in
   `docs/pgm_offaxis_extraction.md`) and encoded as the canonical list in §8;
   the (n,n,0)/(n,n,n)/(2n,n,0) placeholder is replaced. The extraction also
   corrected §3.1: PGM used symmetrized L-shaped / cuboid-edge paths, not a
   lexicographic staircase.
2. **Square pairings: keep PGM's 2×2 basis.** No third (diagonal) pairing; the
   omission stays a documented limitation as in §3.2.
3. **Scope: all of it.** On-axis 2Q (existing) + off-axis 2Q + square + tetra
   are all in scope — the P1→P4 order in §5 is a landing sequence, not a
   selection.
   **Refined 2026-07-08** ("We only measure diagonal off axis for
   subtraction from 4q potentials"): PRODUCTION off-axis = the diagonal
   classes (n,n,0), n = 1…8, only. These are exactly the R = d√2 pair
   potentials entering B(d) = E₄Q − 2·V(R_pair): every tetra vertex pair is
   a (d,d,0)-type separation, and the square's 2×2 pairings are on-axis
   sides already measured. Implemented as `--offaxis-classes diag` (the
   default; `su2_offaxis.DIAGONAL_CLASSES`). The full §8 list remains
   available via `--offaxis-classes all` for rotational-invariance studies
   — it is machinery, not production scope.
4. **GPU-resident throughout** ("implement all the listed measurements using
   gpu fully"): every new measurement inner loop — symmetrized/connector path
   products, 4Q contour traces, multilevel sub-averages — runs on the GPU
   (Metal), same as the existing plaquette/Wilson-loop kernels. CPU is
   orchestration, GEVP (closed-form 2×2), fits, and I/O only. The 8⁴ parity
   gates in §5 compare GPU results against the slow CPU reference.

## 8. Canonical off-axis class list (exact PGM set, extracted 2026-07-08)

One representative per cubic class, lattice units; sources verbatim from the
papers (full extraction with quotes and caveats:
`docs/pgm_offaxis_extraction.md`). Orientation averaging (§3.1) generates the
full orbits.

```
OFF_AXIS_CLASSES = [
  # -- planar L-shaped grid: Green-Michael-Paton-Sainio hep-lat/9301006
  #    Table 1 (beta=2.4, 16^3x32, MC-measured; y = 1..3, x <= 6)
  (1,1,0), (2,1,0), (2,2,0), (3,1,0), (3,2,0), (3,3,0),
  (4,1,0), (4,2,0), (4,3,0), (5,1,0), (5,2,0), (5,3,0),
  (6,1,0), (6,2,0), (6,3,0),
  # -- diagonal series (d,d,0): 9508002 Tables 4-5 for d=4,5;
  #    9804004 flux paper measured sqrt(2)*R diagonals R=2,4,6,8
  (4,4,0), (5,5,0), (6,6,0), (8,8,0),
  (7,7,0),                     # OURS (not in a PGM table): completes the
                               # square-d=7 / tetra-range anchor
  # -- near-diagonal: 9508002 Tables 4-5
  (5,4,0), (6,5,0),
  # -- 3D vectors: Green-Michael-Sainio hep-lat/9404004 Tables 5-6
  (1,1,1), (1,2,1), (1,3,1), (1,4,1), (2,1,2), (2,3,2),
]
```

Notes anchored in the extraction:

- The diagonals (d,d,0) carry |R| = d√2 — exactly the tetra edge lengths
  (§1 goal 1), so tetra binding at edge R uses a directly measured pair
  potential, PGM-style (9804004 measured the diagonals precisely because
  interpolating on-axis data violates rotational invariance at these β).
- PGM's tetra coordinates ((0,0,0),(r,0,d),(0,d,d),(r,d,0) with r=d,
  hep-lat/9412029) are the SAME four sites as our §2 tetra row — consistency
  confirmed, no change needed.
- PGM measured more geometry families (rectangles, tilted rectangles,
  linear, quadrilateral, non-planar; 3×3 bases for the last three) — out of
  scope here per §7.3, listed in the extraction doc as future extensions.
- Caveat from the sources: hep-lat/9404004 says its Table 5 is only a subset
  of all 2Q potentials produced, and hep-lat/9608147 refers to a 29-vector
  two-body list that is not printed in any of the papers. If the original
  run lists still exist offline, they supersede this reconstruction —
  flagged to Petrus.

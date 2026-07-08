# 4Q Flux Measurement Operators — Square & Tetra (Design Spec)

Status: DESIGN for review (requested by Petrus 2026-07-08, Fable pass).
Extends `4q_measurement_spec.md` (P4 there) into a concrete operator design.
Gauge group **SU(2)** throughout. Program lineage: this is the 4Q flux
program of hep-lat/9804004 (squares + diagonal 2Q at β=2.4, R=2,4,6,8) with
the tetrahedron added — tetra flux was NOT measured in the published PGM
papers; it is new territory, on the T_d wedge verified 2026-07-07.

## 1. Physics goal

Map where the chromo-field rearranges when 4Q binding turns on:

    Δf(r) = f_4Q^(0)(r) − [ f_2Q^pair1(r) + f_2Q^pair2(r) ]

with all three terms measured on the SAME configurations so the vacuum and
much of the statistical noise cancel site-by-site. Ground state AND first
excited state (the flip-flop between pairings is visible in f^(1) − f^(0)
— both come free from the same raw data, §3). 9804004 measured exactly the
required reference set: on-axis 2Q, diagonal 2Q at √2·R, and the square —
consistent with the production decision that off-axis 2Q = diagonals only.

## 2. The operator

### 2.1 Flux correlator matrix

For each 4Q geometry (pairing basis i,j ∈ {A,B}, complete for SU(2)) and
probe offset r from the geometry center, plaquette plane class P:

    F_ij,P(r; T) = ⟨ Tr W_ij(x₀; T) · □_P(x₀ + c + r + (T/2)·t̂) ⟩_{x₀}

- W_ij(x₀; T): the 4Q correlation-matrix element as a per-site trace FIELD
  (base point x₀), exactly the P2/P3 objects before volume averaging.
- □_P: bare (UNSMEARED-field) plaquette field of plane class P — probe on
  the unsmeared links, source on the smeared links (PR #12 discipline).
- c: offset from base point to the geometry center (fixed per geometry).
- Probe timeslice: T/2 above the source's t=0 plane → T even for flux
  (T_flux = 4 default; the potential T-scan is unaffected).
- ⟨·⟩_{x₀}: full translation average — implemented as one chained cshift of
  the probe field by (r + c + T/2·t̂) then pointwise multiply and g.sum
  (the existing 2Q flux shift+product path, workspace-buffered).

Also store per measurement: W̄_ij(T) = ⟨Tr W_ij⟩ (already produced by
P2/P3) and the vacuum ⟨□_P⟩. Raw triplet (F, W̄, □̄) goes to disk; NOTHING
is subtracted or divided at measurement time (ensemble-level ratio + single
vacuum subtraction in postprocessing — PR #13/#14/#15 discipline).

### 2.2 State projection (postprocessing only)

Solve the 2×2 GEVP W(T) v = λ W(T₀) v from the same-run potential data.
For state n with eigenvector ṽⁿ:

    f_P^(n)(r) = [ Σ_ij ṽⁿ_i ṽⁿ_j F_ij,P(r; T_flux) ]
                 / [ Σ_ij ṽⁿ_i ṽⁿ_j W̄_ij(T_flux) ]  −  ⟨□_P⟩

Contract with ṽ⁰ → ground-state flux; ṽ¹ → excited state. Keeping the full
F_ij matrix (3 independent elements: AA, BB, AB) makes the (T₀, T) choice a
postprocessing stability check, not a burned-in measurement decision.
Consistency check for free: T_d (square: D₄h) symmetry ⇒ F_AA = F_BB within
errors after frame mapping.

### 2.3 Plane classes and E/B decomposition

Store all 6 plaquette orientations, keyed RELATIVE to the source frame:
E_a (plane spanned by tdir and spatial axis a) and B_a (spatial plane with
normal a), a = 1,2,3 in the geometry's canonical frame. Postprocessing
forms the Euclidean combinations: action density s ∝ Σ_P f_P, energy
density ε ∝ Σ_E f − Σ_B f (Michael sum-rule conventions). Because
measurement iterates over 4 time directions and geometry orientations, each
measured (tdir, lattice plane) pair is mapped back to the canonical frame
by the same signed permutation that generated the orientation — the frame
map is pure bookkeeping and gets its own round-trip unit test (§7).

## 3. Probe region: boxes, wedges, weights

- Probe box: bounding box of the quark positions expanded by margin m
  (default m = 3) in every direction, in center-relative DOUBLED integer
  coordinates (doubling removes the half-integer center for odd d — the
  trick already validated for the viewer wedges).
- Reduce the box by the geometry's full stabilizer group; each kept site
  carries weight |G| / |stab(site)|. This is the 56%-overcount trap: flat
  ×|G| weighting is forbidden; the reduction helper returns (site, weight)
  pairs only.
- **Generic stabilizer engine** (new, replaces per-geometry hand-coded
  conditions): enumerate the 48 signed permutations, keep those mapping the
  quark position set to itself (for 2Q in SU(2), quark ≡ antiquark —
  pseudoreal — so endpoint swap is a symmetry), apply to probe sites
  directly. Expected group orders, asserted as unit tests:
    on-axis 2Q: 16 · diagonal 2Q (d,d,0): 8 · square: 16 (D₄h) ·
    tetra: 24 (T_d).
  The engine must reproduce the hand-verified viewer wedges for square and
  tetra EXACTLY (site-set equality test), and Σ weights == full box volume
  for every geometry, d, and parity.

## 4. Source set (all on the same configs)

| Source | Positions | Sizes (flux) | Reduction |
|---|---|---|---|
| 2Q on-axis | 0 → (d,0,0) | d = 2,4,6,8 | 1/16 |
| 2Q diagonal | 0 → (d,d,0), symmetrized-L connector (P1) | d = 2,4,6 (R=√2d ≈ 2.8–8.5) | 1/8 |
| square 4Q | side d, pairings A,B | d = 2,4,6,8 | 1/16 |
| tetra 4Q | (0,0,0),(d,d,0),(d,0,d),(0,d,d) | d = 2,4 (edge √2d) | 1/24 |

Flux sizes are the PGM 9804004 set (even d), NOT the full potential scan —
potentials at every d, flux at the even subset. Tetra capped at d = 4
initially (box grows as (2d+2m+1)³; d=6 pending cost data from d=4).
The 2Q rows use the same box+wedge machinery so Δf(r) in §1 is formed
site-by-site in postprocessing; pair assignment: square pairs = on-axis
sides (length d), tetra pairs = diagonals (length √2·d).

## 5. Implementation structure

- Flux hooks into the P2/P3 measurement pass and REUSES the W_ij trace
  fields already built per (geometry, d, orientation) — marginal cost is
  probe cshifts × wedge sites × 6 planes only. One extra complex field per
  W_ij retained at T_flux (workspace buffer, ~5 MB at 24⁴).
- 6 probe plane fields built once per (config, tdir) from the unsmeared
  field (~32 MB total at 24⁴), reused across all sources and offsets.
- All shifts through the bounded workspace chain (Ca/Cb ping-pong); offsets
  are bounded by box size, so the cshift-plan cache stays bounded.
- **Hard prerequisite: the 24⁴ device-memory census** (--device-mem 32768
  vs 128 MB default). Flux adds working set on top of off-axis; it does NOT
  enter production until the census gate passes and the flag is standard.
- Same multilevel/multihit copies structure: F, W̄, □̄ accumulated per hit
  on the same field state, then averaged — the linearity argument that
  carries W_ij through hits/blocks applies to F unchanged.

## 6. Output schema (per measurement record)

```
"flux4q": {
  "Gsq_d4": {
    "T": 4, "probe_t": 2, "margin": 3,
    "sites":  [[2rx, 2ry, 2rz, weight], ...],      # doubled center-relative
    "planes": ["E1","E2","E3","B1","B2","B3"],     # canonical frame
    "wp": {"AA": [[site][plane]], "BB": ..., "AB": ...},   # ⟨Tr W · □⟩ raw
    "w":  {"AA": re, "BB": re, "AB": re},                  # ⟨Tr W⟩
    "p":  [6 vacuum ⟨□_P⟩]
  },
  "G2q_d4": { ... same shape, single "11" key instead of pairings ... },
  "G2qdiag_d4": { ... }
}
```
Arrays, not per-site JSON keys (a d=4 square is ~50 wedge sites × 6 planes
× 3 matrix elements ≈ 900 floats/record — fine as arrays, hostile as keys).

## 7. Validation gates (in order, each blocks the next)

1. **Stabilizer engine unit tests**: group orders (§3), viewer-wedge
   equality for square/tetra, Σweights == box volume (both parities),
   frame-map round-trip.
2. **8⁴ parity**: F_ij,P per site vs an allocation-naive reference
   implementation; random-gauge-transform invariance (< 1e-11), same
   harness style as tools/test_offaxis_parity.py.
3. **Continuity gate**: 2Q on-axis flux through the NEW box machinery must
   reproduce the EXISTING perpendicular-profile numbers at matching offsets
   on the same configs (this is the regression test tying old and new
   pipelines together).
4. **Physics gates** (production, 16³×32 first — 9804004's own volume):
   large-d square f^(0) → sum of two independent 2Q tube profiles;
   action sum rule Σ_r s(r) vs dV/dR (loose, statistics permitting);
   profile shapes vs 9804004 at β=2.4, R=2,4,6,8.

## 8. Cost estimate (24⁴, per config, after device-mem fix)

Wedge sites per source at m=3: square d=4 ≈ 53, d=8 ≈ 137; tetra d=4 ≈ 55;
2Q rows ≈ 30–90. Full §4 set ≈ 700 wedge sites × 6 planes × (≤3 matrix
elements) ≈ 8k shift+multiply+sum passes per tdir — comparable to one
on-axis loop scan; the W_ij source fields dominate and are shared with the
potential measurement. Flux stays a per-run opt-in flag (--flux4q 1) with
per-source-class selection, mirroring --offaxis.

## 9. Phasing

- **F1** — stabilizer/wedge/weights engine + unit tests (pure python, no
  lattice; lands independently).
- **F2** — box flux for 2Q on-axis + diagonal, continuity gate vs existing
  profiles. (Depends on P1 connectors; already merged machinery.)
- **F3** — square 4Q flux matrix + postprocessing contraction (ground +
  excited) + Δf maps. (Depends on P2 square W_ij fields.)
- **F4** — tetra 4Q flux. (Depends on P3; beyond published PGM.)
- Dashboard heat-map panel: separate task, fed by the §6 schema.

## 10. Open questions for Petrus

1. Flux size set: even-d PGM set (2,4,6,8 square; 2,4 tetra) as specced,
   or extend? (Potentials still cover every d.)
2. Probe margin m = 3 enough, or match a specific reach from the 90s runs?
3. T_flux = 4 with probe at T/2 = 2 (PGM convention), or also store T = 6
   for a contamination cross-check at ~2× flux cost?
4. Excited-state flux f^(1) is free in postprocessing — treat as a
   first-class deliverable (flip-flop map), or defer?
5. Tetra flux has no published PGM reference — gates rely on internal
   consistency (large-d factorization + sum rule). Comfortable, or do you
   want a 16³×32 square-flux reproduction milestone first as the anchor?

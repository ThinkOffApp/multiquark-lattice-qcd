# PGM off-axis 2Q separations and 4Q geometries — source extraction

Extracted 2026-07-08 from the arXiv LaTeX sources of the Green / Michael /
Pennanen (PGM) SU(2) series, for §8 of `4q_measurement_spec.md`. Quotes are
verbatim from the sources; everything not directly verifiable is flagged at
the end. Non-PGM candidate hep-lat/9509064 (de Forcrand et al., cooling) was
checked and discarded; hep-lat/9709124 and hep-lat/9804003 are model papers
with no new lattice vector lists (9709124 confirms the fitted data set, see
below).

## hep-lat/9209007 — UKQCD (Booth, Hulsebos, Irving, McKerrell, Michael, Spencer, Stephenson), "SU(2) Potentials from Large Lattices", Nucl.Phys. B394 (1993) 509

β=2.85, 48³×56. Section "Off-axis potentials": "We choose vector separations
(R_x,R_y,R_z) with 0 ≤ R_i ≤ 3a. For the 2 and 3 dimensional paths, we sum
over all 2 (6) symmetric routes along the edges of the 2 (3) dimensional
hypercuboid." Operators via Albanese-type blocking, c=2, 100 iterations,
8 configs.

Table 3 vector list (R_x,R_y,R_z)/a — every distinct cubic class with
components ≤ 3: (1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,1,0), (2,1,1),
(2,2,0), (2,2,1), (2,2,2), (3,0,0), (3,1,0), (3,1,1), (3,2,0), (3,2,1),
(3,2,2), (3,3,0), (3,3,1), (3,3,2), (3,3,3). On-axis companion table: even
R = 2..24.

## hep-lat/9209019 — Green, Michael, Paton, "Multi-quark energies in QCD", Nucl.Phys. A554 (1993) 701

β=2.4, 16³×32. Off-axis v measured "directly by Monte Carlo simulation using
L-shaped separations of the two static sources", only "up to (x/a=3, y/a=3)
for z/a=0"; for x/a ≥ 4 off-axis values from the lattice-Coulomb
parametrization v_L = −(e/r)_L + b_S·r + v_0 (e=0.234, b_S=0.0736, v_0=0.542),
normalized at r=(6,0,0).

Table 1 measured off-axis (x,y) [z=0]: (1,1), (2,1), (2,2), (3,1), (3,2),
(3,3); on-axis (1,0)–(7,0). Off-axis paths: single fuzzing level 16, c=4.
4Q (Tables 2, 4): planar squares/rectangles (d,r) = (1,1),(1,2),(1,3),(1,4),
(2,2),(2,3),(3,3),(3,4),(4,4),(5,5),(6,6),(7,7); 2×2 basis {A,B}.

## hep-lat/9301006 — Green, Michael, Paton, Sainio, "Multi-quark Energies in SU(2) Lattice Gauge Theory", Int.J.Mod.Phys. E2 (1993) 479

β=2.4, 16³×32 (720 meas.); scaling check β=2.5, 24³×32. "The off-axis
results (y ≠ 0) correspond to L-shaped separations of the static sources."
Ground E₀ AND first-excited E₁ tabulated.

Table 1 measured (x,y) [z=0]: on-axis (1,0)…(6,0); off-axis (1,1), (2,1),
(2,2), (3,1), (3,2), (3,3), (4,1), (4,2), (4,3), (5,1), (5,2), (5,3), (6,1),
(6,2), (6,3) — y = 1,2,3 for x ≤ 6. ((4,4),(5,5),(6,6),(7,x),(8,0),(9,0)
entries are parametrization-only, no MC value.)

4Q: squares (1,1)–(7,7) on both lattices (Tables 2–3); rectangles (Table 4);
colinear (Table 5): (r,d) = (2,1),(3,1),(4,1),(3,2),(4,2),(5,2),(4,3),(5,3),
(6,3).

## hep-lat/9404004 — Green, Michael, Sainio, "Four-quark Binding Energies from SU(2) Lattice Monte Carlo", Z.Phys. C67 (1995) 291 — the "six geometries" paper

β=2.4, 16³×32.

**Path construction (Sec. 3.1, verbatim):** on-axis = straight fuzzed paths
at 3 fuzzing levels (12, 16, 20; c=4) → 3×3 variational basis. Planar
off-axis V₂(x,y): "the two paths P_i(a,b) shown in fig. 2 are combined with
equal weight, P_i(x,y) = ½[P_i(a) + P_i(b)]" — the symmetrized L-shaped
pair. 3D V₂(x,y,z): "combinations around the sides of a cuboid",
P_i(1→4,xyz) = ⅓[P_i(x)P_i(yz) + P_i(y)P_i(xz) + P_i(z)P_i(xy)], each 2D
factor itself the ½(a+b) average — effectively 6 symmetric cuboid-edge
routes.

Table 5 (2Q; explicitly "only a few of the 2-quark potentials produced in
this work"): on-axis (1,0)…(8,0); off-axis (1,1,0), (1,1,1), (1,2,0),
(1,2,1), (1,3,0), (1,3,1), (1,4,0), (1,4,1).
Table 6 (rotational-invariance pairs, additional measured vectors): (4,3,0)
vs (5,0,0); (2,1,2) vs (3,0,0); (3,3,0); (1,4,1); (4,1,0); (2,3,2). Text
also cites V₂(3,0,0) ≈ V₂(2,2,1).

4Q geometries (two "mesons" of equal length d):
- Rectangles (d,r) = (1,1),(1,2),(1,3),(1,4),(2,2),(2,3),(2,4),(3,3),(3,4),
  (4,4); Large Squares add (5,5),(6,6),(7,7) — Table 7. 2×2 basis.
- Tilted Rectangles — Table 8, d,(x,y): 2,(1,1); 2,(2,1); 3,(2,2); 3,(3,1);
  4,(3,2); 4,(4,1); 5,(4,3); 5,(5,1); 6,(5,3); 6,(6,1). 5,(4,3) is a 5×5
  square tilted off-axis (rotational-invariance test vs on-axis 5×5).
- Linear (d,r) = (1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3) —
  Table 9.
- Quadrilateral: same (d,r) set — Table 9; one meson rotated π/2 in-plane.
- Non-Planar: all 16 combinations (1,1)–(4,4) — Table 10; one meson rotated
  π/2 out of plane (diagonals give the (2,1,2), (2,3,2), (1,4,1) entries of
  Table 6). 3×3 basis {A,B,C} for Linear, Quadrilateral, Non-Planar.

## hep-lat/9412029 — Green, Lukkarinen, Pennanen, Michael, Furui (Lattice'94), tetrahedral geometry

β=2.4. Verbatim: "the four quarks have the coordinates (0,0,0), (r,0,d),
(0,d,d) and (r,d,0)" — tetrahedron when r=d. Table 1: (d,r) = (1,0),(1,1),
(1,2),(2,1),(2,2),(2,3),(3,2),(3,3),(3,4),(4,3),(4,4),(4,5).
For r=d these are the same four sites as the spec §2 tetra row.

## hep-lat/9508002 — Green, Lukkarinen, Pennanen, Michael, "A Study of Degenerate Four-quark states...", Phys.Rev. D53 (1996) 261

β=2.4, 16³×32. Square of side d parallel to yz-plane at distance r. Runs:
(i) r = d, d±1 for d=2..5 (d=1: r=1,2), 2208 meas.; (ii) r=1, d=2..5,
3008 meas.; (iii) squares/rectangles re-run with full 3×3 basis.
Path statement (Sec. 2): for V(x,y) "the appropriate path of links is then
constructed as the average of the two most simple paths connecting x and y —
each consisting of one straight section along the x and y axes."

2Q values quoted (Tables 4–5, V₁ columns), planar (x,y): (1,1)=0.4885(1),
(2,2)=0.6689(4), (3,3)=0.7974(8), (4,4)=0.9102(15), (5,5)=1.017(2);
(2,1)=0.6023(3)/0.6021(3), (3,1)=0.6992(5), (4,1)=0.7841(10),
(5,1)=0.8652(8), (3,2)=0.7421(6), (4,3)=0.8589(11), (5,4)=0.967(2),
(6,5)=1.072(4). These double as parity targets for P1 at 16³×32 β=2.4.

## hep-lat/9608147 — Pennanen, "Continuum extrapolation of energies of a four-quark system", Phys.Rev. D55 (1997) 3958

Squares (2a…6a) and tilted rectangles at β=2.35, 2.4 (16³×32), 2.45
(20³×32), 2.5 (24³×32), 2.55 (26³×32); 2-body-only at β=2.3. On-axis fits
r/a=2..6. "29" two-body potentials enter the TR analysis; trcs.eps labels
(x,y,z) = (3,2,0), (3,3,0), (5,4,3), (6,6,1) [figure labels, not a table].
betr.eps TR set at β=2.5: (3,2,2), (3,3,1), (4,3,2), (4,4,1), (5,4,3),
(5,5,1), (6,5,3), (6,6,1) in d,(x,y) form. The full 29-vector list is NOT
printed in the paper (text refers to the companion for more).

## Flux-distribution papers

- hep-lat/9610011 (Green, Michael, Spencer, PRD 55 (1997) 1216): 2Q on-axis
  only, R = 2, 4, 6, 8 (ground A₁g and excited E_u).
- hep-lat/9705033 / 9708012 (Pennanen, Green, Michael): 2Q on-axis only,
  R = 2,3,4,6,8 (β=2.4), R = 2,3,4,6,12 (β=2.5).
- hep-lat/9804004 (Pennanen, Green, Michael, PRD 59 (1999) 014504), β=2.4,
  verbatim: "The quark distances we measured were R=2,4,6,8. For all these
  values... a) two quarks on a lattice axis separated by R lattice units,
  b) two quarks on an axis diagonal with respect to the lattice axis and
  separated by √2·R units [(2,2,0),(4,4,0),(6,6,0),(8,8,0)] and c) four
  quarks at the corners of a square with side length R." Diagonal tubes
  measured directly rather than interpolated, citing measured
  rotational-invariance violations.
- hep-lat/9709124 (model paper) confirms the fitted data set: "15 Tetrahedra,
  6 Squares, 12 Rectangles (including Tilted), 4 Quadrilaterals, 9 Non-Planar
  and 4 Linear", 100 energies (E₀,E₁), 16³×32, β=2.4, a=0.119(1) fm; configs
  with flux links shorter than 2a excluded.

## Caveats / not directly verified

- Exact quark coordinates for Tilted Rectangles, Quadrilateral, Non-Planar
  in 9404004 are defined only via its Fig. 1 (figures absent from the LaTeX
  source); the side/diagonal identifications above are reconstructed from
  the in-text checks ((5,0,0)↔(4,3,0), (3,0,0)↔(2,1,2), path lengths
  a + √13·a) and 9608147's figure macros. The measured V₂ vectors themselves
  are verbatim from Tables 5–6.
- 9608147's complete 29-vector two-body list and 9508002's Fig. 1 were not
  readable as text; labels quoted from EPS internals where noted.
- 9404004 states Table 5 is a subset of all 2Q potentials produced; no
  complete list exists in any of these papers. Original offline run lists,
  if they survive, supersede this reconstruction.

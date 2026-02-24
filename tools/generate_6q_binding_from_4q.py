#!/usr/bin/env python3
"""
Build a model 6Q 3D binding visualization from reconstructed 4Q data.

This is a geometry + profile-calibrated proxy, not a direct 6Q lattice measurement.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class ProfileModel:
    radii: np.ndarray
    values: np.ndarray
    sigma: float
    center_amp: float

    def strength(self, distance: float) -> float:
        d = float(max(0.0, distance))
        r_eval = 0.5 * d
        v = np.interp(r_eval, self.radii, self.values, left=self.values[0], right=self.values[-1])
        floor = float(self.values[-1])
        num = max(0.0, float(v) - floor)
        den = max(1.0e-12, float(self.center_amp) - floor)
        return num / den


def load_reconstructed_profile(csv_path: Path) -> ProfileModel:
    xs: list[float] = []
    ys: list[float] = []
    vals: list[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            vals.append(float(row["action_density"]))
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    z = np.asarray(vals, dtype=float)
    cx = 0.5 * (float(x.min()) + float(x.max()))
    cy = 0.5 * (float(y.min()) + float(y.max()))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    nbin = 80
    r_edges = np.linspace(0.0, float(r.max()) + 1.0e-9, nbin + 1)
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])
    profile = np.empty(nbin, dtype=float)
    for i in range(nbin):
        m = (r >= r_edges[i]) & (r < r_edges[i + 1])
        profile[i] = float(z[m].mean()) if np.any(m) else (profile[i - 1] if i > 0 else float(z.mean()))

    baseline = float(profile[-1])
    w = np.clip(z - baseline, 0.0, None)
    if float(w.sum()) > 0.0:
        mean_r2 = float((w * (r**2)).sum() / w.sum())
        sigma = max(0.25, np.sqrt(mean_r2 / 2.0))
    else:
        sigma = 0.9
    return ProfileModel(
        radii=r_mid,
        values=profile,
        sigma=float(sigma),
        center_amp=float(profile[0]),
    )


def octahedron_vertices(nn_distance: float) -> np.ndarray:
    a = float(nn_distance) / np.sqrt(2.0)
    return np.array(
        [
            [a, 0.0, 0.0],
            [-a, 0.0, 0.0],
            [0.0, a, 0.0],
            [0.0, -a, 0.0],
            [0.0, 0.0, a],
            [0.0, 0.0, -a],
        ],
        dtype=float,
    )


def nearest_neighbor_edges(vertices: np.ndarray) -> list[tuple[int, int]]:
    n = vertices.shape[0]
    dmat = np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=2)
    d_sorted = np.sort(dmat[np.triu_indices(n, 1)])
    d0 = float(d_sorted[0])
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(vertices[i] - vertices[j]) <= d0 * 1.01:
                edges.append((i, j))
    return edges


def distance_to_segment_sq(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ab = b - a
    ab2 = float(np.dot(ab, ab))
    ap = points - a
    t = np.clip((ap @ ab) / ab2, 0.0, 1.0)
    closest = a + t[:, None] * ab[None, :]
    d2 = np.sum((points - closest) ** 2, axis=1)
    return d2, t


def sixq_binding_field(profile: ProfileModel, nn_distance: float, ngrid: int = 36) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    vertices = octahedron_vertices(nn_distance)
    rlim = float(np.max(np.abs(vertices))) * 1.45
    axis = np.linspace(-rlim, rlim, ngrid)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="xy")
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    field = np.zeros(pts.shape[0], dtype=float)

    n = vertices.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            a = vertices[i]
            b = vertices[j]
            L = float(np.linalg.norm(b - a))
            amp = profile.strength(L)
            if amp <= 1.0e-9:
                continue
            d2, t = distance_to_segment_sq(pts, a, b)
            width = profile.sigma * (1.0 + 0.06 * L)
            long_taper = np.exp(-((t - 0.5) ** 2) / (2.0 * 0.24**2))
            field += amp * np.exp(-0.5 * d2 / (width**2)) * long_taper

    vol = (axis[1] - axis[0]) ** 3
    e_proxy = float(np.sum(field) * vol)
    return pts, field, e_proxy, vertices


def make_plot(profile_csv: Path, out_png: Path, distances: list[float]) -> None:
    profile = load_reconstructed_profile(profile_csv)
    fig = plt.figure(figsize=(16, 6))
    cmap = plt.cm.inferno
    norm_all = None
    panels: list[tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []
    for d in distances:
        panels.append(sixq_binding_field(profile, d, ngrid=36))
    vmax = max(float(np.max(p[1])) for p in panels)
    vmin = min(float(np.quantile(p[1], 0.94)) for p in panels)
    norm_all = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    for idx, (d, panel) in enumerate(zip(distances, panels), start=1):
        pts, field, e_proxy, vertices = panel
        ax = fig.add_subplot(1, len(distances), idx, projection="3d")

        q = 0.958 if idx == 2 else 0.965
        mask = field >= float(np.quantile(field, q))
        sel_pts = pts[mask]
        sel_val = field[mask]
        ax.scatter(
            sel_pts[:, 0],
            sel_pts[:, 1],
            sel_pts[:, 2],
            c=sel_val,
            cmap=cmap,
            norm=norm_all,
            s=9,
            alpha=0.35,
            linewidths=0.0,
        )

        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c="#2ec4ff", s=52, edgecolors="white", linewidths=0.8)
        for i, j in nearest_neighbor_edges(vertices):
            ax.plot(
                [vertices[i, 0], vertices[j, 0]],
                [vertices[i, 1], vertices[j, 1]],
                [vertices[i, 2], vertices[j, 2]],
                color="#6ec6ff",
                alpha=0.25,
                linewidth=1.0,
            )

        ax.set_title(f"6Q octahedron  d_nn={d:.1f}\nE6 proxy={e_proxy:.3f}", fontsize=11, pad=12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=23, azim=38)
        lim = float(np.max(np.abs(vertices))) * 1.55
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.grid(False)

    sm = matplotlib.cm.ScalarMappable(norm=norm_all, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.025, pad=0.02)
    cbar.set_label("Modeled 6Q binding density (proxy units)")
    fig.suptitle(
        "6Q Binding Clouds at Different Distances\ncalibrated from reconstructed 4Q planar binding profile",
        fontsize=14,
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Model note: profile-driven superposition using 4Q data; this is a visualization proxy, not a direct 6Q lattice observable.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 3D 6Q binding picture from reconstructed 4Q data.")
    parser.add_argument(
        "--input-csv",
        default="images/papers/reconstructed_4q_planar_data.csv",
        help="Path to reconstructed 4Q profile CSV",
    )
    parser.add_argument(
        "--output-png",
        default="images/papers/sixq_binding_3d_from_4q.png",
        help="Path to output PNG",
    )
    parser.add_argument(
        "--distances",
        default="2.5,4.0,5.5",
        help="Comma-separated nearest-neighbor distances for 6Q octahedron panels",
    )
    args = parser.parse_args()

    distances = [float(x.strip()) for x in args.distances.split(",") if x.strip()]
    make_plot(Path(args.input_csv), Path(args.output_png), distances)
    print(f"Saved {args.output_png}")


if __name__ == "__main__":
    main()

"""Shared spatial classification utilities for cavity volume analysis.

Provides the core Delaunay + cKDTree point classification logic
used by both the volume estimator and visualization classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree

DEFAULT_VDW_RADIUS: float = 1.5
DEFAULT_GRID_RESOLUTION: float = 0.1
MAX_BOND_DISTANCE: float = 2.0
SPHERE_U_RESOLUTION: int = 30
SPHERE_V_RESOLUTION: int = 15


def _classify_inside_atoms(
    points: np.ndarray,
    coords: np.ndarray,
    vdw_radii: np.ndarray,
    max_radius: float,
) -> np.ndarray:
    """Determine which points lie inside any atom's VDW sphere.

    Uses vectorized distance computation per neighbor batch for performance.

    Parameters
    ----------
    points : np.ndarray
        (M, 3) array of query points.
    coords : np.ndarray
        (N, 3) array of atomic coordinates.
    vdw_radii : np.ndarray
        (N,) array of van der Waals radii.
    max_radius : float
        Maximum VDW radius (for KDTree query range).

    Returns
    -------
    np.ndarray
        Boolean mask of length M.
    """
    tree = cKDTree(coords)
    nearby_indices = tree.query_ball_point(points, r=max_radius)

    inside_atoms = np.zeros(len(points), dtype=bool)
    for i, indices in enumerate(nearby_indices):
        if not indices:
            continue
        idx = np.array(indices)
        diffs = points[i] - coords[idx]
        dists_sq = np.einsum("ij,ij->i", diffs, diffs)
        if np.any(dists_sq <= vdw_radii[idx] ** 2):
            inside_atoms[i] = True

    return inside_atoms


@dataclass(frozen=True)
class SpatialClassification:
    """Result of classifying random points relative to a molecular convex hull.

    Attributes
    ----------
    hull : ConvexHull
        The convex hull of the atomic coordinates.
    points_in_hull : np.ndarray
        (M, 3) array of points that lie inside the convex hull.
    inside_atoms_mask : np.ndarray
        Boolean mask of length M — True where point is inside a VDW sphere.
    """

    hull: ConvexHull
    points_in_hull: np.ndarray
    inside_atoms_mask: np.ndarray


def classify_points(
    coords: np.ndarray,
    vdw_radii: np.ndarray,
    num_points: int = 100_000,
) -> SpatialClassification:
    """Classify random points as inside atoms or in cavities.

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) array of atomic coordinates.
    vdw_radii : np.ndarray
        (N,) array of van der Waals radii for each atom.
    num_points : int
        Number of random points to sample in the bounding box.

    Returns
    -------
    SpatialClassification
        Frozen dataclass with hull, points inside hull, and classification mask.
    """
    hull = ConvexHull(coords)
    hull_points = coords[hull.vertices]

    max_radius = float(np.max(vdw_radii)) if len(vdw_radii) else DEFAULT_VDW_RADIUS
    min_coords = np.min(hull_points, axis=0) - max_radius
    max_coords = np.max(hull_points, axis=0) + max_radius

    random_points = np.random.uniform(low=min_coords, high=max_coords, size=(num_points, 3))

    delaunay = Delaunay(hull_points)
    inside_hull = delaunay.find_simplex(random_points) >= 0
    points_in_hull = random_points[inside_hull]

    inside_atoms = _classify_inside_atoms(points_in_hull, coords, vdw_radii, max_radius)

    return SpatialClassification(
        hull=hull,
        points_in_hull=points_in_hull,
        inside_atoms_mask=inside_atoms,
    )


def classify_grid_points(
    coords: np.ndarray,
    vdw_radii: np.ndarray,
    grid_resolution: float = DEFAULT_GRID_RESOLUTION,
) -> tuple[float, float, float]:
    """Classify uniform grid points for precise volume estimation.

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) array of atomic coordinates.
    vdw_radii : np.ndarray
        (N,) array of van der Waals radii for each atom.
    grid_resolution : float
        Spacing of the uniform 3D grid.

    Returns
    -------
    tuple[float, float, float]
        (total_volume, atomic_volume, cavity_volume)
    """
    hull = ConvexHull(coords)
    hull_vertices = coords[hull.vertices]

    max_r = float(np.max(vdw_radii))
    min_bounds = np.min(hull_vertices, axis=0) - max_r
    max_bounds = np.max(hull_vertices, axis=0) + max_r

    grid_x = np.arange(min_bounds[0], max_bounds[0], grid_resolution)
    grid_y = np.arange(min_bounds[1], max_bounds[1], grid_resolution)
    grid_z = np.arange(min_bounds[2], max_bounds[2], grid_resolution)
    grid_points = np.vstack(
        np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    ).reshape(3, -1).T

    delaunay = Delaunay(hull_vertices)
    inside_hull_mask = delaunay.find_simplex(grid_points) >= 0
    points_inside = grid_points[inside_hull_mask]

    inside_atoms = _classify_inside_atoms(points_inside, coords, vdw_radii, max_r)

    vol_total = hull.volume
    vol_atom = np.sum(inside_atoms) * (grid_resolution ** 3)
    vol_cavity = vol_total - vol_atom

    return vol_total, vol_atom, vol_cavity

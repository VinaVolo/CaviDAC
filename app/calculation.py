import os
import sys
sys.path.append('..')
import json
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from src.utils.paths import get_project_path


def read_coordinates_from_file(file_path):
    """
    Reads atomic coordinates from a file.

    Parameters
    ----------
    file_path : str
        The path to the file containing atomic coordinates.

    Returns
    -------
    atomic_symbols : list of str
        A list of atomic symbols.
    coordinates : numpy array (n_atoms, 3)
        A 2D numpy array of shape (n_atoms, 3) containing the atomic coordinates
        in Angstroms.
    """

    atomic_symbols = []
    coordinates = []
    with open(file_path) as file:
        for line in file:
            atom_data = line.strip().split()
            atomic_symbols.append(atom_data[0])
            coordinates.append([float(x) for x in atom_data[1:4]])
    return atomic_symbols, np.array(coordinates)


def estimate_internal_volume(atoms, coordinates, grid_resolution=0.1):
    """
    Estimates the internal volume of a molecule given by a .xyz file.

    Parameters
    ----------
    filename : str
        The name of the .xyz file to read from
    grid_resolution : float, optional
        The spacing between points in the grid to use when estimating the volume

    Returns
    -------
    volume_total : float
        The total volume of the convex hull
    volume_atom : float
        The volume of the region inside the convex hull that is not inside any
        atom
    volume_cavity : float
        The volume of the region inside the convex hull that is not inside any
        atom and is not part of the hull itself
    """

    atom_radii = np.array([vdw_radii.get(atom, 1.5) for atom in atoms])

    convex_hull = ConvexHull(coordinates)
    hull_vertices = coordinates[convex_hull.vertices]

    max_vdw_radius = max(vdw_radii.values())
    min_bounds = np.min(hull_vertices, axis=0) - max_vdw_radius
    max_bounds = np.max(hull_vertices, axis=0) + max_vdw_radius

    grid_x = np.arange(min_bounds[0], max_bounds[0], grid_resolution)
    grid_y = np.arange(min_bounds[1], max_bounds[1], grid_resolution)
    grid_z = np.arange(min_bounds[2], max_bounds[2], grid_resolution)
    grid_x, grid_y, grid_z = np.meshgrid(grid_x, grid_y, grid_z)
    grid_points = np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T

    delaunay = Delaunay(hull_vertices)
    points_within_hull = delaunay.find_simplex(grid_points) >= 0
    hull_points = grid_points[points_within_hull]

    kdtree = cKDTree(coordinates)
    indices_within_atoms = kdtree.query_ball_point(hull_points, r=np.max(atom_radii))
    is_inside_atom = np.zeros(len(hull_points), dtype=bool)

    for idx, atom_indices in enumerate(indices_within_atoms):
        point = hull_points[idx]
        for atom_index in atom_indices:
            if np.linalg.norm(point - coordinates[atom_index]) <= atom_radii[atom_index]:
                is_inside_atom[idx] = True
                break

    volume_total = convex_hull.volume
    volume_atom = np.sum(is_inside_atom) * (grid_resolution ** 3)
    volume_cavity = np.sum(~is_inside_atom) * (grid_resolution ** 3)

    return volume_total, volume_atom, volume_cavity


if __name__ == "__main__":

    with open(os.path.join(get_project_path(), 'data', 'vdw', 'vdw_radius.json'), "r") as file:
        vdw_radii = json.load(file)

    filename = os.path.join(get_project_path(), 'data', 'molecules', '3.txt')

    atoms, coordinates = read_coordinates_from_file(filename)

    volume_total, volume_atom, volume_cavity  = estimate_internal_volume(atoms, coordinates)

    print(f"Общий объем выпуклой оболочки: {volume_total:.2f} Å³")
    print(f"Объем, занятый атомами: {volume_atom:.2f} Å³")
    print(f"Объем полости: {volume_cavity:.2f} Å³")

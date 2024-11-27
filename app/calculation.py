import os
import sys
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree
import json

def read_coordinates(filename):
    """
    Read atomic coordinates from a file.

    Parameters
    ----------
    filename : str
        Path to the file containing atomic coordinates.

    Returns
    -------
    atoms : list of str
        Atomic symbols.
    coords : numpy array (n_atoms, 3)
        Atomic coordinates in Angstroms.
    """
    
    atoms = []
    coords = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            atoms.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
    return atoms, np.array(coords)

def estimate_internal_volume(filename, grid_resolution=0.1):
    
    """
    Estimates the internal volume of a molecule given by a .xyz file.

    Parameters
    ----------
    filename : str
        The name of the .xyz file to read from
    azim : int, optional
        The azimuthal angle to use when constructing the convex hull
    elev : int, optional
        The elevation angle to use when constructing the convex hull
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

    atoms, coords = read_coordinates(filename)
    atom_radii = np.array([vdw_radii.get(atom, 1.5) for atom in atoms])

    hull = ConvexHull(coords)
    hull_points = coords[hull.vertices]

    max_radius = max(vdw_radii.values())
    min_coords = np.min(hull_points, axis=0) - max_radius
    max_coords = np.max(hull_points, axis=0) + max_radius
    
    x = np.arange(min_coords[0], max_coords[0], grid_resolution)
    y = np.arange(min_coords[1], max_coords[1], grid_resolution)
    z = np.arange(min_coords[2], max_coords[2], grid_resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    grid_points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

    delaunay = Delaunay(hull_points)
    inside_hull = delaunay.find_simplex(grid_points) >= 0
    points_in_hull = grid_points[inside_hull]

    tree = cKDTree(coords)
    max_atom_radius = max(atom_radii)
    indices = tree.query_ball_point(points_in_hull, r=max_atom_radius)
    inside_atom = np.zeros(len(points_in_hull), dtype=bool)

    for i, inds in enumerate(indices):
        point = points_in_hull[i]
        for j in inds:
            distance = np.linalg.norm(point - coords[j])
            if distance <= atom_radii[j]:
                inside_atom[i] = True
                break

    volume_total = hull.volume
    volume_atom = np.sum(inside_atom) * (grid_resolution ** 3)
    volume_cavity = np.sum(~inside_atom) * (grid_resolution ** 3)

    return (volume_total, volume_atom, volume_cavity)

if __name__ == "__main__":

    # Глобальное определение ван-дер-ваальсовых радиусов
    vdw_radii = {
        'H': 1.2,
        'C': 1.7,
        'O': 1.52,
        'Br': 1.85,
        'S': 1.80,
        'N': 1.55,
        'Na': 2.27
    }


    filename = './data/3.txt'  # Укажите путь к вашему файлу с координатами

    volume_total, volume_atom, volume_cavity  = estimate_internal_volume(filename)

    print(f"Общий объем выпуклой оболочки: {volume_total:.2f} Å³")
    print(f"Объем, занятый атомами: {volume_atom:.2f} Å³")
    print(f"Объем полости: {volume_cavity:.2f} Å³")

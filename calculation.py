import os
import sys
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from visualization import visualize_molecule, visualize_molecule_with_vdw, visualize_convex_hull, visualize_cavity

def read_coordinates(filename):
    atoms = []
    coords = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                atom = parts[0]
                x, y, z = map(float, parts[1:4])
                atoms.append(atom)
                coords.append([x, y, z])
    return atoms, np.array(coords)

def estimate_internal_volume(filename, azim=45, elev=30, num_points=1000000):

    vdw_radii = {
        'H': 1.2,
        'C': 1.7,
        'O': 1.52,
        'Br': 1.85,
        'S': 1.80,
        'N': 1.55,
        'Na': 2.27
        }
    
    atoms, coords = read_coordinates(filename)

    fig1 = visualize_molecule(atoms, coords, azim=azim, elev=elev)
    fig2 = visualize_molecule_with_vdw(atoms, coords, azim=azim, elev=elev)

    hull = ConvexHull(coords)

    fig3 = visualize_convex_hull(atoms, coords, hull, azim=azim, elev=elev)

    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    random_points = np.random.uniform(min_coords, max_coords, size=(num_points, 3))
    delaunay = Delaunay(coords[hull.vertices])
    inside_hull = delaunay.find_simplex(random_points) >= 0
    points_in_hull = random_points[inside_hull]

    atom_radii = np.array([vdw_radii.get(atom, 1.5) for atom in atoms])
    coords_expanded = coords[:, np.newaxis, :]
    radii_expanded = atom_radii[:, np.newaxis]

    num_points_in_hull = len(points_in_hull)
    num_points_in_atoms = 0

    batch_size = 100000
    points_in_cavity = []

    for i in range(0, num_points_in_hull, batch_size):
        batch_points = points_in_hull[i:i+batch_size]
        batch_points_expanded = batch_points[np.newaxis, :, :]
        distances = np.linalg.norm(coords_expanded - batch_points_expanded, axis=2)
        inside_any_atom = np.any(distances <= radii_expanded, axis=0)
        num_points_in_atoms += np.sum(inside_any_atom)
        cavity_points = batch_points[~inside_any_atom]
        points_in_cavity.append(cavity_points)

    points_in_cavity = np.concatenate(points_in_cavity, axis=0)

    fraction_in_atoms = num_points_in_atoms / num_points_in_hull
    atom_volume = fraction_in_atoms * hull.volume

    cavity_volume = hull.volume - atom_volume

    fig4 = visualize_cavity(atoms, coords, hull, azim=azim, elev=elev, num_points=num_points)

    return (fig1, fig2, fig3, fig4), (hull.volume, atom_volume, cavity_volume)

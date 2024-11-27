import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def visualize_molecule(atoms, coords, azim=45, elev=30):
    fig = plt.Figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')

    colors = {
        'H': 'white',
        'C': 'black',
        'O': 'red',
        'N': 'blue',
        'S': 'yellow',
        'Br': 'brown',
        'Na': 'green'
    }

    for atom, coord in zip(atoms, coords):
        color = colors.get(atom, 'green')
        ax.scatter(coord[0], coord[1], coord[2], color=color, s=50)

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title('Молекула без ван-дер-ваальсовых радиусов')

    ax.view_init(azim=azim, elev=elev)
    return fig

def visualize_molecule_with_vdw(atoms, coords, azim=45, elev=30):
    fig = plt.Figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')

    colors = {
        'H': 'white',
        'C': 'black',
        'O': 'red',
        'N': 'blue',
        'S': 'yellow',
        'Br': 'brown',
        'Na': 'green'
    }

    vdw_radii = {
        'H': 1.2,
        'C': 1.7,
        'O': 1.52,
        'Br': 1.85,
        'S': 1.80,
        'N': 1.55,
        'Na': 2.27
        }

    for atom, coord in zip(atoms, coords):
        color = colors.get(atom, 'green')
        radius = vdw_radii.get(atom, 1.5)
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = coord[0] + radius * np.cos(u) * np.sin(v)
        y = coord[1] + radius * np.sin(u) * np.sin(v)
        z = coord[2] + radius * np.cos(v)
        ax.plot_surface(x, y, z, color=color, shade=True, alpha=0.6)

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title('Молекула с ван-дер-ваальсовыми радиусами')

    ax.view_init(azim=azim, elev=elev)
    return fig

def visualize_convex_hull(atoms, coords, hull, azim=45, elev=30):
    fig = plt.Figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')

    colors = {
        'H': 'white',
        'C': 'black',
        'O': 'red',
        'N': 'blue',
        'S': 'yellow',
        'Br': 'brown',
        'Na': 'green'
    }

    vdw_radii = {
        'H': 1.2,
        'C': 1.7,
        'O': 1.52,
        'Br': 1.85,
        'S': 1.80,
        'N': 1.55,
        'Na': 2.27
        }

    for atom, coord in zip(atoms, coords):
        color = colors.get(atom, 'green')
        radius = vdw_radii.get(atom, 1.5)
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = coord[0] + radius * np.cos(u) * np.sin(v)
        y = coord[1] + radius * np.sin(u) * np.sin(v)
        z = coord[2] + radius * np.cos(v)
        ax.plot_surface(x, y, z, color=color, shade=True, alpha=0.6)

    simplices = hull.simplices
    for simplex in simplices:
        triangle = coords[simplex]
        tri = Poly3DCollection([triangle], facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.3)
        ax.add_collection3d(tri)

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title('Молекула с выпуклой оболочкой')

    ax.view_init(azim=azim, elev=elev)
    return fig

def visualize_cavity(atoms, coords, hull, azim=45, elev=30, num_points=1000000):
    fig = plt.Figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')

    colors = {
        'H': 'white',
        'C': 'black',
        'O': 'red',
        'N': 'blue',
        'S': 'yellow',
        'Br': 'brown',
        'Na': 'green'
    }

    vdw_radii = {
        'H': 1.2,
        'C': 1.7,
        'O': 1.52,
        'Br': 1.85,
        'S': 1.80,
        'N': 1.55,
        'Na': 2.27
        }

    simplices = hull.simplices
    for simplex in simplices:
        triangle = coords[simplex]
        tri = Poly3DCollection([triangle], facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.2)
        ax.add_collection3d(tri)

    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    random_points = np.random.uniform(min_coords, max_coords, size=(num_points, 3))
    delaunay = Delaunay(coords[hull.vertices])
    inside_hull = delaunay.find_simplex(random_points) >= 0
    points_in_hull = random_points[inside_hull]

    atom_radii = np.array([vdw_radii.get(atom, 1.5) for atom in atoms])
    coords_expanded = coords[:, np.newaxis, :]
    radii_expanded = atom_radii[:, np.newaxis]

    batch_size = 100000
    points_in_cavity = []

    for i in range(0, len(points_in_hull), batch_size):
        batch_points = points_in_hull[i:i+batch_size]
        batch_points_expanded = batch_points[np.newaxis, :, :]
        distances = np.linalg.norm(coords_expanded - batch_points_expanded, axis=2)
        inside_any_atom = np.any(distances <= radii_expanded, axis=0)
        cavity_points = batch_points[~inside_any_atom]
        points_in_cavity.append(cavity_points)

    points_in_cavity = np.concatenate(points_in_cavity, axis=0)

    ax.scatter(points_in_cavity[:, 0], points_in_cavity[:, 1], points_in_cavity[:, 2],
               color='magenta', s=1, alpha=0.6, label='Полость')

    for atom, coord in zip(atoms, coords):
        color = colors.get(atom, 'green')
        radius = vdw_radii.get(atom, 1.5)
        u, v = np.mgrid[0:2*np.pi:8j, 0:np.pi:4j]
        x = coord[0] + radius * np.cos(u) * np.sin(v)
        y = coord[1] + radius * np.sin(u) * np.sin(v)
        z = coord[2] + radius * np.cos(v)
        ax.plot_surface(x, y, z, color=color, shade=True, alpha=0.2)

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title('Внутренняя полость молекулы')

    ax.view_init(azim=azim, elev=elev)
    return fig
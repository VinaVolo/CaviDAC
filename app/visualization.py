import os
import sys
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_simple_3d_molecule(azim=360, elev=-64):
    """
    Plots a simple 3D visualization of a molecule using matplotlib.

    This function creates a 3D scatter plot of atoms in a molecule with specified
    azimuth and elevation angles for the view. Atoms are represented as colored
    points, with colors determined by the `atom_colors` dictionary. A legend
    is generated to label each type of atom present in the visualization.

    Parameters:
    azim (int): Azimuth angle for the 3D plot view. Default is 360.
    elev (int): Elevation angle for the 3D plot view. Default is -64.

    The function displays the plot with a title and hides axis frames for a
    cleaner visualization.
    """

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d', facecolor='whitesmoke')

    for atom, coord in zip(atoms, coords):
        color = atom_colors.get(atom, 'grey')
        ax.scatter(coord[0], coord[1], coord[2], color=color, s=400, edgecolors='k', alpha=0.9)

    unique_atoms = set(atoms)
    legend_elements = []
    for element in unique_atoms:
        color = atom_colors.get(element, 'grey')
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=element,
                                      markerfacecolor=color, markersize=14, markeredgecolor='k'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True, facecolor='white', edgecolor='black')
    # ax.set_xlabel('X координата', fontsize=12, fontweight='bold', labelpad=15)
    # ax.set_ylabel('Y координата', fontsize=12, fontweight='bold', labelpad=15)
    # ax.set_zlabel('Z координата', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_title('3D визуализация молекулы', fontsize=18, fontweight='bold', pad=30)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.set_axis_off()
    # plt.savefig('step_1.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.tight_layout()
    plt.show()


def plot_molecule(azim=360, elev=-64):
    """
    Plots a 3D visualization of a molecule using matplotlib.

    This function creates a 3D scatter plot of atoms in a molecule with specified
    azimuth and elevation angles for the view. Atoms are represented as colored
    points, with colors determined by the `atom_colors` dictionary. A legend
    is generated to label each type of atom present in the visualization.

    Parameters:
    azim (int): Azimuth angle for the 3D plot view. Default is 360.
    elev (int): Elevation angle for the 3D plot view. Default is -64.

    The function displays the plot with a title and hides axis frames for a
    cleaner visualization.
    """

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d', facecolor='whitesmoke')

    for atom, coord in zip(atoms, coords):
        color = atom_colors.get(atom, '#808080')
        radius = vdw_radii.get(atom, 1.5)
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        x = radius * np.cos(u) * np.sin(v) + coord[0]
        y = radius * np.sin(u) * np.sin(v) + coord[1]
        z = radius * np.cos(v) + coord[2]
        ax.plot_surface(x, y, z, color=color, linewidth=2, antialiased=True, shade=True, alpha=0.8)

    for i, coord1 in enumerate(coords):
        for j, coord2 in enumerate(coords):
            if i < j:
                distance = np.linalg.norm(coord1 - coord2)
                if distance < 2.0: 
                    ax.plot([coord1[0], coord2[0]], 
                            [coord1[1], coord2[1]], 
                            [coord1[2], coord2[2]], 
                            color='grey', linewidth=1.5, alpha=0.7)

    unique_atoms = set(atoms)
    legend_elements = []
    for element in unique_atoms:
        color = atom_colors.get(element, '#808080')
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=element,
                                      markerfacecolor=color, markersize=14, markeredgecolor='k'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True, facecolor='white', edgecolor='black')
    ax.set_title('3D визуализация молекулы с ван-дер-ваальсовыми радиусами', fontsize=18, fontweight='bold', pad=30)
    ax.dist = 12
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.set_axis_off()
    # plt.savefig('step_2.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.tight_layout()
    plt.show()


def plot_molecule_with_hull(azim=360, elev=-64):
    """
    Plots a 3D visualization of a molecule with a convex hull using matplotlib.

    This function creates a 3D scatter plot of atoms in a molecule with specified
    azimuth and elevation angles for the view. Atoms are represented as colored
    points, with colors determined by the `atom_colors` dictionary. A legend
    is generated to label each type of atom present in the visualization. The
    function also plots the convex hull of the molecule as a translucent cyan
    surface. The plot is displayed with a title and hides axis frames for a
    cleaner visualization.

    Parameters:
    azim (int): Azimuth angle for the 3D plot view. Default is 360.
    elev (int): Elevation angle for the 3D plot view. Default is -64.
    """

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d', facecolor='whitesmoke')

    for atom, coord in zip(atoms, coords):
        color = atom_colors.get(atom, '#808080')
        radius = vdw_radii.get(atom, 1.5)
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        x = radius * np.cos(u) * np.sin(v) + coord[0]
        y = radius * np.sin(u) * np.sin(v) + coord[1]
        z = radius * np.cos(v) + coord[2]
        ax.plot_surface(x, y, z, color=color, linewidth=0, antialiased=True, shade=True, alpha=0.6)

    hull = ConvexHull(coords)
    for simplex in hull.simplices:
        triangle = coords[simplex]
        x = triangle[:, 0]
        y = triangle[:, 1]
        z = triangle[:, 2]
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        z = np.append(z, z[0])
        ax.plot_trisurf(x, y, z, color='cyan', alpha=0.5, linewidth=0)

    max_radius = max(vdw_radii.values())

    unique_atoms = set(atoms)
    legend_elements = []
    for element in unique_atoms:
        color = atom_colors.get(element, '#808080')
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=element,
                                      markerfacecolor=color, markersize=14, markeredgecolor='k'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True, facecolor='white', edgecolor='black')
    ax.set_title('3D визуализация молекулы с выпуклой оболочкой', fontsize=18, fontweight='bold', pad=30)
    ax.dist = 12
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.set_axis_off()
    # plt.savefig('step_3.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.tight_layout()
    plt.show()


def plot_grid_and_hull(azim=360, elev=-64):

    """
    Plot a 3D visualization of a molecule with its convex hull and a grid of
    points. The function takes two optional parameters: `azim` and `elev`, which
    specify the azimuth and elevation angles for the view. Atoms are represented
    as colored points, with colors determined by the `atom_colors` dictionary.
    A legend is generated to label each type of atom present in the
    visualization. The convex hull of the molecule is plotted as a translucent
    cyan surface. The plot is displayed with a title and hides axis frames for a
    cleaner visualization.

    Parameters:
    azim (int): Azimuth angle for the 3D plot view. Default is 360.
    elev (int): Elevation angle for the 3D plot view. Default is -64.
    """

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d', facecolor='whitesmoke')

    hull = ConvexHull(coords)

    max_radius = max(vdw_radii.values())
    min_coords = np.min(coords[hull.vertices], axis=0) - max_radius
    max_coords = np.max(coords[hull.vertices], axis=0) + max_radius

    x = np.arange(min_coords[0], max_coords[0])
    y = np.arange(min_coords[1], max_coords[1])
    z = np.arange(min_coords[2], max_coords[2])
    X, Y, Z = np.meshgrid(x, y, z)
    grid_points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

    # Отображение атомов в виде сфер с учетом ван-дер-ваальсовых радиусов
    for atom, coord in zip(atoms, coords):
        color = atom_colors.get(atom, '#808080')  # Серый цвет по умолчанию
        radius = vdw_radii.get(atom, 1.5)  # Используем 1.5 Å по умолчанию
        # Создаем сферу
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        x = radius * np.cos(u) * np.sin(v) + coord[0]
        y = radius * np.sin(u) * np.sin(v) + coord[1]
        z = radius * np.cos(v) + coord[2]
        ax.plot_surface(x, y, z, color=color, linewidth=0, antialiased=True, shade=True, alpha=0.6)

    # Отображение выпуклой оболочки
    for simplex in hull.simplices:
        triangle = coords[simplex]
        x_triangle = triangle[:, 0]
        y_triangle = triangle[:, 1]
        z_triangle = triangle[:, 2]
        ax.plot_trisurf(x_triangle, y_triangle, z_triangle, color='cyan', alpha=0.5, linewidth=0)

    unique_atoms = set(atoms)
    legend_elements = []
    for element in unique_atoms:
        color = atom_colors.get(element, '#808080')
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=element,
                                      markerfacecolor=color, markersize=14, markeredgecolor='k'))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Grid of points',
                                  markerfacecolor='blue', markersize=10, markeredgecolor='k'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True, facecolor='white', edgecolor='black')
    ax.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2],
               color='blue', s=5, alpha=0.6, label='Сетка точек')
    ax.set_title('3D визуализация молекулы с сеткой точек и выпуклой оболочкой', fontsize=18, fontweight='bold', pad=30)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.set_axis_off()
    # plt.savefig('step_4.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.tight_layout()
    plt.show()


def plot_points_in_hull(azim=360, elev=-64):

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d', facecolor='whitesmoke')

    grid_resolution = 0.25

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

    for atom, coord in zip(atoms, coords):
        color = atom_colors.get(atom, '#808080')
        radius = vdw_radii.get(atom, 1.5)
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        x = radius * np.cos(u) * np.sin(v) + coord[0]
        y = radius * np.sin(u) * np.sin(v) + coord[1]
        z = radius * np.cos(v) + coord[2]
        ax.plot_surface(x, y, z, color=color, linewidth=0, antialiased=True, shade=True, alpha=0.6)

    for simplex in hull.simplices:
        triangle = coords[simplex]
        x_triangle = triangle[:, 0]
        y_triangle = triangle[:, 1]
        z_triangle = triangle[:, 2]
        ax.plot_trisurf(x_triangle, y_triangle, z_triangle, color='cyan', alpha=0, linewidth=0)

    unique_atoms = set(atoms)
    legend_elements = []
    for element in unique_atoms:
        color = atom_colors.get(element, '#808080')
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=element,
                                      markerfacecolor=color, markersize=14, markeredgecolor='k'))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Points in ConvexHull',
                                  markerfacecolor='blue', markersize=10, markeredgecolor='k'))
    


    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True, facecolor='white', edgecolor='black')
    ax.scatter(points_in_hull[:, 0], points_in_hull[:, 1], points_in_hull[:, 2],
               color='blue', s=1, alpha=0.3, label='Сетка точек')
    ax.set_title('3D визуализация молекулы с точками внутри выпуклой оболочки', fontsize=18, fontweight='bold', pad=30)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    # plt.savefig('step_5.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.tight_layout()
    plt.show()



def plot_molecule_with_points(azim=360, elev=-64):
    """
    Plot a 3D visualization of a molecule with points inside its convex hull
    and within atoms. The function takes two optional parameters: `azim` and
    `elev`, which specify the azimuth and elevation angles for the view.
    Atoms are represented as spheres colored based on the `atom_colors`
    dictionary. Points inside the convex hull are displayed, distinguishing
    between those inside atoms and those in the free cavities. A legend is
    generated to label each type of atom and point category. The plot is
    displayed with a title, and axis frames are hidden for a cleaner
    visualization.

    Parameters:
    azim (int): Azimuth angle for the 3D plot view. Default is 360.
    elev (int): Elevation angle for the 3D plot view. Default is -64.
    """

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d', facecolor='whitesmoke')
    
    grid_resolution = 0.3
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

    for atom, coord in zip(atoms, coords):
        color = atom_colors.get(atom, '#808080')
        radius = vdw_radii.get(atom, 1.5)
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x_sphere = radius * np.cos(u) * np.sin(v) + coord[0]
        y_sphere = radius * np.sin(u) * np.sin(v) + coord[1]
        z_sphere = radius * np.cos(v) + coord[2]
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color=color, alpha=0.6)

    for simplex in hull.simplices:
        triangle = coords[simplex]
        x_triangle = triangle[:, 0]
        y_triangle = triangle[:, 1]
        z_triangle = triangle[:, 2]
        ax.plot_trisurf(x_triangle, y_triangle, z_triangle, color='cyan', alpha=0)

    unique_atoms = set(atoms)
    legend_elements = []
    for element in unique_atoms:
        color = atom_colors.get(element, '#808080')
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=element,
                                      markerfacecolor=color, markersize=14, markeredgecolor='k'))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Points in the free cavities of the molecule',
                                  markerfacecolor='blue', markersize=10, markeredgecolor='k'))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Dots inside atoms',
                                  markerfacecolor='green', markersize=10, markeredgecolor='k'))

    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True, facecolor='white', edgecolor='black')
    ax.scatter(points_in_hull[inside_atom, 0], points_in_hull[inside_atom, 1], points_in_hull[inside_atom, 2],
               color='forestgreen', s=4, alpha=0.2, label='Точки внутри атомов')
    ax.scatter(points_in_hull[~inside_atom, 0], points_in_hull[~inside_atom, 1], points_in_hull[~inside_atom, 2],
               color='blue', s=5, alpha=0.7, label='Точки в полостях')
    ax.set_title('3D визуализация молекулы с точками в полостях и внутри атомов', fontsize=18, fontweight='bold', pad=30)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.set_axis_off()
    # plt.savefig('step_6.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.tight_layout()
    plt.show()
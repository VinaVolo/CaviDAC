import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from abc import ABC, abstractmethod

class MoleculeData:
    def __init__(self, atoms, coords, atom_colors, vdw_radii):
        self.atoms = atoms
        self.coords = coords
        self.atom_colors = atom_colors
        self.vdw_radii = vdw_radii

    def get_unique_atoms(self):
        return set(self.atoms)


class BasePlot(ABC):
    def __init__(self, molecule_data: MoleculeData):
        self.mol = molecule_data

    @abstractmethod
    def plot(self, azim=45, elev=30):
        pass

    def _create_3d_axis(self, title, figsize=(12, 11)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        return fig, ax

    def _add_legend(self, ax):
        unique_atoms = self.mol.get_unique_atoms()
        legend_elements = []
        for element in unique_atoms:
            color = self.mol.atom_colors.get(element, 'grey')
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=element,
                                        markerfacecolor=color, markersize=12, markeredgecolor='k'))  # Увеличен размер маркеров

        ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=12,  
            frameon=True,
            facecolor='white',
            edgecolor='black',
            markerscale=0.5 
        )


    def _style_axis(self, ax, azim, elev, axis_labels=False):
        ax.view_init(elev=elev, azim=azim)
        ax.grid(False)
        ax.set_axis_off()

        if not axis_labels:
            ax.xaxis.pane.set_edgecolor('w')
            ax.yaxis.pane.set_edgecolor('w')
            ax.zaxis.pane.set_edgecolor('w')

    def _plot_atoms_as_points(self, ax):
        # Отображение атомов в виде точек
        for atom, coord in zip(self.mol.atoms, self.mol.coords):
            color = self.mol.atom_colors.get(atom, 'grey')
            ax.scatter(coord[0], coord[1], coord[2], color=color, s=400, edgecolors='k', alpha=0.9)


class SimpleMoleculePlot(BasePlot):
    def plot(self, azim=45, elev=30):
        fig, ax = self._create_3d_axis('3D визуализация молекулы')
        self._plot_atoms_as_points(ax)
        self._add_legend(ax)
        self._style_axis(ax, azim, elev)
        plt.tight_layout()
        return fig


class VDWMoleculePlot(BasePlot):
    def plot(self, azim=45, elev=30):
        fig, ax = self._create_3d_axis('3D визуализация с ван-дер-ваальсовыми радиусами')

        # Атомы как сферы
        for atom, coord in zip(self.mol.atoms, self.mol.coords):
            color = self.mol.atom_colors.get(atom, '#808080')
            radius = self.mol.vdw_radii.get(atom, 1.5)
            u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
            x = radius * np.cos(u) * np.sin(v) + coord[0]
            y = radius * np.sin(u) * np.sin(v) + coord[1]
            z = radius * np.cos(v) + coord[2]
            ax.plot_surface(x, y, z, color=color, linewidth=2, antialiased=True, shade=True, alpha=0.8)

        # Связи между близкими атомами
        for i, coord1 in enumerate(self.mol.coords):
            for j, coord2 in enumerate(self.mol.coords):
                if i < j:
                    distance = np.linalg.norm(coord1 - coord2)
                    if distance < 2.0:
                        ax.plot([coord1[0], coord2[0]], 
                                [coord1[1], coord2[1]], 
                                [coord1[2], coord2[2]], 
                                color='grey', linewidth=1.5, alpha=0.7)

        self._add_legend(ax)
        self._style_axis(ax, azim, elev)
        plt.tight_layout()
        return fig


class HullMoleculePlot(BasePlot):
    def plot(self, azim=45, elev=30):
        fig, ax = self._create_3d_axis('3D визуализация молекулы с выпуклой оболочкой')

        for atom, coord in zip(self.mol.atoms, self.mol.coords):
            color = self.mol.atom_colors.get(atom, '#808080')
            radius = self.mol.vdw_radii.get(atom, 1.5)
            u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
            x = radius * np.cos(u) * np.sin(v) + coord[0]
            y = radius * np.sin(u) * np.sin(v) + coord[1]
            z = radius * np.cos(v) + coord[2]
            ax.plot_surface(x, y, z, color=color, linewidth=0, antialiased=True, shade=True, alpha=0.6)

        hull = ConvexHull(self.mol.coords)
        for simplex in hull.simplices:
            triangle = self.mol.coords[simplex]
            x = triangle[:, 0]
            y = triangle[:, 1]
            z = triangle[:, 2]
            ax.plot_trisurf(x, y, z, color='cyan', alpha=0.5, linewidth=0)

        self._add_legend(ax)
        self._style_axis(ax, azim, elev)
        plt.tight_layout()
        return fig


class GridHullPlot(BasePlot):
    def __init__(self, molecule_data: MoleculeData, grid_resolution=0.1):
        super().__init__(molecule_data)
        self.grid_resolution = grid_resolution

    def plot(self, azim=45, elev=30):
        fig, ax = self._create_3d_axis('3D молекула с сеткой точек и выпуклой оболочкой')

        number_of_points = 20000

        hull = ConvexHull(self.mol.coords)
        max_radius = max(self.mol.vdw_radii.values())
        min_coords = np.min(self.mol.coords[hull.vertices], axis=0) - max_radius
        max_coords = np.max(self.mol.coords[hull.vertices], axis=0) + max_radius

        cube_points = np.random.uniform(low=min_coords, high=max_coords, size=(number_of_points, 3))

        ax.scatter(cube_points[:, 0], cube_points[:, 1], cube_points[:, 2],
               color='blue', s=1, alpha=0.6, label='Случайные точки')

        for atom, coord in zip(self.mol.atoms, self.mol.coords):
            color = self.mol.atom_colors.get(atom, '#808080')
            radius = self.mol.vdw_radii.get(atom, 1.5)
            u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
            x_sphere = radius * np.cos(u) * np.sin(v) + coord[0]
            y_sphere = radius * np.sin(u) * np.sin(v) + coord[1]
            z_sphere = radius * np.cos(v) + coord[2]
            ax.plot_surface(x_sphere, y_sphere, z_sphere, color=color, linewidth=0, shade=True, alpha=0.6)

        for simplex in hull.simplices:
            triangle = self.mol.coords[simplex]
            x_triangle = triangle[:, 0]
            y_triangle = triangle[:, 1]
            z_triangle = triangle[:, 2]
            ax.plot_trisurf(x_triangle, y_triangle, z_triangle, color='cyan', alpha=0.5, linewidth=0)

        unique_atoms = self.mol.get_unique_atoms()
        legend_elements = []
        for element in unique_atoms:
            color = self.mol.atom_colors.get(element, '#808080')
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=element,
                                          markerfacecolor=color, markersize=14, markeredgecolor='k'))
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Grid of points',
                                      markerfacecolor='blue', markersize=10, markeredgecolor='k'))
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True, facecolor='white', edgecolor='black')

        self._style_axis(ax, azim, elev)
        plt.tight_layout()
        return fig


class PointsInHullPlot(BasePlot):
    def __init__(self, molecule_data: MoleculeData, grid_resolution=0.25):
        super().__init__(molecule_data)
        self.grid_resolution = grid_resolution

    def plot(self, azim=45, elev=30):
        fig, ax = self._create_3d_axis('3D молекула с точками внутри выпуклой оболочки')

        number_of_points = 80000

        hull = ConvexHull(self.mol.coords)
        hull_points = self.mol.coords[hull.vertices]
        max_radius = max(self.mol.vdw_radii.values())
        min_coords = np.min(hull_points, axis=0) - max_radius
        max_coords = np.max(hull_points, axis=0) + max_radius

        random_points = np.random.uniform(low=min_coords, high=max_coords, size=(number_of_points, 3))

        delaunay = Delaunay(hull_points)
        inside_hull = delaunay.find_simplex(random_points) >= 0
        points_in_hull = random_points[inside_hull]

        # Точки внутри оболочки
        ax.scatter(points_in_hull[:, 0], points_in_hull[:, 1], points_in_hull[:, 2],
               color='cyan', s=3, alpha=0.8, label='Случайные точки внутри выпуклой оболочки')


        # Атомы
        for atom, coord in zip(self.mol.atoms, self.mol.coords):
            color = self.mol.atom_colors.get(atom, '#808080')
            radius = self.mol.vdw_radii.get(atom, 1.5)
            u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
            x_sphere = radius * np.cos(u) * np.sin(v) + coord[0]
            y_sphere = radius * np.sin(u) * np.sin(v) + coord[1]
            z_sphere = radius * np.cos(v) + coord[2]
            ax.plot_surface(x_sphere, y_sphere, z_sphere, color=color, linewidth=0, shade=True, alpha=0.6)

        # Выпуклая оболочка
        for simplex in hull.simplices:
            triangle = self.mol.coords[simplex]
            ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                            color='cyan', alpha=0, linewidth=0)

        # Легенда
        unique_atoms = self.mol.get_unique_atoms()
        legend_elements = []
        for element in unique_atoms:
            color = self.mol.atom_colors.get(element, '#808080')
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=element,
                                          markerfacecolor=color, markersize=14, markeredgecolor='k'))
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Points in ConvexHull',
                                      markerfacecolor='blue', markersize=10, markeredgecolor='k'))
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True, facecolor='white', edgecolor='black')

        self._style_axis(ax, azim, elev)
        plt.tight_layout()
        return fig


class PointsInAtomsPlot(BasePlot):
    def __init__(self, molecule_data: MoleculeData, grid_resolution=0.3):
        super().__init__(molecule_data)
        self.grid_resolution = grid_resolution

    def plot(self, azim=45, elev=30):
        fig, ax = self._create_3d_axis('3D визуализация: точки внутри атомов и в полостях')

        number_of_points = 100000

        hull = ConvexHull(self.mol.coords)
        hull_points = self.mol.coords[hull.vertices]
        max_radius = max(self.mol.vdw_radii.values())
        min_coords = np.min(hull_points, axis=0) - max_radius
        max_coords = np.max(hull_points, axis=0) + max_radius

        random_points = np.random.uniform(low=min_coords, high=max_coords, size=(number_of_points, 3))

        delaunay = Delaunay(hull_points)
        inside_hull = delaunay.find_simplex(random_points) >= 0
        points_in_hull = random_points[inside_hull]

        atom_radii = np.array([self.mol.vdw_radii.get(atom, 1.5) for atom in self.mol.atoms])
        tree = cKDTree(self.mol.coords)
        max_atom_radius = max(atom_radii)
        indices = tree.query_ball_point(points_in_hull, r=max_atom_radius)

        inside_atom = np.zeros(len(points_in_hull), dtype=bool)
        for i, inds in enumerate(indices):
            point = points_in_hull[i]
            for j in inds:
                distance = np.linalg.norm(point - self.mol.coords[j])
                if distance <= atom_radii[j]:
                    inside_atom[i] = True
                    break

        # Точки внутри атомов (зеленые)
        ax.scatter(points_in_hull[inside_atom, 0],
                   points_in_hull[inside_atom, 1],
                   points_in_hull[inside_atom, 2],
                   color='green', s=4, alpha=0.2, label='Точки внутри атомов')

        # Точки в полостях (синие)
        ax.scatter(points_in_hull[~inside_atom, 0],
                   points_in_hull[~inside_atom, 1],
                   points_in_hull[~inside_atom, 2],
                   color='blue', s=5, alpha=0.7, label='Точки в полостях')

        # Атомы
        for atom, coord in zip(self.mol.atoms, self.mol.coords):
            color = self.mol.atom_colors.get(atom, '#808080')
            radius = self.mol.vdw_radii.get(atom, 1.5)
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x_sphere = radius * np.cos(u) * np.sin(v) + coord[0]
            y_sphere = radius * np.sin(u) * np.sin(v) + coord[1]
            z_sphere = radius * np.cos(v) + coord[2]
            ax.plot_surface(x_sphere, y_sphere, z_sphere, color=color, alpha=0.6)

        # Выпуклая оболочка
        for simplex in hull.simplices:
            triangle = self.mol.coords[simplex]
            ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], color='cyan', alpha=0)

        # Легенда
        unique_atoms = self.mol.get_unique_atoms()
        legend_elements = []
        for element in unique_atoms:
            color = self.mol.atom_colors.get(element, '#808080')
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=element,
                                          markerfacecolor=color, markersize=12, markeredgecolor='k'))
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Points in free cavities',
                                      markerfacecolor='blue', markersize=12, markeredgecolor='k'))
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Dots inside atoms',
                                      markerfacecolor='green', markersize=12, markeredgecolor='k'))

        ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=12,  
            frameon=True,
            facecolor='white',
            edgecolor='black',
            markerscale=0.5 
        )

        self._style_axis(ax, azim, elev)
        plt.tight_layout()
        return fig


# Пример использования:
# Создаем объект MoleculeData:
# mol_data = MoleculeData(atoms, coords, atom_colors, vdw_radii)

# Создаем объект для каждого типа графика и строим:
# simple_plot = SimpleMoleculePlot(mol_data)
# fig = simple_plot.plot(azim=30, elev=45)
# fig.savefig('simple_plot.png')

# Таким образом мы можем достать любой график независимо друг от друга.

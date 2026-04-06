"""3D molecular visualization using Matplotlib.

Provides multiple plot types for rendering molecules with van der Waals spheres,
convex hulls, and cavity point classifications.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from abc import ABC, abstractmethod
from scipy.spatial import ConvexHull, Delaunay, cKDTree

from cavidac.constants import (
    DEFAULT_VDW_RADIUS,
    MAX_BOND_DISTANCE,
    SPHERE_U_RESOLUTION,
    SPHERE_V_RESOLUTION,
)
from cavidac.geometry.spatial import classify_points
from cavidac.visualization.molecule_data import MoleculeData


class BasePlot(ABC):
    def __init__(self, molecule_data: MoleculeData) -> None:
        self.mol = molecule_data

    @abstractmethod
    def plot(self, azim: float = 45, elev: float = 30) -> plt.Figure:
        """Create a 3D visualization of the molecule."""
        pass

    def _create_3d_axis(
        self, figsize: tuple[float, float] = (6, 5)
    ) -> tuple[plt.Figure, plt.Axes]:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        return fig, ax

    def _plot_atoms_as_points(self, ax: plt.Axes) -> None:
        for atom, coord in zip(self.mol.atoms, self.mol.coords):
            color = self.mol.atom_colors.get(atom, "grey")
            ax.scatter(coord[0], coord[1], coord[2], color=color, s=300, edgecolors="k", alpha=0.9)

    def _plot_vdw_spheres(
        self,
        ax: plt.Axes,
        alpha: float = 0.8,
        u_res: int = SPHERE_U_RESOLUTION,
        v_res: int = SPHERE_V_RESOLUTION,
        linewidth: float = 0,
    ) -> None:
        """Render atoms as VDW spheres on the given 3D axis."""
        u_grid = complex(0, u_res)
        v_grid = complex(0, v_res)
        for atom, coord in zip(self.mol.atoms, self.mol.coords):
            color = self.mol.atom_colors.get(atom, "#808080")
            radius = self.mol.vdw_radii.get(atom, DEFAULT_VDW_RADIUS)
            u, v = np.mgrid[0 : 2 * np.pi : u_grid, 0 : np.pi : v_grid]
            x = radius * np.cos(u) * np.sin(v) + coord[0]
            y = radius * np.sin(u) * np.sin(v) + coord[1]
            z = radius * np.cos(v) + coord[2]
            ax.plot_surface(x, y, z, color=color, linewidth=linewidth, antialiased=True, shade=True, alpha=alpha)

    def _plot_hull_surface(
        self,
        ax: plt.Axes,
        color: str = "cyan",
        alpha: float = 0.5,
        linewidth: float = 0,
    ) -> ConvexHull:
        """Render convex hull triangles and return the hull object."""
        hull = ConvexHull(self.mol.coords)
        for simplex in hull.simplices:
            triangle = self.mol.coords[simplex]
            ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                            color=color, alpha=alpha, linewidth=linewidth)
        return hull

    def _plot_bonds(self, ax: plt.Axes) -> None:
        """Draw bonds between atoms closer than MAX_BOND_DISTANCE."""
        tree = cKDTree(self.mol.coords)
        pairs = tree.query_pairs(MAX_BOND_DISTANCE)
        for i, j in pairs:
            ax.plot(
                [self.mol.coords[i][0], self.mol.coords[j][0]],
                [self.mol.coords[i][1], self.mol.coords[j][1]],
                [self.mol.coords[i][2], self.mol.coords[j][2]],
                color="grey", linewidth=1.5, alpha=0.7,
            )

    def _build_legend(
        self,
        ax: plt.Axes,
        extra_items: list[tuple[str, str]] | None = None,
        extra_patches: list[Patch] | None = None,
    ) -> None:
        """Build and add a legend with atom elements and optional extra items."""
        unique_atoms = self.mol.get_unique_atoms()
        legend_elements = []
        for element in unique_atoms:
            color = self.mol.atom_colors.get(element, "grey")
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", label=element,
                       markerfacecolor=color, markersize=12, markeredgecolor="k")
            )
        if extra_items:
            for label, color in extra_items:
                legend_elements.append(
                    Line2D([0], [0], marker="o", color="w", label=label,
                           markerfacecolor=color, markersize=12, markeredgecolor="k")
                )
        if extra_patches:
            legend_elements.extend(extra_patches)

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            fontsize=7,
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
            markerscale=0.4,
            handletextpad=0.3,
            borderpad=0.4,
            labelspacing=0.3,
        )

    def _style_axis(self, ax: plt.Axes, azim: float, elev: float) -> None:
        ax.view_init(elev=elev, azim=azim)
        ax.grid(False)
        ax.set_axis_off()
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")
        ax.dist = 6


class SimpleMoleculePlot(BasePlot):
    def plot(self, azim: float = 45, elev: float = 30) -> plt.Figure:
        fig, ax = self._create_3d_axis()
        self._plot_atoms_as_points(ax)
        self._build_legend(ax)
        self._style_axis(ax, azim, elev)
        return fig


class VDWMoleculePlot(BasePlot):
    def plot(self, azim: float = 45, elev: float = 30) -> plt.Figure:
        fig, ax = self._create_3d_axis()
        self._plot_vdw_spheres(ax, alpha=0.8, linewidth=2)
        self._plot_bonds(ax)
        self._build_legend(ax)
        self._style_axis(ax, azim, elev)
        return fig


class HullMoleculePlot(BasePlot):
    def plot(self, azim: float = 45, elev: float = 30) -> plt.Figure:
        fig, ax = self._create_3d_axis()
        self._plot_vdw_spheres(ax, alpha=0.6)
        self._plot_hull_surface(ax, color="cyan", alpha=0.5)
        self._build_legend(ax)
        self._style_axis(ax, azim, elev)
        return fig


class GridHullPlot(BasePlot):
    def __init__(self, molecule_data: MoleculeData, grid_resolution: float = 0.1) -> None:
        super().__init__(molecule_data)
        self.grid_resolution = grid_resolution

    def plot(self, azim: float = 45, elev: float = 30) -> plt.Figure:
        fig, ax = self._create_3d_axis()

        number_of_points = 20_000

        hull = ConvexHull(self.mol.coords)
        max_radius = max(self.mol.vdw_radii.values())
        min_coords = np.min(self.mol.coords[hull.vertices], axis=0) - max_radius
        max_coords = np.max(self.mol.coords[hull.vertices], axis=0) + max_radius

        cube_points = np.random.uniform(low=min_coords, high=max_coords, size=(number_of_points, 3))

        ax.scatter(cube_points[:, 0], cube_points[:, 1], cube_points[:, 2],
                   color="blue", s=1, alpha=0.6, label="Random points")

        self._plot_vdw_spheres(ax, alpha=0.6)

        for simplex in hull.simplices:
            triangle = self.mol.coords[simplex]
            ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                            color="cyan", alpha=0.5, linewidth=0)

        self._build_legend(ax, extra_items=[("Grid of points", "blue")])
        self._style_axis(ax, azim, elev)
        return fig


class PointsInHullPlot(BasePlot):
    def __init__(self, molecule_data: MoleculeData, grid_resolution: float = 0.25) -> None:
        super().__init__(molecule_data)
        self.grid_resolution = grid_resolution

    def plot(self, azim: float = 45, elev: float = 30) -> plt.Figure:
        fig, ax = self._create_3d_axis()

        number_of_points = 80_000

        hull = ConvexHull(self.mol.coords)
        hull_points = self.mol.coords[hull.vertices]
        max_radius = max(self.mol.vdw_radii.values())
        min_coords = np.min(hull_points, axis=0) - max_radius
        max_coords = np.max(hull_points, axis=0) + max_radius

        random_points = np.random.uniform(low=min_coords, high=max_coords, size=(number_of_points, 3))

        delaunay = Delaunay(hull_points)
        inside_hull = delaunay.find_simplex(random_points) >= 0
        points_in_hull = random_points[inside_hull]

        ax.scatter(points_in_hull[:, 0], points_in_hull[:, 1], points_in_hull[:, 2],
                   color="cyan", s=3, alpha=0.8, label="Random points inside a convex hull")

        self._plot_vdw_spheres(ax, alpha=0.6)

        for simplex in hull.simplices:
            triangle = self.mol.coords[simplex]
            ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                            color="cyan", alpha=0, linewidth=0)

        self._build_legend(ax, extra_items=[("Points in ConvexHull", "blue")])
        self._style_axis(ax, azim, elev)
        return fig


class PointsInAtomsPlot(BasePlot):
    def __init__(
        self,
        molecule_data: MoleculeData,
        vdw_alpha: float = 0.6,
        dot_size: float = 1.0,
        dot_color_in_atoms: str = "green",
        dot_color_cavities: str = "blue",
        plot_size: tuple[float, float] = (6, 5),
        number_of_points: int = 100_000,
    ) -> None:
        super().__init__(molecule_data)
        self.vdw_alpha = vdw_alpha
        self.plot_size = plot_size
        self.dot_size = dot_size
        self.dot_color_in_atoms = dot_color_in_atoms
        self.dot_color_cavities = dot_color_cavities
        self.number_of_points = number_of_points

    def plot(self, azim: float = 45, elev: float = 30) -> plt.Figure:
        fig, ax = self._create_3d_axis(figsize=self.plot_size)

        result = classify_points(
            self.mol.coords,
            self.mol.get_radii_array(),
            num_points=self.number_of_points,
        )

        ax.scatter(
            result.points_in_hull[result.inside_atoms_mask, 0],
            result.points_in_hull[result.inside_atoms_mask, 1],
            result.points_in_hull[result.inside_atoms_mask, 2],
            color=self.dot_color_in_atoms, s=self.dot_size, alpha=0.2,
            label="Points inside vdw spheres",
        )

        ax.scatter(
            result.points_in_hull[~result.inside_atoms_mask, 0],
            result.points_in_hull[~result.inside_atoms_mask, 1],
            result.points_in_hull[~result.inside_atoms_mask, 2],
            color=self.dot_color_cavities, s=self.dot_size, alpha=0.7,
            label="Points in the cavities",
        )

        self._plot_vdw_spheres(ax, alpha=self.vdw_alpha, u_res=20, v_res=10)

        for simplex in result.hull.simplices:
            triangle = self.mol.coords[simplex]
            ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2],
                            color="cyan", alpha=0)

        self._build_legend(ax, extra_items=[
            ("Points in the cavities", "blue"),
            ("Points inside vdw spheres", "green"),
        ])
        self._style_axis(ax, azim, elev)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        return fig


class VisualizationCavityWithConvex(BasePlot):
    def __init__(
        self,
        molecule_data: MoleculeData,
        number_of_points: int = 100_000,
        show_hull: bool = True,
        dot_color: str = "blue",
        vdw_alpha: float = 0.3,
        hull_face_color: str = "cyan",
        hull_face_alpha: float = 0.12,
        hull_edge_color: str = "k",
        hull_edge_width: float = 0.4,
        dot_size: float = 8.0,
    ) -> None:
        super().__init__(molecule_data)
        self.number_of_points = number_of_points
        self.show_hull = show_hull
        self.dot_color = dot_color
        self.vdw_alpha = vdw_alpha
        self.hull_face_color = hull_face_color
        self.hull_face_alpha = hull_face_alpha
        self.hull_edge_color = hull_edge_color
        self.hull_edge_width = hull_edge_width
        self.dot_size = dot_size

    def plot(self, azim: float = 45, elev: float = 30) -> plt.Figure:
        fig, ax = self._create_3d_axis(figsize=(6, 5))

        result = classify_points(
            self.mol.coords,
            self.mol.get_radii_array(),
            num_points=self.number_of_points,
        )

        ax.scatter(
            result.points_in_hull[~result.inside_atoms_mask, 0],
            result.points_in_hull[~result.inside_atoms_mask, 1],
            result.points_in_hull[~result.inside_atoms_mask, 2],
            color=self.dot_color, s=self.dot_size, alpha=0.7,
            label="Points in the cavities", linewidths=0.5,
        )

        self._plot_vdw_spheres(ax, alpha=self.vdw_alpha, u_res=20, v_res=10)

        extra_patches = []
        if self.show_hull:
            hull_faces = [self.mol.coords[simplex] for simplex in result.hull.simplices]
            hull_mesh = Poly3DCollection(
                hull_faces,
                facecolor=self.hull_face_color,
                edgecolor=self.hull_edge_color,
                linewidth=self.hull_edge_width,
                alpha=self.hull_face_alpha,
            )
            hull_mesh.set_zorder(0)
            ax.add_collection3d(hull_mesh)
            extra_patches.append(
                Patch(
                    facecolor=self.hull_face_color,
                    edgecolor=self.hull_edge_color,
                    label="Convex hull",
                    alpha=self.hull_face_alpha,
                )
            )

        self._build_legend(
            ax,
            extra_items=[("Points in the cavities", "blue")],
            extra_patches=extra_patches,
        )
        self._style_axis(ax, azim, elev)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        return fig

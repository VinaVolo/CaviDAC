"""3D molecular visualization using PyVista.

Provides multiple plot types for rendering molecules with van der Waals spheres,
convex hulls, and cavity point classifications.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull, cKDTree

from cavidac.constants import (
    DEFAULT_VDW_RADIUS,
    MAX_BOND_DISTANCE,
    SPHERE_U_RESOLUTION,
)
from cavidac.geometry.spatial import classify_points
from cavidac.visualization.molecule_data import MoleculeData


class BasePlot:
    def __init__(self, molecule_data: MoleculeData) -> None:
        self.mol = molecule_data

    def _add_vdw_spheres(
        self,
        plotter: pv.Plotter,
        alpha: float = 1.0,
        resolution: int = SPHERE_U_RESOLUTION,
    ) -> None:
        """Render atoms as VDW spheres."""
        for atom, coord in zip(self.mol.atoms, self.mol.coords):
            color = self.mol.atom_colors.get(atom, "#808080")
            radius = self.mol.vdw_radii.get(atom, DEFAULT_VDW_RADIUS)
            sphere = pv.Sphere(
                radius=radius, center=coord,
                theta_resolution=resolution, phi_resolution=resolution,
            )
            plotter.add_mesh(
                sphere, color=color, opacity=alpha,
                smooth_shading=True, specular=0.3, specular_power=15,
            )

    def _add_bonds(self, plotter: pv.Plotter) -> None:
        """Draw bonds between atoms closer than MAX_BOND_DISTANCE."""
        tree = cKDTree(self.mol.coords)
        pairs = tree.query_pairs(MAX_BOND_DISTANCE)
        for i, j in pairs:
            line = pv.Line(self.mol.coords[i], self.mol.coords[j])
            tube = line.tube(radius=0.06)
            plotter.add_mesh(tube, color="#aaaaaa", opacity=0.9)

    def _add_hull_surface(
        self,
        plotter: pv.Plotter,
        color: str = "cyan",
        alpha: float = 0.3,
        show_edges: bool = True,
    ) -> ConvexHull:
        """Render convex hull as a mesh and return the hull object."""
        hull = ConvexHull(self.mol.coords)
        faces = hull.simplices
        n_faces = len(faces)
        pv_faces = np.column_stack([np.full(n_faces, 3), faces]).ravel()
        mesh = pv.PolyData(self.mol.coords, pv_faces)
        plotter.add_mesh(
            mesh, color=color, opacity=alpha,
            show_edges=show_edges, edge_color="#555555", line_width=0.5,
        )
        return hull

    def _style_plotter(self, plotter: pv.Plotter) -> None:
        plotter.set_background("white")
        plotter.enable_anti_aliasing("ssaa")


class VDWMoleculePlot(BasePlot):
    def plot(self, plotter: pv.Plotter) -> None:
        self._add_vdw_spheres(plotter, alpha=1.0)
        self._add_bonds(plotter)
        self._style_plotter(plotter)


class HullMoleculePlot(BasePlot):
    def plot(self, plotter: pv.Plotter) -> None:
        self._add_vdw_spheres(plotter, alpha=0.6)
        self._add_hull_surface(plotter, color="cyan", alpha=0.3, show_edges=True)
        self._style_plotter(plotter)


class PointsInAtomsPlot(BasePlot):
    def __init__(
        self,
        molecule_data: MoleculeData,
        vdw_alpha: float = 0.15,
        dot_color_in_atoms: str = "#2ecc71",
        dot_color_cavities: str = "#e74c3c",
        number_of_points: int = 100_000,
    ) -> None:
        super().__init__(molecule_data)
        self.vdw_alpha = vdw_alpha
        self.dot_color_in_atoms = dot_color_in_atoms
        self.dot_color_cavities = dot_color_cavities
        self.number_of_points = number_of_points

    def plot(self, plotter: pv.Plotter) -> None:
        result = classify_points(
            self.mol.coords,
            self.mol.get_radii_array(),
            num_points=self.number_of_points,
        )

        in_atoms = result.points_in_hull[result.inside_atoms_mask]
        in_cavity = result.points_in_hull[~result.inside_atoms_mask]

        self._add_vdw_spheres(plotter, alpha=self.vdw_alpha, resolution=12)

        if len(in_atoms) > 0:
            cloud_atoms = pv.PolyData(in_atoms)
            plotter.add_mesh(
                cloud_atoms, color=self.dot_color_in_atoms,
                point_size=3, render_points_as_spheres=True, opacity=0.6,
            )

        if len(in_cavity) > 0:
            cloud_cavity = pv.PolyData(in_cavity)
            plotter.add_mesh(
                cloud_cavity, color=self.dot_color_cavities,
                point_size=5, render_points_as_spheres=True, opacity=1.0,
            )
        self._style_plotter(plotter)


class VisualizationCavityWithConvex(BasePlot):
    def __init__(
        self,
        molecule_data: MoleculeData,
        number_of_points: int = 100_000,
        show_hull: bool = True,
        dot_color: str = "#e74c3c",
        vdw_alpha: float = 0.15,
        hull_face_color: str = "cyan",
        hull_face_alpha: float = 0.12,
    ) -> None:
        super().__init__(molecule_data)
        self.number_of_points = number_of_points
        self.show_hull = show_hull
        self.dot_color = dot_color
        self.vdw_alpha = vdw_alpha
        self.hull_face_color = hull_face_color
        self.hull_face_alpha = hull_face_alpha

    def plot(self, plotter: pv.Plotter) -> None:
        result = classify_points(
            self.mol.coords,
            self.mol.get_radii_array(),
            num_points=self.number_of_points,
        )

        in_cavity = result.points_in_hull[~result.inside_atoms_mask]
        if len(in_cavity) > 0:
            cloud = pv.PolyData(in_cavity)
            plotter.add_mesh(
                cloud, color=self.dot_color,
                point_size=4, render_points_as_spheres=True, opacity=0.7,
            )

        self._add_vdw_spheres(plotter, alpha=self.vdw_alpha, resolution=12)

        if self.show_hull:
            n_faces = len(result.hull.simplices)
            pv_faces = np.column_stack([np.full(n_faces, 3), result.hull.simplices]).ravel()
            hull_mesh = pv.PolyData(self.mol.coords, pv_faces)
            plotter.add_mesh(
                hull_mesh, color=self.hull_face_color, opacity=self.hull_face_alpha,
                show_edges=True, edge_color="black", line_width=0.4,
            )

        self._style_plotter(plotter)

import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(project_root)
import json
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from abc import ABC, abstractmethod
from CalixVolApp.utils.paths import get_project_path

class IMoleculeReader(ABC):
    @abstractmethod
    def read(self, file_path: str):
        """
        Reads molecule from a file and returns its atoms and coordinates.

        Parameters
        ----------
        file_path : str
            Path to a file containing molecule coordinates.

        Returns
        -------
        atoms : list
            List of atoms in a molecule.
        coordinates : numpy.ndarray
            Array of shape (n_atoms, 3) containing coordinates of atoms in a molecule.
        """
        pass


class IVDWRadiusProvider(ABC):
    @abstractmethod
    def get_radius(self, atom: str) -> float:
        """
        Returns van der Waals radius of given atom in angstroms.

        Parameters
        ----------
        atom : str
            Atomic symbol.

        Returns
        -------
        float
            Van der Waals radius in angstroms.

        Raises
        ------
        KeyError
            If atom is not found in the provider database.
        """
        pass


class IVolumeEstimator(ABC):
    @abstractmethod
    def estimate(self, atoms, coordinates, vdw_provider: IVDWRadiusProvider, grid_resolution: float = 0.1):
        """
        Estimates volume of a molecule.

        Parameters
        ----------
        atoms : list
            List of atoms in a molecule.
        coordinates : numpy.ndarray
            Array of shape (n_atoms, 3) containing coordinates of atoms in a molecule.
        vdw_provider : IVDWRadiusProvider
            Provider of van der Waals radii.
        grid_resolution : float, optional
            Resolution of the grid for estimation, by default 0.1

        Returns
        -------
        tuple
            Tuple of three floats containing total volume, volume of atoms and volume of cavity.

        Raises
        ------
        ValueError
            If grid_resolution is not positive.
        """
        pass


class MoleculeFileReader(IMoleculeReader):
    def read(self, file_path: str):
        """
        Reads atomic symbols and coordinates from a file.

        Parameters
        ----------
        file_path : str
            Path to the file containing atomic symbols and coordinates.

        Returns
        -------
        tuple
            Tuple of two elements. The first element is a list of atomic symbols, the second is a 2D numpy array of shape (n_atoms, 3) containing coordinates of atoms.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        
        atomic_symbols = []
        coordinates = []
        with open(file_path) as file:
            for line in file:
                atom_data = line.strip().split()
                atomic_symbols.append(atom_data[0])
                coordinates.append([float(x) for x in atom_data[1:4]])
        return atomic_symbols, np.array(coordinates)


class JsonVDWRadiusProvider(IVDWRadiusProvider):
    def __init__(self, json_file_path: str):
        """
        Initializes the provider from a JSON file.

        Parameters
        ----------
        json_file_path : str
            Path to the JSON file containing van der Waals radii.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        with open(json_file_path, "r") as file:
            self._vdw_radii = json.load(file)

    def get_radius(self, atom: str) -> float:
        """
        Returns the van der Waals radius for the given atom.

        Parameters
        ----------
        atom : str
            Atomic symbol

        Returns
        -------
        float
            Van der Waals radius in Angstroms

        Raises
        ------
        KeyError
            If the atom is not found in the database.
        """

        return self._vdw_radii.get(atom, 1.5)


class ConvexHullVolumeEstimator(IVolumeEstimator):
    def estimate(self, atoms, coordinates, vdw_provider: IVDWRadiusProvider, grid_resolution: float = 0.1):
        """
        Estimates the volume of a molecule.

        Parameters
        ----------
        atoms : list
            List of atomic symbols.
        coordinates : numpy.ndarray
            Array of shape (n_atoms, 3) containing coordinates of atoms in a molecule.
        vdw_provider : IVDWRadiusProvider
            Provider of van der Waals radii.
        grid_resolution : float, optional
            Resolution of the grid for estimation, by default 0.1

        Returns
        -------
        tuple
            Tuple of three floats containing total volume, volume of atoms and volume of cavity.

        Raises
        ------
        ValueError
            If grid_resolution is not positive.
        """
        atom_radii = np.array([vdw_provider.get_radius(atom) for atom in atoms])
        convex_hull = ConvexHull(coordinates)
        hull_vertices = coordinates[convex_hull.vertices]

        max_vdw_radius = max(atom_radii)
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
        volume_cavity = volume_total - volume_atom

        return volume_total, volume_atom, volume_cavity

class MoleculeVolumeCalculator:
    def __init__(self, 
                 reader: IMoleculeReader, 
                 vdw_provider: IVDWRadiusProvider,
                 estimator: IVolumeEstimator):
        """
        Constructor of MoleculeVolumeCalculator

        Parameters
        ----------
        reader : IMoleculeReader
            The reader that reads atomic symbols and coordinates from a file.
        vdw_provider : IVDWRadiusProvider
            The provider that provides Van der Waals radii of atoms.
        estimator : IVolumeEstimator
            The estimator that estimates the volume of a molecule.
        """
        self.reader = reader
        self.vdw_provider = vdw_provider
        self.estimator = estimator

    def calculate(self, molecule_file_path: str, grid_resolution=0.1):
        """
        Calculates the volume of a molecule.

        Parameters
        ----------
        molecule_file_path : str
            Path to the file containing the molecule.
        grid_resolution : float, optional
            Resolution of the grid for estimation, by default 0.1

        Returns
        -------
        tuple
            Tuple of three floats containing total volume, volume of atoms and volume of cavity.

        Raises
        ------
        ValueError
            If grid_resolution is not positive.
        """

        atoms, coordinates = self.reader.read(molecule_file_path)
        volume_total, volume_atom, volume_cavity = self.estimator.estimate(atoms, coordinates, self.vdw_provider, grid_resolution)
        
        return volume_total, volume_atom, volume_cavity


if __name__ == "__main__":
    
    vdw_file = os.path.join(get_project_path(), 'CalixVolApp', 'data', 'vdw', 'vdw_radius.json')
    molecule_file = os.path.join(get_project_path(), 'CalixVolApp', 'data', 'molecules', '3.txt')

    reader = MoleculeFileReader()
    vdw_provider = JsonVDWRadiusProvider(vdw_file)
    estimator = ConvexHullVolumeEstimator()

    calculator = MoleculeVolumeCalculator(reader, vdw_provider, estimator)

    volume_total, volume_atom, volume_cavity = calculator.calculate(molecule_file, grid_resolution=0.1)

    print(f"Общий объем выпуклой оболочки: {volume_total:.2f} Å³")
    print(f"Объем, занятый атомами: {volume_atom:.2f} Å³")
    print(f"Объем полости: {volume_cavity:.2f} Å³")

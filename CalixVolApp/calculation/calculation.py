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
        pass


class IVDWRadiusProvider(ABC):
    @abstractmethod
    def get_radius(self, atom: str) -> float:
        pass


class IVolumeEstimator(ABC):
    @abstractmethod
    def estimate(self, atoms, coordinates, vdw_provider: IVDWRadiusProvider, grid_resolution: float = 0.1):
        pass


class MoleculeFileReader(IMoleculeReader):
    """
    Класс, отвечающий за чтение молекулярных данных из файла.
    """
    def read(self, file_path: str):
        atomic_symbols = []
        coordinates = []
        with open(file_path) as file:
            for line in file:
                atom_data = line.strip().split()
                atomic_symbols.append(atom_data[0])
                coordinates.append([float(x) for x in atom_data[1:4]])
        return atomic_symbols, np.array(coordinates)


class JsonVDWRadiusProvider(IVDWRadiusProvider):
    """
    Класс, отвечающий за предоставление ван-дер-ваальсовых радиусов атомов из JSON-файла.
    """
    def __init__(self, json_file_path: str):
        with open(json_file_path, "r") as file:
            self._vdw_radii = json.load(file)

    def get_radius(self, atom: str) -> float:
        # Возвращаем радиус, при отсутствии используем значение по умолчанию
        return self._vdw_radii.get(atom, 1.5)


class ConvexHullVolumeEstimator(IVolumeEstimator):
    """
    Класс для оценки объема молекулы по выпуклой оболочке и распределению точек.
    Отвечает за вычисления, не зависит напрямую от конкретных реализаций чтения данных или радиусов.
    """
    def estimate(self, atoms, coordinates, vdw_provider: IVDWRadiusProvider, grid_resolution: float = 0.1):
        # Получаем радиусы для всех атомов
        atom_radii = np.array([vdw_provider.get_radius(atom) for atom in atoms])

        # Вычисляем выпуклую оболочку
        convex_hull = ConvexHull(coordinates)
        hull_vertices = coordinates[convex_hull.vertices]

        max_vdw_radius = max(atom_radii)
        min_bounds = np.min(hull_vertices, axis=0) - max_vdw_radius
        max_bounds = np.max(hull_vertices, axis=0) + max_vdw_radius

        # Создаем 3D сетку точек
        grid_x = np.arange(min_bounds[0], max_bounds[0], grid_resolution)
        grid_y = np.arange(min_bounds[1], max_bounds[1], grid_resolution)
        grid_z = np.arange(min_bounds[2], max_bounds[2], grid_resolution)
        grid_x, grid_y, grid_z = np.meshgrid(grid_x, grid_y, grid_z)
        grid_points = np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T

        # Определяем точки внутри оболочки
        delaunay = Delaunay(hull_vertices)
        points_within_hull = delaunay.find_simplex(grid_points) >= 0
        hull_points = grid_points[points_within_hull]

        # Определяем точки внутри атомов
        kdtree = cKDTree(coordinates)
        indices_within_atoms = kdtree.query_ball_point(hull_points, r=np.max(atom_radii))
        is_inside_atom = np.zeros(len(hull_points), dtype=bool)

        for idx, atom_indices in enumerate(indices_within_atoms):
            point = hull_points[idx]
            for atom_index in atom_indices:
                if np.linalg.norm(point - coordinates[atom_index]) <= atom_radii[atom_index]:
                    is_inside_atom[idx] = True
                    break

        # Общий объем выпуклой оболочки
        volume_total = convex_hull.volume
        # Объем, занятый атомами (как и прежде)
        volume_atom = np.sum(is_inside_atom) * (grid_resolution ** 3)
        # Новый способ вычисления объема полости: разность между объемом конвекс-холла и атомным объемом
        volume_cavity = volume_total - volume_atom

        return volume_total, volume_atom, volume_cavity

# ===== Высокоуровневый код (композиция) =====

class MoleculeVolumeCalculator:
    """
    Высокоуровневый класс, который объединяет в себе все шаги:
    - Чтение молекулы
    - Получение радиусов
    - Расчет объемов
    Это точки входа в функциональность.
    """

    def __init__(self, 
                 reader: IMoleculeReader, 
                 vdw_provider: IVDWRadiusProvider,
                 estimator: IVolumeEstimator):
        self.reader = reader
        self.vdw_provider = vdw_provider
        self.estimator = estimator

    def calculate(self, molecule_file_path: str, grid_resolution=0.1):
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

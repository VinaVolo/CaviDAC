"""Data container for molecule visualization."""

from __future__ import annotations

import numpy as np

from cavidac.constants import DEFAULT_VDW_RADIUS


class MoleculeData:
    """Immutable container for molecule visualization data."""

    __slots__ = ("atoms", "coords", "atom_colors", "vdw_radii")

    def __init__(
        self,
        atoms: list[str],
        coords: np.ndarray,
        atom_colors: dict[str, str],
        vdw_radii: dict[str, float],
    ) -> None:
        self.atoms = atoms
        self.coords = coords
        self.atom_colors = atom_colors
        self.vdw_radii = vdw_radii

    def get_unique_atoms(self) -> set[str]:
        return set(self.atoms)

    def get_radii_array(self) -> np.ndarray:
        """Return (N,) array of VDW radii for each atom."""
        return np.array([self.vdw_radii.get(a, DEFAULT_VDW_RADIUS) for a in self.atoms])

"""Van der Waals radius providers."""

from __future__ import annotations

import json
import os

from abc import ABC, abstractmethod

from cavidac.constants import DEFAULT_VDW_RADIUS


class IVDWRadiusProvider(ABC):
    @abstractmethod
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
        """
        pass


class JsonVDWRadiusProvider(IVDWRadiusProvider):
    def __init__(self, json_file_path: str) -> None:
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
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"The radius file was not found: {json_file_path}")

        with open(json_file_path, "r") as file:
            self._radii = json.load(file)

    def get_radius(self, atom: str) -> float:
        """Returns the van der Waals radius, defaulting to 1.5 A for unknown atoms."""
        return self._radii.get(atom, DEFAULT_VDW_RADIUS)

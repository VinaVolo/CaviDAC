"""Molecule coordinate file readers."""

from __future__ import annotations

import os

import numpy as np
from abc import ABC, abstractmethod


class IMoleculeReader(ABC):
    @abstractmethod
    def read(self, file_path: str) -> tuple[list[str], np.ndarray]:
        """
        Reads atomic symbols and coordinates from a file.

        Parameters
        ----------
        file_path : str
            Path to the file containing atomic symbols and coordinates.

        Returns
        -------
        tuple[list[str], np.ndarray]
            Tuple of atomic symbols and (n_atoms, 3) coordinate array.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        pass


class MoleculeFileReader(IMoleculeReader):
    def read(self, file_path: str) -> tuple[list[str], np.ndarray]:
        """
        Reads atomic symbols and coordinates from a text file.

        Each line: ELEMENT x y z (whitespace-separated).
        Blank lines are skipped.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If a line has fewer than 4 columns.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        atoms, coords = [], []
        with open(file_path, "r") as file:
            for line_num, line in enumerate(file, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) < 4:
                    raise ValueError(
                        f"Line {line_num} in {file_path} has {len(parts)} columns, "
                        f"expected at least 4 (ELEMENT x y z): {line.strip()!r}"
                    )
                atoms.append(parts[0])
                coords.append([float(x) for x in parts[1:4]])

        return atoms, np.array(coords)

"""Tests for MoleculeFileReader."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from CalixVolApp.calculation.calculation import MoleculeFileReader


class TestMoleculeFileReader:
    def setup_method(self) -> None:
        self.reader = MoleculeFileReader()

    def test_read_returns_atoms_and_coords(self, tmp_molecule_file: str) -> None:
        atoms, coords = self.reader.read(tmp_molecule_file)
        assert atoms == ["C", "C", "C", "C"]
        assert coords.shape == (4, 3)

    def test_read_correct_coordinates(self, tmp_molecule_file: str) -> None:
        _, coords = self.reader.read(tmp_molecule_file)
        np.testing.assert_allclose(coords[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(coords[1], [2.0, 0.0, 0.0])

    def test_read_real_molecule_file(self, sample_molecule_file: str) -> None:
        atoms, coords = self.reader.read(sample_molecule_file)
        assert len(atoms) > 0
        assert coords.shape[1] == 3
        assert len(atoms) == coords.shape[0]

    def test_read_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="File not found"):
            self.reader.read("/nonexistent/path/molecule.txt")

    def test_read_skips_blank_lines(self) -> None:
        content = "C  0.0  0.0  0.0\n\n\nO  1.0  1.0  1.0\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            atoms, coords = self.reader.read(path)
            assert atoms == ["C", "O"]
            assert coords.shape == (2, 3)
        finally:
            os.unlink(path)

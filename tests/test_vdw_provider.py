"""Tests for JsonVDWRadiusProvider."""

from __future__ import annotations

import pytest

from CalixVolApp.calculation.calculation import JsonVDWRadiusProvider
from CalixVolApp.calculation.spatial import DEFAULT_VDW_RADIUS


class TestJsonVDWRadiusProvider:
    def test_load_from_json(self, tmp_vdw_json: str) -> None:
        provider = JsonVDWRadiusProvider(tmp_vdw_json)
        assert provider.get_radius("C") == 1.7
        assert provider.get_radius("O") == 1.52

    def test_default_radius_for_unknown_atom(self, tmp_vdw_json: str) -> None:
        provider = JsonVDWRadiusProvider(tmp_vdw_json)
        assert provider.get_radius("Zz") == DEFAULT_VDW_RADIUS

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="radius file was not found"):
            JsonVDWRadiusProvider("/nonexistent/vdw.json")

    def test_load_real_vdw_file(self, vdw_radius_file: str) -> None:
        provider = JsonVDWRadiusProvider(vdw_radius_file)
        assert provider.get_radius("H") == pytest.approx(1.2)
        assert provider.get_radius("C") == pytest.approx(1.7)

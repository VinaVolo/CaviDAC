# CaviDAC

Computational prediction of cavity volumes in calixarenes via tessellation and divide-and-conquer algorithms.

## Overview

CaviDAC estimates internal cavity volumes of calixarene molecules by constructing a convex hull around atomic coordinates, sampling a 3D grid of points inside the hull, and classifying each point as belonging to an atom (within its van der Waals radius) or to the cavity. The difference between the hull volume and the volume occupied by atoms gives the cavity volume.

## Project structure

```
CaviDAC/
├── CalixVolApp/
│   ├── calculation/
│   │   ├── spatial.py          # Shared spatial classification (Delaunay + cKDTree)
│   │   ├── calculation.py      # Volume estimation interfaces and implementations
│   │   └── visualization.py    # 8 Matplotlib 3D plot classes
│   ├── data/
│   │   ├── molecules/          # 15 calixarene coordinate files (.txt + .pdb)
│   │   └── vdw/                # Van der Waals radii and element colours (JSON)
│   ├── GUI/
│   │   └── app.py              # PyQt5 desktop application
│   └── utils/
│       └── paths.py            # Project root path helper
├── notebooks/                  # Jupyter notebooks for benchmarking and visualization
├── results/                    # Pre-computed volumes (CaviDAC, pyKVFinder, PyWindow)
├── tests/                      # pytest test suite
├── molecule_volume_calculator.ui  # Qt Designer UI file
└── pyproject.toml
```

## Algorithm

1. Compute the **ConvexHull** of atomic coordinates (scipy).
2. Build a **Delaunay triangulation** of hull vertices to classify grid points as inside/outside the hull.
3. Construct a **uniform 3D grid** covering the hull bounding box (spacing = `grid_resolution`).
4. For each grid point inside the hull, query a **cKDTree** to find nearby atoms and check whether the point falls within any atom's van der Waals radius.
5. **Cavity volume** = hull volume - (count of points inside atoms x grid_resolution^3).

The `grid_resolution` parameter controls accuracy vs. speed: smaller values yield more precise estimates at the cost of longer computation times.

## Installation

```bash
git clone https://github.com/VinaVolo/CaviDAC.git
cd CaviDAC
uv sync
```

To install optional dependencies for Jupyter notebooks (pyKVFinder, PyWindow comparison):

```bash
uv sync --extra notebooks
```

To install development dependencies (pytest, coverage):

```bash
uv sync --extra dev
```

## Usage

### Command-line

```python
from CalixVolApp.calculation.calculation import (
    MoleculeFileReader,
    JsonVDWRadiusProvider,
    ConvexHullVolumeEstimator,
    MoleculeVolumeCalculator,
)
from CalixVolApp.utils.paths import get_project_path

proj = get_project_path()
vdw_file = proj / "CalixVolApp" / "data" / "vdw" / "vdw_radius.json"
mol_file = proj / "CalixVolApp" / "data" / "molecules" / "txt_calix" / "1.txt"

reader = MoleculeFileReader()
vdw_provider = JsonVDWRadiusProvider(str(vdw_file))
estimator = ConvexHullVolumeEstimator()
calculator = MoleculeVolumeCalculator(reader, vdw_provider, estimator)

total, atoms, cavity = calculator.calculate(str(mol_file), grid_resolution=0.1)
print(f"Convex hull volume: {total:.2f} A^3")
print(f"Atomic volume:      {atoms:.2f} A^3")
print(f"Cavity volume:      {cavity:.2f} A^3")
```

### GUI

```bash
python -m CalixVolApp.GUI.app
```

The main window allows loading up to three molecular coordinate files. Click **Calculate** to compute volumes and render 3D visualizations (VDW spheres, convex hull overlay, cavity point classification).

## Input file format

Plain text, one atom per line: element symbol followed by Cartesian coordinates in Angstroms.

```
O  5.68628  3.77511  2.61536
O  6.32204  5.93875  1.28402
H  5.88298  4.68061  2.26094
```

Blank lines are skipped. Lines with fewer than 4 columns raise a `ValueError` with the line number. See `CalixVolApp/data/molecules/` for examples.

## Testing

```bash
uv run pytest tests/ -v
```

With coverage:

```bash
uv run pytest tests/ --cov=CalixVolApp --cov-report=term-missing
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `results_cavidac.ipynb` | Grid resolution sweep and volume computation for all 14 molecules |
| `results_kvfinder.ipynb` | Comparison with pyKVFinder |
| `results_pywindow.ipynb` | Comparison with PyWindow |
| `visualization.ipynb` | Demonstration of all visualization classes |

## Dependencies

**Core:** numpy, scipy, matplotlib, PyQt5, tqdm

**Notebooks (optional):** ipykernel, pandas, pykvfinder, pywindowx, pytoml

**Dev (optional):** pytest, pytest-cov

See `pyproject.toml` for version constraints. Requires Python >= 3.13.

## Contributing

Contributions are welcome. When adding new functionality or fixing bugs, please include tests and adhere to the existing code style. New molecules can be added as `.txt` files in `CalixVolApp/data/molecules/txt_calix/`. Van der Waals parameters can be edited in the JSON files under `CalixVolApp/data/vdw/`.

## License

MIT License. See [LICENSE](LICENSE) for details.

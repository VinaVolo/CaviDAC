# CaviDAC

Computational prediction of cavity volumes in calixarenes via tessellation and divide-and-conquer algorithms.

## Overview

CaviDAC estimates internal cavity volumes of calixarene molecules by constructing a convex hull around atomic coordinates, sampling a 3D grid of points inside the hull, and classifying each point as belonging to an atom (within its van der Waals radius) or to the cavity. The difference between the hull volume and the volume occupied by atoms gives the cavity volume.

## Project structure

```
CaviDAC/
├── cavidac/                    # Python package
│   ├── __init__.py             # Public API
│   ├── constants.py            # Shared constants and path helpers
│   ├── io/
│   │   ├── reader.py           # IMoleculeReader, MoleculeFileReader
│   │   └── vdw_provider.py     # IVDWRadiusProvider, JsonVDWRadiusProvider
│   ├── geometry/
│   │   ├── spatial.py          # Delaunay + cKDTree point classification
│   │   └── volume.py           # ConvexHullVolumeEstimator, MoleculeVolumeCalculator
│   ├── visualization/
│   │   ├── molecule_data.py    # MoleculeData container
│   │   └── plots.py            # PyVista 3D plot classes (VDW, hull, cavity)
│   └── gui/
│       └── app.py              # PyQt5 + PyVista desktop application
├── data/
│   ├── molecules/              # Calixarene coordinate files (.txt + .pdb)
│   └── vdw/                    # Van der Waals radii and element colours (JSON)
├── notebooks/                  # Jupyter notebooks for benchmarking and visualization
├── results/                    # Pre-computed volumes (CaviDAC, pyKVFinder, PyWindow)
├── tests/                      # pytest test suite
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
from cavidac import (
    MoleculeFileReader,
    JsonVDWRadiusProvider,
    ConvexHullVolumeEstimator,
    MoleculeVolumeCalculator,
)
from cavidac.constants import get_data_path

data = get_data_path()
vdw_file = str(data / "vdw" / "vdw_radius.json")
mol_file = str(data / "molecules" / "txt_calix" / "1.txt")

reader = MoleculeFileReader()
vdw_provider = JsonVDWRadiusProvider(vdw_file)
estimator = ConvexHullVolumeEstimator()
calculator = MoleculeVolumeCalculator(reader, vdw_provider, estimator)

total, atoms, cavity = calculator.calculate(mol_file, grid_resolution=0.1)
print(f"Convex hull volume: {total:.2f} A^3")
print(f"Atomic volume:      {atoms:.2f} A^3")
print(f"Cavity volume:      {cavity:.2f} A^3")
```

### GUI

```bash
uv run python -m cavidac.gui.app
```

The GUI uses PyVista (VTK) for interactive 3D rendering. Load up to two molecular coordinate files and click **Calculate** to compute volumes and render three visualizations:

- **Van der Waals spheres** with atomic bonds
- **Convex hull** overlay on the molecule
- **Cavity classification** showing points inside atoms vs. cavity space

All 3D views support interactive rotation, zoom, and pan with the mouse.

## Input file format

Plain text, one atom per line: element symbol followed by Cartesian coordinates in Angstroms.

```
O  5.68628  3.77511  2.61536
O  6.32204  5.93875  1.28402
H  5.88298  4.68061  2.26094
```

Blank lines are skipped. Lines with fewer than 4 columns raise a `ValueError` with the line number. See `data/molecules/` for examples.

## Testing

```bash
uv run pytest tests/ -v
```

With coverage:

```bash
uv run pytest tests/ --cov=cavidac --cov-report=term-missing
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `results_cavidac.ipynb` | Grid resolution sweep and volume computation for all 14 molecules |
| `results_kvfinder.ipynb` | Comparison with pyKVFinder |
| `results_pywindow.ipynb` | Comparison with PyWindow |
| `visualization.ipynb` | Demonstration of all visualization classes |

## Dependencies

**Core:** numpy, scipy, matplotlib, PyQt5, PyVista, pyvistaqt (VTK), tqdm

**Notebooks (optional):** ipykernel, pandas, pykvfinder, pywindowx, pytoml

**Dev (optional):** pytest, pytest-cov

See `pyproject.toml` for version constraints. Requires Python >= 3.13.

## Building standalone executables

CaviDAC can be packaged into a standalone desktop application using [PyInstaller](https://pyinstaller.org/). The resulting binary includes Python, all dependencies, and the `data/` directory — no separate installation required.

### Prerequisites

```bash
uv pip install pyinstaller
```

### macOS

```bash
pyinstaller \
    --name CaviDAC \
    --windowed \
    --onedir \
    --add-data "data:data" \
    --hidden-import vtkmodules \
    --hidden-import vtkmodules.all \
    --hidden-import pyvistaqt \
    --collect-all vtkmodules \
    --collect-all pyvista \
    cavidac/gui/app.py
```

The application bundle will be created at `dist/CaviDAC.app`. You can move it to `/Applications/` or distribute it as a `.dmg`.

To create a DMG image:

```bash
hdiutil create -volname CaviDAC -srcfolder dist/CaviDAC.app -ov -format UDZO dist/CaviDAC.dmg
```

### Windows

```cmd
pyinstaller ^
    --name CaviDAC ^
    --windowed ^
    --onedir ^
    --add-data "data;data" ^
    --hidden-import vtkmodules ^
    --hidden-import vtkmodules.all ^
    --hidden-import pyvistaqt ^
    --collect-all vtkmodules ^
    --collect-all pyvista ^
    cavidac\gui\app.py
```

The executable will be at `dist\CaviDAC\CaviDAC.exe`. To distribute, archive the `dist\CaviDAC\` folder as a ZIP or create an installer with [NSIS](https://nsis.sourceforge.io/) or [Inno Setup](https://jrsoftware.org/isinfo.php).

> **Note:** On Windows, use `;` (semicolon) as the `--add-data` separator instead of `:` (colon).

### Linux

```bash
pyinstaller \
    --name CaviDAC \
    --windowed \
    --onedir \
    --add-data "data:data" \
    --hidden-import vtkmodules \
    --hidden-import vtkmodules.all \
    --hidden-import pyvistaqt \
    --collect-all vtkmodules \
    --collect-all pyvista \
    cavidac/gui/app.py
```

The binary will be at `dist/CaviDAC/CaviDAC`. To run:

```bash
./dist/CaviDAC/CaviDAC
```

For distribution, archive the directory or create an [AppImage](https://appimage.org/):

```bash
tar -czf CaviDAC-linux-x86_64.tar.gz -C dist CaviDAC
```

### Troubleshooting

- **Missing VTK modules:** If the app crashes with VTK import errors, add `--collect-all vtk` to the PyInstaller command.
- **Data files not found:** Ensure `--add-data` paths are correct relative to your working directory. The app locates `data/` relative to the package root.
- **Large binary size:** VTK adds ~100-200 MB. Use `--onedir` (default above) instead of `--onefile` for faster startup. To reduce size, run `pyinstaller` with `--strip` on Linux/macOS.
- **macOS code signing:** Unsigned apps trigger Gatekeeper warnings. To sign: `codesign --deep --force --sign - dist/CaviDAC.app`.

## Citation

If you use CaviDAC in your research, please cite our paper:

> Karalash, S. A., Shmurygina, A. V., Krotkov, N. A., Aliev, T. A., Skorb, E. V., & Muravev, A. A. (2026). CaviDAC: Computational Prediction of Cavity Volumes in Calixarenes via Tessellation and Divide-and-Conquer Algorithms. *Advanced Theory and Simulations*, *9*(2), e01444.

BibTeX:

```bibtex
@article{karalash2026cavidac,
  title={CaviDAC: Computational Prediction of Cavity Volumes in Calixarenes via Tessellation and Divide-and-Conquer Algorithms},
  author={Karalash, Sergei A and Shmurygina, Anna V and Krotkov, Nikita A and Aliev, Timur A and Skorb, Ekaterina V and Muravev, Anton A},
  journal={Advanced Theory and Simulations},
  volume={9},
  number={2},
  pages={e01444},
  year={2026},
  publisher={Wiley Online Library}
}
```

## Contributing

Contributions are welcome. When adding new functionality or fixing bugs, please include tests and adhere to the existing code style. New molecules can be added as `.txt` files in `data/molecules/txt_calix/`. Van der Waals parameters can be edited in the JSON files under `data/vdw/`.

## License

MIT License. See [LICENSE](LICENSE) for details.

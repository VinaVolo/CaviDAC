# CaviDAC: computational prediction of cavity volumes in calixarenes via tessellation and divide-and-conquer algorithms

## Features

- **Volume estimation:**  The core of CaviDAC is implemented in `CalixVolApp/calculation/calculation.py`.  It defines abstract interfaces for reading molecular coordinates (`IMoleculeReader`), retrieving van der Waals radii (`IVDWRadiusProvider`) and estimating volumes (`IVolumeEstimator`).  A concrete `MoleculeFileReader` parses coordinate files where each line contains an element symbol and three Cartesian coordinates; if the file does not exist it raises a `FileNotFoundError`.  A `JsonVDWRadiusProvider` reads van der Waals radii from a JSON file, and `ConvexHullVolumeEstimator` computes the convex hull of the atoms, constructs a grid of points covering the hull, classifies grid points as inside or outside atoms using a KD‑tree and returns the total, atomic and cavity volumes.

- **Divide‑and‑conquer algorithm:**  The `ConvexHullVolumeEstimator` builds a Delaunay triangulation of the hull and checks grid points to see whether they lie inside the hull and within any atom’s van der Waals radius.  The grid resolution is configurable; a smaller spacing yields a more accurate but slower estimation.  The difference between the hull volume and the sum of grid points inside atoms estimates the cavity volume.

- **3‑D visualisation:**  The module `CalixVolApp/calculation/visualization.py` contains several classes that use Matplotlib to display molecules:
  - `SimpleMoleculePlot` shows atoms as coloured points with an element‑specific legend.
  - `VDWMoleculePlot` draws each atom as a sphere scaled to its van der Waals radius and connects atoms that are closer than 2 Å.
  - `HullMoleculePlot` overlays the convex hull surface on the atomic spheres.
  - `GridHullPlot`, `PointsInHullPlot` and `PointsInAtomsPlot` illustrate random grids of points, points inside the hull and points classified inside atoms versus cavities.

- **PyQt5 GUI:**  The GUI in `CalixVolApp/GUI/app.py` loads a Qt Designer `.ui` file and wires up buttons to load up to three molecules and compute their volumes.  It constructs an `AppVolumeCalculator` window which accepts a molecule reader, van der Waals provider and volume estimator.  When the user clicks the **Calculate** button, the GUI reads atom colours and radii from JSON files, computes the hull, atomic and cavity volumes for each loaded file and displays the results.  It then renders three Matplotlib plots for each molecule (van der Waals spheres, convex hull surface and points‑in‑atoms view).

- **Sample data:**  The `CalixVolApp/data` directory stores van der Waals radii (`vdw_radius.json`) and colour definitions (`vdw_colors.json`) for elements.  For example, the JSON defines colours for common elements such as hydrogen ("H": "#FFFFFF"), carbon ("C": "#555555") and oxygen ("O": "#FF0D0D") and van der Waals radii such as 1.2 Å for hydrogen, 1.70 Å for carbon and 1.52 Å for oxygen.  The `molecules` sub‑folder contains coordinate files in both `.txt` and `.pdb` formats; each `.txt` line lists an element followed by its (x,y,z) coordinates, e.g. “O 5.68628 3.77511 2.61536”.

- **Jupyter notebooks:**  The `notebooks` directory includes notebooks (`results_cavidac.ipynb`, `results_kvfinder.ipynb`, `results_pywindow.ipynb`, `visualization.ipynb`) that reproduce calculations, compare CaviDAC with other cavity–volume tools such as pyKVFinder and PyWindow, and demonstrate plotting routines.

- **Results and benchmarks:**  Pre‑computed results are stored in `results/`.  For example, `results/CaviDAC/calc_volumes_cavidac.csv` lists cavity volumes for calixarene molecules at grid resolution 0.1; molecule 2 has a cavity volume of ≈116.4 Å³ and molecule 3 has ≈81.5 Å³.  Additional subdirectories `pyKVFinder` and `pywindow` contain volumes computed with those external packages for comparison.  An `exp_with_grid_res` folder explores how cavity volume depends on grid resolution.

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/VinaVolo/CaviDAC.git
   cd CaviDAC

2. **Set up a virtual environment and install dependencies**

    ```bash
    uv sync


## Usage

### Command‑line example

You can compute the cavity volume of a molecule without the GUI by instantiating the provided classes:

    from CalixVolApp.calculation.calculation import MoleculeFileReader, JsonVDWRadiusProvider, ConvexHullVolumeEstimator, MoleculeVolumeCalculator
    from CalixVolApp.utils.paths import get_project_path

    # paths to data
    proj = get_project_path()
    vdw_file = proj / "CalixVolApp" / "data" / "vdw" / "vdw_radius.json"
    mol_file = proj / "CalixVolApp" / "data" / "molecules" / "txt_calix" / "1.txt"

    # create objects
    reader = MoleculeFileReader()
    vdw_provider = JsonVDWRadiusProvider(str(vdw_file))
    estimator = ConvexHullVolumeEstimator()
    calculator = MoleculeVolumeCalculator(reader, vdw_provider, estimator)

    # calculate volumes
    hull, atoms, cavity = calculator.calculate(str(mol_file), grid_resolution=0.1)
    print(f"Convex hull volume: {hull:.2f} Å³")
    print(f"Atomic volume: {atoms:.2f} Å³")
    print(f"Cavity volume: {cavity:.2f} Å³")


  The grid_resolution parameter controls the spacing of the sampling grid: smaller values give more accurate results at the expense of runtime.

## Launching the GUI

  From the repository root, run the application:
  ```bash
  python CalixVolApp/GUI/app.py
  ```
The main window allows you to load up to three molecular coordinate files. When you click Calculate, CaviDAC loads element colours and radii from the JSON files, computes the volumes and displays the results in the interface

## Input file format

Molecule files are plain text; each line contains an element symbol followed by its (x,y,z) Cartesian coordinates in Ångströms:

```bash
O 5.68628 3.77511 2.61536
O 6.32204 5.93875 1.28402
O 5.72621 4.69752 -0.93629
…
```
Lines beginning with whitespace are tolerated; blank lines are ignored by the reader. See the CalixVolApp/data/molecules folder for examples.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests. When adding new functionality or fixing bugs, please include docstrings and adhere to the existing code style. Adding new molecules or van der Waals parameters can be done by editing the JSON files in CalixVolApp/data/vdw.

## License

This project is licensed under the MIT License.

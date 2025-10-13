import os
import pandas as pd
from tqdm import tqdm
from CalixVolApp.calculation.calculation import MoleculeFileReader, JsonVDWRadiusProvider, ConvexHullVolumeEstimator, MoleculeVolumeCalculator
from CalixVolApp.utils.paths import get_project_path

df = pd.DataFrame()
# resolutions = [0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
resolutions = [0.025, 0.05, 0.075, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
for res in resolutions:
    volume = []
    for mol in range(1, 15):
        vdw_file = os.path.join(get_project_path(), 'CalixVolApp', 'data', 'vdw', 'vdw_radius.json')
        molecule_file = os.path.join(get_project_path(), 'CalixVolApp', 'data', 'molecules', 'calix', f'{mol}.txt')
        reader = MoleculeFileReader()
        vdw_provider = JsonVDWRadiusProvider(vdw_file)
        estimator = ConvexHullVolumeEstimator()
        calculator = MoleculeVolumeCalculator(reader, vdw_provider, estimator)
        volume_total, volume_atom, volume_cavity = calculator.calculate(molecule_file, grid_resolution=res) 
        volume.append(volume_cavity)
    df[res] = volume
    
df.to_csv('results.csv', index=False)
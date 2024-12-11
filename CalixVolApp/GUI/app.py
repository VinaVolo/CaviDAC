import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(project_root)
import json
import matplotlib
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
matplotlib.use('QtAgg')
from CalixVolApp.calculation.calculation import *
from CalixVolApp.calculation.visualization import *
from CalixVolApp.utils.paths import get_project_path

class AppVolumeCalculator(QtWidgets.QMainWindow):
    def __init__(self, reader, vdw_provider, estimator):
        """
        Constructor of AppVolumeCalculator

        Parameters
        ----------
        reader : IMoleculeReader
            The reader that reads atomic symbols and coordinates from a file.
        vdw_provider : IVDWRadiusProvider
            The provider that provides Van der Waals radii of atoms.
        estimator : IVolumeEstimator
            The estimator that estimates the volume of a molecule.
        """
        super(AppVolumeCalculator, self).__init__()
        uic.loadUi('molecule_volume_calculator.ui', self)

        self.loadFileButton_1.clicked.connect(self.load_file_1)
        self.loadFileButton_2.clicked.connect(self.load_file_2)
        self.loadFileButton_3.clicked.connect(self.load_file_3)
        self.calculateButton.clicked.connect(self.calculate_volumes)

        self.file_path_1 = None
        self.file_path_2 = None
        self.file_path_3 = None

        self.calculator = MoleculeVolumeCalculator(reader, vdw_provider, estimator)

    def load_file_1(self):
        """
        Opens a file dialog to load a file with molecular coordinates.

        The name of the loaded file is then displayed in the GUI.
        """
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите файл с координатами", "", "Text Files (*.txt);;All Files (*)")
        if file_name:
            self.file_path_1 = file_name
            self.file_name_1.setText(f"Файл загружен: {os.path.basename(file_name)}")

    def load_file_2(self):
        """
        Opens a file dialog to load a file with molecular coordinates for the second file input.

        The name of the loaded file is then displayed in the GUI.
        """
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите файл с координатами", "", "Text Files (*.txt);;All Files (*)")
        if file_name:
            self.file_path_2 = file_name
            self.file_name_2.setText(f"Файл загружен: {os.path.basename(file_name)}")

    def load_file_3(self):
        """
        Opens a file dialog to load a file with molecular coordinates for the third file input.

        The name of the loaded file is then displayed in the GUI.
        """
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите файл с координатами", "", "Text Files (*.txt);;All Files (*)")
        if file_name:
            self.file_path_3 = file_name
            self.file_name_3.setText(f"Файл загружен: {os.path.basename(file_name)}")

    def calculate_volumes(self):
        """
        Starts the calculation of the volume of a molecule in a file loaded by the user.

        The method reads the file loaded by the user, calculates the volume of the molecule and
        displays the result in the GUI. Also, it renders a 3D visualization of the molecule and
        its volume.

        If the file is not loaded, a warning message is displayed to the user.

        :return: None
        """
        if not (self.file_path_1 or self.file_path_2 or self.file_path_3):
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите хотя бы один файл!")
            return

        try:
            with open(os.path.join(get_project_path(), 'CalixVolApp', 'data', 'vdw', 'vdw_colors.json'), "r") as file:
                atom_colors = json.load(file)

            with open(os.path.join(get_project_path(), 'CalixVolApp', 'data', 'vdw', 'vdw_radius.json'), "r") as file:
                vdw_radii = json.load(file)

            if self.file_path_1:
                hull_volume_1, atom_volume_1, cavity_volume_1 = self.calculator.calculate(self.file_path_1, grid_resolution=0.1)
                result_text_1 = (f"Объём выпуклой оболочки: {hull_volume_1:.2f} Å³\n"
                                 f"Объём атомов: {atom_volume_1:.2f} Å³\n"
                                 f"Объём полости: {cavity_volume_1:.2f} Å³")
                self.volume_print_1.setText(result_text_1)

                atoms_1, coords_1 = self.calculator.reader.read(self.file_path_1)
                mol_data_1 = MoleculeData(atoms_1, coords_1, atom_colors, vdw_radii)
                plots_1 = [
                    VDWMoleculePlot(mol_data_1).plot(azim=360, elev=-64),
                    HullMoleculePlot(mol_data_1).plot(azim=360, elev=-64),
                    PointsInAtomsPlot(mol_data_1).plot(azim=360, elev=-64),
                ]
                self.plot_graphs(plots_1, container="file1")


            if self.file_path_2:
                hull_volume_2, atom_volume_2, cavity_volume_2 = self.calculator.calculate(self.file_path_2, grid_resolution=0.1)
                result_text_2 = (f"Объём выпуклой оболочки: {hull_volume_2:.2f} Å³\n"
                                 f"Объём атомов: {atom_volume_2:.2f} Å³\n"
                                 f"Объём полости: {cavity_volume_2:.2f} Å³")
                self.volume_print_2.setText(result_text_2)

                atoms_2, coords_2 = self.calculator.reader.read(self.file_path_2)
                mol_data_2 = MoleculeData(atoms_2, coords_2, atom_colors, vdw_radii)
                plots_2 = [
                    VDWMoleculePlot(mol_data_2).plot(azim=360, elev=-64),
                    HullMoleculePlot(mol_data_2).plot(azim=360, elev=-64),
                    PointsInAtomsPlot(mol_data_2).plot(azim=360, elev=-64),
                ]
                self.plot_graphs(plots_2, container="file2")

            if self.file_path_3:
                hull_volume_3, atom_volume_3, cavity_volume_3 = self.calculator.calculate(self.file_path_3, grid_resolution=0.1)
                result_text_3 = (f"Объём выпуклой оболочки: {hull_volume_3:.2f} Å³\n"
                                 f"Объём атомов: {atom_volume_3:.2f} Å³\n"
                                 f"Объём полости: {cavity_volume_3:.2f} Å³")
                self.volume_print_3.setText(result_text_3)

                atoms_3, coords_3 = self.calculator.reader.read(self.file_path_3)
                mol_data_3 = MoleculeData(atoms_3, coords_3, atom_colors, vdw_radii)
                plots_3 = [
                    VDWMoleculePlot(mol_data_3).plot(azim=360, elev=-64),
                    HullMoleculePlot(mol_data_3).plot(azim=360, elev=-64),
                    PointsInAtomsPlot(mol_data_3).plot(azim=360, elev=-64),
                ]
                self.plot_graphs(plots_3, container="file3")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при расчете объема: {str(e)}")

    def plot_graphs(self, figures, container):
        if container == "file1":
            widgets = [self.fig_1_1, self.fig_1_2, self.fig_1_3]
        elif container == "file2":
            widgets = [self.fig_2_1, self.fig_2_2, self.fig_2_3]
        elif container == "file3":
            widgets = [self.fig_3_1, self.fig_3_2, self.fig_3_3]
        else:
            return

        for widget, figure in zip(widgets, figures):
            canvas = FigureCanvas(figure)
            canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            layout = QtWidgets.QVBoxLayout(widget)
            layout.addWidget(canvas)
            layout.setContentsMargins(0, 0, 0, 0)
            canvas.figure.tight_layout()
            canvas.draw()


app = QtWidgets.QApplication(sys.argv)

vdw_file = os.path.join(get_project_path(), 'CalixVolApp', 'data', 'vdw', 'vdw_radius.json')
reader = MoleculeFileReader()
vdw_provider = JsonVDWRadiusProvider(vdw_file)
estimator = ConvexHullVolumeEstimator()

window = AppVolumeCalculator(reader, vdw_provider, estimator)
window.show()
sys.exit(app.exec_())
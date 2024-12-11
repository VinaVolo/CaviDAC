import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_directory, "..", ".."))
sys.path.append(project_root)
import json
import matplotlib
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QGridLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
matplotlib.use('QtAgg')
from CalixVolApp.calculation.calculation import MoleculeVolumeCalculator, MoleculeFileReader, JsonVDWRadiusProvider, ConvexHullVolumeEstimator
from CalixVolApp.calculation.visualization import *

class AppVolumeCalculator(QtWidgets.QMainWindow):
    def __init__(self, reader, vdw_provider, estimator):
        super(AppVolumeCalculator, self).__init__()
        uic.loadUi('molecule_volume_calculator.ui', self)

        # Подключение кнопок
        self.loadFileButton_1.clicked.connect(self.load_file_1)
        self.loadFileButton_2.clicked.connect(self.load_file_2)
        self.loadFileButton_3.clicked.connect(self.load_file_3)
        self.calculateButton.clicked.connect(self.calculate_volumes)

        # Переменные для хранения путей к файлам
        self.file_path_1 = None
        self.file_path_2 = None
        self.file_path_3 = None

        # Инициализация calculator
        self.calculator = MoleculeVolumeCalculator(reader, vdw_provider, estimator)

    def load_file_1(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите файл с координатами", "", "Text Files (*.txt);;All Files (*)")
        if file_name:
            self.file_path_1 = file_name
            self.file_name_1.setText(f"Файл загружен: {os.path.basename(file_name)}")

    def load_file_2(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите файл с координатами", "", "Text Files (*.txt);;All Files (*)")
        if file_name:
            self.file_path_2 = file_name
            self.file_name_2.setText(f"Файл загружен: {os.path.basename(file_name)}")

    def load_file_3(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите файл с координатами", "", "Text Files (*.txt);;All Files (*)")
        if file_name:
            self.file_path_3 = file_name
            self.file_name_3.setText(f"Файл загружен: {os.path.basename(file_name)}")

    def calculate_volumes(self):
        if not (self.file_path_1 or self.file_path_2 or self.file_path_3):
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите хотя бы один файл!")
            return

        try:
            # Чтение вспомогательных файлов
            with open(os.path.join(get_project_path(), 'CalixVolApp', 'data', 'vdw', 'vdw_colors.json'), "r") as file:
                atom_colors = json.load(file)

            with open(os.path.join(get_project_path(), 'CalixVolApp', 'data', 'vdw', 'vdw_radius.json'), "r") as file:
                vdw_radii = json.load(file)

            # Рассчёт для файла 1 (если загружен)
            if self.file_path_1:
                hull_volume_1, atom_volume_1, cavity_volume_1 = self.calculator.calculate(self.file_path_1, grid_resolution=0.1)
                result_text_1 = (f"Объём выпуклой оболочки: {hull_volume_1:.2f} Å³\n"
                                 f"Объём атомов: {atom_volume_1:.2f} Å³\n"
                                 f"Объём полости: {cavity_volume_1:.2f} Å³")
                self.volume_print_1.setText(result_text_1)

                # Визуализация для файла 1
                atoms_1, coords_1 = self.calculator.reader.read(self.file_path_1)
                mol_data_1 = MoleculeData(atoms_1, coords_1, atom_colors, vdw_radii)
                plots_1 = [
                    VDWMoleculePlot(mol_data_1).plot(azim=360, elev=-64),
                    HullMoleculePlot(mol_data_1).plot(azim=360, elev=-64),
                    PointsInAtomsPlot(mol_data_1).plot(azim=360, elev=-64),
                ]
                self.plot_graphs(plots_1, container="file1")

            # Рассчёт для файла 2 (если загружен)
            if self.file_path_2:
                hull_volume_2, atom_volume_2, cavity_volume_2 = self.calculator.calculate(self.file_path_2, grid_resolution=0.1)
                result_text_2 = (f"Объём выпуклой оболочки: {hull_volume_2:.2f} Å³\n"
                                 f"Объём атомов: {atom_volume_2:.2f} Å³\n"
                                 f"Объём полости: {cavity_volume_2:.2f} Å³")
                self.volume_print_2.setText(result_text_2)

                # Визуализация для файла 2
                atoms_2, coords_2 = self.calculator.reader.read(self.file_path_2)
                mol_data_2 = MoleculeData(atoms_2, coords_2, atom_colors, vdw_radii)
                plots_2 = [
                    VDWMoleculePlot(mol_data_2).plot(azim=360, elev=-64),
                    HullMoleculePlot(mol_data_2).plot(azim=360, elev=-64),
                    PointsInAtomsPlot(mol_data_2).plot(azim=360, elev=-64),
                ]
                self.plot_graphs(plots_2, container="file2")

            # Рассчёт для файла 3 (если загружен)
            if self.file_path_3:
                hull_volume_3, atom_volume_3, cavity_volume_3 = self.calculator.calculate(self.file_path_3, grid_resolution=0.1)
                result_text_3 = (f"Объём выпуклой оболочки: {hull_volume_3:.2f} Å³\n"
                                 f"Объём атомов: {atom_volume_3:.2f} Å³\n"
                                 f"Объём полости: {cavity_volume_3:.2f} Å³")
                self.volume_print_3.setText(result_text_3)

                # Визуализация для файла 3
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
            canvas.setMinimumSize(450, 400)  # Установите минимальные размеры холста
            layout = QtWidgets.QVBoxLayout(widget)
            layout.addWidget(canvas)
            layout.setContentsMargins(0, 0, 0, 0)
            canvas.figure.tight_layout()
            canvas.draw()

# Основная часть
if __name__ == "__main__":
    from CalixVolApp.utils.paths import get_project_path

    app = QtWidgets.QApplication(sys.argv)

    # Загрузка зависимостей
    vdw_file = os.path.join(get_project_path(), 'CalixVolApp', 'data', 'vdw', 'vdw_radius.json')
    reader = MoleculeFileReader()
    vdw_provider = JsonVDWRadiusProvider(vdw_file)
    estimator = ConvexHullVolumeEstimator()

    # Создание и отображение окна
    window = AppVolumeCalculator(reader, vdw_provider, estimator)
    window.show()
    sys.exit(app.exec_())
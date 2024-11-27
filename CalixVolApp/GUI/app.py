import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QGridLayout
import matplotlib
matplotlib.use('QtAgg')  # Включаем оптимизированный бэкэнд
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os

from app.calculation import estimate_internal_volume

class MoleculeVolumeCalculator(QtWidgets.QMainWindow):
    def __init__(self):
        super(MoleculeVolumeCalculator, self).__init__()
        uic.loadUi('molecule_volume_calculator.ui', self)

        self.loadFileButton.clicked.connect(self.load_file)
        self.calculateButton.clicked.connect(self.calculate_volume)

        self.file_path = None

        self.gridLayout = QGridLayout(self.graphContainer)

    def load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите файл с координатами", "", "Text Files (*.txt);;All Files (*)")
        if file_name:
            self.file_path = file_name
            self.fileLabel.setText(f"Файл загружен: {os.path.basename(file_name)}")

    def calculate_volume(self):
        if not self.file_path:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите файл!")
            return

        try:        
            figures, volumes = estimate_internal_volume(self.file_path)

            hull_volume, atom_volume, cavity_volume = volumes

            result_text = (f"Объём выпуклой оболочки: {hull_volume:.2f} Å³\n"
                           f"Оценённый объём, занимаемый атомами: {atom_volume:.2f} Å³\n"
                           f"Оценённый объём внутренней полости: {cavity_volume:.2f} Å³")
            self.resultTextEdit.setPlainText(result_text)

            self.plot_graphs(figures)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при расчете объема: {str(e)}")

    def plot_graphs(self, figures):
        for i in reversed(range(self.gridLayout.count())):
            widget = self.gridLayout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        canvas1 = FigureCanvas(figures[0])
        self.gridLayout.addWidget(canvas1, 0, 0)

        canvas2 = FigureCanvas(figures[1])
        self.gridLayout.addWidget(canvas2, 0, 1)

        canvas3 = FigureCanvas(figures[2])
        self.gridLayout.addWidget(canvas3, 1, 0)

        canvas4 = FigureCanvas(figures[3])
        self.gridLayout.addWidget(canvas4, 1, 1)

        for i in range(4):
            self.gridLayout.itemAt(i).widget().draw()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MoleculeVolumeCalculator()
    window.show()
    sys.exit(app.exec_())

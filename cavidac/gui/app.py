"""PyQt5 GUI application for molecular cavity volume calculation."""

from __future__ import annotations

import json
import os
import sys
from functools import partial

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from cavidac.constants import get_data_path
from cavidac.io.reader import IMoleculeReader, MoleculeFileReader
from cavidac.io.vdw_provider import IVDWRadiusProvider, JsonVDWRadiusProvider
from cavidac.geometry.volume import (
    ConvexHullVolumeEstimator,
    IVolumeEstimator,
    MoleculeVolumeCalculator,
)
from cavidac.visualization.molecule_data import MoleculeData
from cavidac.visualization.plots import (
    HullMoleculePlot,
    PointsInAtomsPlot,
    VDWMoleculePlot,
)

_STYLE = """
QMainWindow {
    background-color: #f5f5f7;
}
QWidget#toolbar {
    background-color: #ffffff;
    border-bottom: 1px solid #d1d1d6;
}
QPushButton {
    background-color: #007aff;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #0066d6;
}
QPushButton:pressed {
    background-color: #004eb3;
}
QPushButton:disabled {
    background-color: #b0b0b8;
}
QPushButton#calcBtn {
    background-color: #34c759;
    font-size: 14px;
    font-weight: 600;
    padding: 10px 28px;
}
QPushButton#calcBtn:hover {
    background-color: #2db84d;
}
QPushButton#loadBtn {
    background-color: #5856d6;
}
QPushButton#loadBtn:hover {
    background-color: #4a48c4;
}
QDoubleSpinBox {
    border: 1px solid #d1d1d6;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
    background-color: white;
}
QLabel#sectionTitle {
    font-size: 15px;
    font-weight: 600;
    color: #1d1d1f;
}
QLabel#fileName {
    font-size: 13px;
    color: #6e6e73;
}
QLabel#volumeLabel {
    font-size: 13px;
    color: #1d1d1f;
    line-height: 1.5;
}
QWidget#resultCard {
    background-color: #ffffff;
    border: 1px solid #e5e5ea;
    border-radius: 10px;
}
QWidget#plotCard {
    background-color: #ffffff;
    border: 1px solid #e5e5ea;
    border-radius: 10px;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QWidget#scrollContent {
    background-color: transparent;
}
"""

_MAX_MOLECULES = 2


class MoleculePanel(QtWidgets.QWidget):
    """A panel displaying results and plots for a single molecule."""

    def __init__(self, index: int, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.index = index
        self._canvases: list[FigureCanvas] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        result_card = QtWidgets.QWidget()
        result_card.setObjectName("resultCard")
        card_layout = QtWidgets.QVBoxLayout(result_card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(6)

        title = QtWidgets.QLabel(f"Molecule {self.index + 1}")
        title.setObjectName("sectionTitle")
        card_layout.addWidget(title)

        self.file_label = QtWidgets.QLabel("No file loaded")
        self.file_label.setObjectName("fileName")
        card_layout.addWidget(self.file_label)

        self.volume_label = QtWidgets.QLabel("")
        self.volume_label.setObjectName("volumeLabel")
        self.volume_label.setTextFormat(QtCore.Qt.RichText)
        self.volume_label.setWordWrap(True)
        self.volume_label.hide()
        card_layout.addWidget(self.volume_label)

        layout.addWidget(result_card)

        self.plot_titles = ["Van der Waals spheres", "Convex hull", "Cavity classification"]
        self.plot_containers: list[QtWidgets.QWidget] = []
        self.plot_cards: list[QtWidgets.QWidget] = []

        self.plots_widget = QtWidgets.QWidget()
        self.plots_widget.hide()
        plots_grid = QtWidgets.QGridLayout(self.plots_widget)
        plots_grid.setContentsMargins(0, 0, 0, 0)
        plots_grid.setSpacing(12)

        for idx, title_text in enumerate(self.plot_titles):
            plot_card = QtWidgets.QWidget()
            plot_card.setObjectName("plotCard")
            plot_layout = QtWidgets.QVBoxLayout(plot_card)
            plot_layout.setContentsMargins(12, 10, 12, 12)
            plot_layout.setSpacing(4)

            label = QtWidgets.QLabel(title_text)
            label.setStyleSheet("font-size: 12px; font-weight: 500; color: #6e6e73;")
            plot_layout.addWidget(label)

            container = QtWidgets.QWidget()
            container.setMinimumHeight(300)
            container.setMaximumHeight(500)
            container_layout = QtWidgets.QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            plot_layout.addWidget(container)

            self.plot_containers.append(container)
            self.plot_cards.append(plot_card)

            if idx < 2:
                plots_grid.addWidget(plot_card, 0, idx)
            else:
                plots_grid.addWidget(plot_card, 1, 0, 1, 2, QtCore.Qt.AlignCenter)

        plots_grid.setColumnStretch(0, 1)
        plots_grid.setColumnStretch(1, 1)

        layout.addWidget(self.plots_widget)
        layout.addStretch()

    def set_file(self, file_name: str) -> None:
        self.file_label.setText(file_name)

    def set_volumes(self, hull_vol: float, atom_vol: float, cavity_vol: float) -> None:
        self.volume_label.setText(
            f"<b>Convex hull:</b> {hull_vol:.4f} \u00c5\u00b3 &nbsp;&nbsp; "
            f"<b>Atoms:</b> {atom_vol:.4f} \u00c5\u00b3 &nbsp;&nbsp; "
            f"<b>Cavity:</b> {cavity_vol:.4f} \u00c5\u00b3"
        )
        self.volume_label.show()

    def set_figures(self, figures: list[Figure]) -> None:
        for canvas in self._canvases:
            canvas.setParent(None)
            canvas.deleteLater()
        self._canvases.clear()

        for container, figure in zip(self.plot_containers, figures):
            layout = container.layout()
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            canvas = FigureCanvas(figure)
            canvas.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
            )
            layout.addWidget(canvas)
            canvas.draw()
            self._canvases.append(canvas)

        self.plots_widget.show()


class AppVolumeCalculator(QtWidgets.QMainWindow):
    def __init__(
        self,
        reader: IMoleculeReader,
        vdw_provider: IVDWRadiusProvider,
        estimator: IVolumeEstimator,
    ) -> None:
        super().__init__()
        self.setWindowTitle("CaviDAC \u2014 Cavity Volume Calculator")
        self.setMinimumSize(900, 700)
        self.showMaximized()

        self.file_paths: list[str | None] = [None] * _MAX_MOLECULES
        self.calculator = MoleculeVolumeCalculator(reader, vdw_provider, estimator)

        data_path = get_data_path()
        with open(data_path / "vdw" / "vdw_colors.json", "r") as f:
            self.atom_colors: dict[str, str] = json.load(f)
        with open(data_path / "vdw" / "vdw_radius.json", "r") as f:
            self.vdw_radii: dict[str, float] = json.load(f)

        self._build_ui()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        toolbar = QtWidgets.QWidget()
        toolbar.setObjectName("toolbar")
        toolbar.setFixedHeight(64)
        tb_layout = QtWidgets.QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(20, 0, 20, 0)
        tb_layout.setSpacing(12)

        app_label = QtWidgets.QLabel("CaviDAC")
        app_label.setStyleSheet(
            "font-size: 17px; font-weight: 700; color: #1d1d1f; margin-right: 8px;"
        )
        tb_layout.addWidget(app_label)

        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.VLine)
        separator.setStyleSheet("color: #d1d1d6;")
        tb_layout.addWidget(separator)

        self.load_buttons: list[QtWidgets.QPushButton] = []
        for i in range(_MAX_MOLECULES):
            btn = QtWidgets.QPushButton(f"Load Molecule {i + 1}")
            btn.setObjectName("loadBtn")
            btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            btn.clicked.connect(partial(self._load_file, i))
            tb_layout.addWidget(btn)
            self.load_buttons.append(btn)

        tb_layout.addSpacing(8)

        res_label = QtWidgets.QLabel("Grid resolution:")
        res_label.setStyleSheet("font-size: 13px; color: #6e6e73;")
        tb_layout.addWidget(res_label)

        self.resolution_spin = QtWidgets.QDoubleSpinBox()
        self.resolution_spin.setRange(0.01, 2.0)
        self.resolution_spin.setSingleStep(0.05)
        self.resolution_spin.setValue(0.1)
        self.resolution_spin.setDecimals(3)
        self.resolution_spin.setSuffix(" \u00c5")
        self.resolution_spin.setFixedWidth(100)
        tb_layout.addWidget(self.resolution_spin)

        tb_layout.addStretch()

        self.calc_button = QtWidgets.QPushButton("Calculate")
        self.calc_button.setObjectName("calcBtn")
        self.calc_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.calc_button.clicked.connect(self._calculate)
        tb_layout.addWidget(self.calc_button)

        main_layout.addWidget(toolbar)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        scroll_content = QtWidgets.QWidget()
        scroll_content.setObjectName("scrollContent")
        self.content_layout = QtWidgets.QHBoxLayout(scroll_content)
        self.content_layout.setContentsMargins(20, 20, 20, 20)
        self.content_layout.setSpacing(10)

        self.panels: list[MoleculePanel] = []
        for i in range(_MAX_MOLECULES):
            if i > 0:
                divider = QtWidgets.QFrame()
                divider.setFrameShape(QtWidgets.QFrame.VLine)
                divider.setStyleSheet(
                    "QFrame { color: #d1d1d6; margin-top: 4px; margin-bottom: 4px; }"
                )
                self.content_layout.addWidget(divider)
            panel = MoleculePanel(i)
            self.panels.append(panel)
            self.content_layout.addWidget(panel)

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        self.statusBar().setStyleSheet(
            "font-size: 12px; color: #8e8e93; background-color: #f5f5f7;"
        )
        self.statusBar().showMessage("Load a molecule file to begin.")

    def _load_file(self, slot: int) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select molecule coordinate file",
            "",
            "Text Files (*.txt);;PDB Files (*.pdb);;All Files (*)",
        )
        if file_name:
            self.file_paths[slot] = file_name
            base = os.path.basename(file_name)
            self.panels[slot].set_file(base)
            self.load_buttons[slot].setText(f"Molecule {slot + 1}: {base}")
            self.statusBar().showMessage(f"Loaded {base} into slot {slot + 1}.")

    def _calculate(self) -> None:
        if not any(self.file_paths):
            QMessageBox.warning(self, "No files", "Load at least one molecule file first.")
            return

        resolution = self.resolution_spin.value()
        self.calc_button.setEnabled(False)
        self.calc_button.setText("Calculating...")
        QtWidgets.QApplication.processEvents()

        for slot, file_path in enumerate(self.file_paths):
            if not file_path:
                continue

            try:
                atoms, coords, (hull_vol, atom_vol, cavity_vol) = (
                    self.calculator.calculate_with_data(file_path, grid_resolution=resolution)
                )

                panel = self.panels[slot]
                panel.set_volumes(hull_vol, atom_vol, cavity_vol)

                mol_data = MoleculeData(atoms, coords, self.atom_colors, self.vdw_radii)
                figures = [
                    VDWMoleculePlot(mol_data).plot(azim=360, elev=-64),
                    HullMoleculePlot(mol_data).plot(azim=360, elev=-64),
                    PointsInAtomsPlot(mol_data).plot(azim=360, elev=-64),
                ]
                panel.set_figures(figures)

                self.statusBar().showMessage(
                    f"Molecule {slot + 1}: cavity = {cavity_vol:.2f} \u00c5\u00b3"
                )
                QtWidgets.QApplication.processEvents()

            except Exception as e:
                QMessageBox.critical(
                    self, "Calculation Error",
                    f"Error processing molecule {slot + 1}:\n{e}",
                )

        self.calc_button.setEnabled(True)
        self.calc_button.setText("Calculate")


def main() -> None:
    import matplotlib
    matplotlib.use("QtAgg")

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(_STYLE)

    vdw_file = str(get_data_path() / "vdw" / "vdw_radius.json")
    reader = MoleculeFileReader()
    vdw_provider = JsonVDWRadiusProvider(vdw_file)
    estimator = ConvexHullVolumeEstimator()

    window = AppVolumeCalculator(reader, vdw_provider, estimator)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

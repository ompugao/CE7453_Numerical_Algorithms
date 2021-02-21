# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from scipy import interpolate
import numpy as np
import sys
from waypoints import Waypoints
from bspline import BSplineCurve, BSplineInterpolationSolver

from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QWidget, \
    QAction, \
    QApplication, \
    QColorDialog, \
    QDesktopWidget, \
    QFileDialog, \
    QFontDialog, \
    QHBoxLayout, \
    QInputDialog, \
    QInputDialog, \
    QLabel, \
    QLineEdit, \
    QMainWindow, \
    QMessageBox, \
    QPushButton, \
    QTextBrowser, \
    QToolTip, \
    QVBoxLayout, \
    qApp
from PyQt5.QtGui import QFont, \
    QIcon

class CBIMainWindow(QMainWindow):
    waypoints_changed = pyqtSignal()
    def __init__(self, ):
        super().__init__()

        self.solver = BSplineInterpolationSolver()

        self.waypoints = Waypoints()
        self.waypoints_changed.connect(self._draw_curve)

        self.figure = plt.figure(figsize=(7, 5))
        self.figure.tight_layout()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)

        exit_action = QAction(QIcon('assets/exit.jpg'), '&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Quit')
        exit_action.triggered.connect(qApp.quit)

        open_file_action = QAction(QIcon('assets/file.png'), '&Open file', self)
        open_file_action.setShortcut('Ctrl+O')
        open_file_action.setStatusTip('Open a file')
        open_file_action.triggered.connect(self._open_file)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(open_file_action)
        fileMenu.addAction(exit_action)

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        widget = QWidget()
        widget.setLayout(vbox)
        self.setCentralWidget(widget)

        self.statusBar().showMessage('Initialization Done')

    def _open_file(self,):
        fname = QFileDialog.getOpenFileName(self, 'Open a File', 'waypoints.txt', '*.txt')
        self.waypoints.read_file(fname[0])
        self.waypoints_changed.emit()

    def _reset_canvas(self,):
        self.axes.cla()
        self.axes.grid()
        self.axes.legend()

    def _draw_curve(self,):
        self._reset_canvas()
        curve = self.solver.solve(self.waypoints)
        x, y = interpolate.splev(np.arange(0, 1.01, 0.01), (curve.knot_vector, curve.control_points.T, 3))
        self.axes.set_xlabel('x', horizontalalignment='center', fontsize=23)
        self.axes.set_ylabel('y', horizontalalignment='center', fontsize=23)
        self.axes.plot(x, y, color='green', linestyle='-', label='b-spline interpolation')
        self._plot_waypoints(self.waypoints)
        self._plot_control_points(curve)
        self.canvas.draw()

    def _plot_waypoints(self, waypoints):
        self.axes.plot(waypoints.points[:, 0], waypoints.points[:, 1], 'x', color='blue')

    def _plot_control_points(self, curve):
        self.axes.plot(curve.control_points[:, 0], curve.control_points[:, 1], 'o', color='orange')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = CBIMainWindow()
    win.resize(1080, 880)
    win.show()
    sys.exit(app.exec_())

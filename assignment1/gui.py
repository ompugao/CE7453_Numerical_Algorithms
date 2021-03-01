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

class InteractionInfo:
    def __init__(self, ):
        self.prevclicked = None
        self.origpos = None
        self.iselectedwaypoint = None

class CBIMainWindow(QMainWindow):
    waypoints_changed = pyqtSignal()
    def __init__(self, ):
        super().__init__()

        self.neareset_waypoints_search_threshold = 0.1

        self.solver = BSplineInterpolationSolver()

        self.waypoints = Waypoints()
        self.waypoints_changed.connect(self._draw_curve)

        self.figure = plt.figure(figsize=(7, 5))
        self.figure.tight_layout()
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)

        exit_action = QAction(QIcon('assets/exit.png'), '&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Quit')
        exit_action.triggered.connect(qApp.quit)

        open_file_action = QAction(QIcon('assets/file.png'), '&Load waypoints', self)
        open_file_action.setShortcut('Ctrl+O')
        open_file_action.setStatusTip('Load Waypoints')
        open_file_action.triggered.connect(self._open_file)

        write_file_action = QAction(QIcon('assets/file.png'), '&Export waypoints', self)
        write_file_action.setShortcut('Ctrl+E')
        write_file_action.setStatusTip('Export Waypoints')
        write_file_action.triggered.connect(self._write_file)

        menubar = self.menuBar()
        file_menu = menubar.addMenu('&File')
        file_menu.addAction(open_file_action)
        file_menu.addAction(write_file_action)
        file_menu.addAction(exit_action)

        clear_button = QPushButton("Clear Points")
        clear_button.clicked.connect(self._clear_points)
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(clear_button)

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addLayout(hbox)
        widget = QWidget()
        widget.setLayout(vbox)
        self.setCentralWidget(widget)

        self.statusBar().showMessage('Initialization Done')

        self._draw_curve_callback = None
        self.canvas_size = dict(xlim=(-3, 3), ylim=(-3, 3))
        self._reset_canvas()
        self.interactioninfo = InteractionInfo()
        self._connect()

    def set_draw_curve_callback(self, callback):
        self._draw_curve_callback = callback

    def set_canvas_size(self, xlim, ylim):
        self.canvas_size = dict(xlim=xlim, ylim=ylim)

    def _clear_points(self,):
        self.waypoints = Waypoints()
        self.waypoints_changed.emit()

    def _open_file(self,):
        fname = QFileDialog.getOpenFileName(self, 'Open a File', 'waypoints.txt', '*.txt')
        if len(fname[0]) == 0:
            self.statusBar().showMessage('cancelled')
            return
        self.waypoints.read_file(fname[0])
        self.waypoints_changed.emit()
        self.statusBar().showMessage('waypoints loaded from %s'%fname[0])

    def _write_file(self,):
        fname = QFileDialog.getSaveFileName(self, 'Export waypoints into a File', 'waypoints.txt', '*.txt')
        if len(fname[0]) == 0:
            self.statusBar().showMessage('cancelled')
            return
        self.waypoints.write_file(fname[0])
        self.statusBar().showMessage('waypoints exported to %s'%fname[0])

    def _reset_canvas(self,):
        self.axes.cla()
        self.axes.grid()
        #self.axes.axis('equal')
        self.axes.set_aspect('equal', 'box')
        self.axes.set(xlim=self.canvas_size['xlim'], ylim=self.canvas_size['ylim'])
        self.canvas.draw()

    def _draw_curve(self,):
        self._reset_canvas()
        self._plot_waypoints(self.waypoints)
        if len(self.waypoints) < 2:
            return
        curve = self.solver.solve(self.waypoints)
        x, y = interpolate.splev(np.arange(0, 1.01, 0.01), (curve.knot_vector, curve.control_points.T, 3))
        self.axes.set_xlabel('x', horizontalalignment='center', fontsize=23)
        self.axes.set_ylabel('y', horizontalalignment='center', fontsize=23)
        self.axes.plot(x, y, color='green', linestyle='-', label='b-spline interpolation')
        self._plot_control_points(curve)

        if callable(self._draw_curve_callback):
            self._draw_curve_callback(self)

        self.axes.legend()
        self.canvas.draw()

    def _plot_waypoints(self, waypoints):
        if len(waypoints) > 0:
            return self.axes.plot(waypoints.points[:, 0], waypoints.points[:, 1], 'x', color='blue', label='waypoints')

    def _plot_control_points(self, curve):
        return self.axes.plot(curve.control_points[:, 0], curve.control_points[:, 1], 'o', color='orange', label='control points')

    def _connect(self):
        self.cidpress = self.canvas.mpl_connect(
            'button_press_event', self._on_press)
        self.cidrelease = self.canvas.mpl_connect(
            'button_release_event', self._on_release)
        self.cidmotion = self.canvas.mpl_connect(
            'motion_notify_event', self._on_motion)

    def _on_press(self, event):
        self.interactioninfo = InteractionInfo()
        iclosestwaypoint = None
        if len(self.waypoints) != 0:
            squareddists = np.sum((self.waypoints.points - np.array([event.xdata, event.ydata]))**2, axis=1)
            if np.min(squareddists) < self.neareset_waypoints_search_threshold*self.neareset_waypoints_search_threshold:
                iclosestwaypoint = np.argmin(squareddists)
                self.interactioninfo.origpos = self.waypoints.points[iclosestwaypoint]
                self.interactioninfo.iselectedwaypoint = iclosestwaypoint
        self.interactioninfo.prevclicked = np.array([event.xdata, event.ydata])


    def _on_motion(self, event):
        if self.interactioninfo is not None and \
                self.interactioninfo.iselectedwaypoint is not None:
            self.waypoints.points[self.interactioninfo.iselectedwaypoint] = [event.xdata, event.ydata]
            self.waypoints_changed.emit()

    def _on_release(self, event):
        if self.interactioninfo is not None and \
                self.interactioninfo.iselectedwaypoint is not None:
            pass
        else:
            if len(self.waypoints) != 0:
                self.waypoints.points = np.vstack([self.waypoints.points, [event.xdata, event.ydata]]) # append last
            else:
                self.waypoints.points = np.array([[event.xdata, event.ydata]])
            self.statusBar().showMessage('a waypoint added')

        self.interactioninfo = None
        self.waypoints_changed.emit()

    def disconnect(self):
        self.canvas.mpl_disconnect(self.cidpress)
        self.canvas.mpl_disconnect(self.cidrelease)
        self.canvas.mpl_disconnect(self.cidmotion)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = CBIMainWindow()
    win.resize(1080, 880)
    win.show()
    sys.exit(app.exec_())

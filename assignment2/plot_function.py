# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import optimize
import sys
sys.path.append('../assignment1')
from utils import plot_waypoints, plot_control_points
from bspline import BSplineInterpolationSolver

plt.rcParams["font.size"] = 18

h = 9


def x_func(u):
    global h
    return (np.exp(np.cos(6.2*u + h/30)) + 0.1)*np.cos(12.4*u)


def y_func(u):
    global h
    return (np.exp(np.cos(6.2*u + h/30)) + 0.1)*np.sin(12.4*u)


def question1():
    numsteps = 10000
    u = np.linspace(0, 1.0, numsteps)
    x = x_func(u)
    y = y_func(u)
    plt.plot(x, y, '-')
    plt.axis('equal')
    # plt.show()
    plt.savefig('a1.pdf')


def cubicpolynomial(u, params):
    u2 = (u**2)
    u3 = (u**3)
    x = params[0] + params[1] * u + params[2] * u2 + params[3] * u3
    y = params[3] + params[4] * u + params[6] * u2 + params[7] * u3
    return x, y


def question2():
    # sampling_us = np.linspace(0, 1.0, 10)
    # sampling_us = np.array([0,0.1,0.2,0.3, 0.7,0.8,0.9,1.0])
    def fitfunc(params, us):
        xs = x_func(us)
        ys = y_func(us)
        xfit, yfit = cubicpolynomial(us, params)

        return np.hstack(((xs - xfit), (ys - yfit)))

    np.random.seed(5)
    params = np.random.random(8)
    optparams, retflg = optimize.leastsq(fitfunc, params, args=(sampling_us,), maxfev=1000000)
    if retflg > 4:
        print('solution not found')
    print(optparams)

    numsteps = 10000
    u = np.linspace(0, 1.0, numsteps)
    x = x_func(u)
    y = y_func(u)
    plt.plot(x, y, '-')
    cubicx, cubicy = cubicpolynomial(u, optparams)
    plt.plot(cubicx, cubicy, 'green')
    plt.show()

def draw_curve_callback(self):
    numsteps = 10000
    u = np.linspace(0, 1.0, numsteps)
    x = x_func(u)
    y = y_func(u)
    self.axes.plot(x, y, '-')

def bspline_interactive():
    import gui
    from PyQt5.QtWidgets import QWidget, QAction, QApplication
    app = QApplication(sys.argv)
    win = gui.CBIMainWindow()
    win.set_draw_curve_callback(draw_curve_callback)
    win.resize(1080, 880)
    win.show()
    sys.exit(app.exec_())

def question3():
    from waypoints import Waypoints
    numsteps = 10
    u = np.linspace(0, 1.0, numsteps)
    x = x_func(u)
    y = y_func(u)
    waypoints = Waypoints()
    waypoints.points = np.hstack([x.reshape((len(x), 1)), y.reshape((len(y), 1))])
    solver = BSplineInterpolationSolver()
    curve = solver.solve(waypoints)
    plot_waypoints(waypoints)
    #plot_control_points(curve)
    xinterp, yinterp = interpolate.splev(np.linspace(0.0, 1.0, 10000), (curve.knot_vector, curve.control_points.T, 3))
    plt.plot(xinterp, yinterp, linestyle='-', label='b-spline interpolation')
    numsteps = 10000
    u = np.linspace(0, 1.0, numsteps)
    x = x_func(u)
    y = y_func(u)
    plt.plot(x, y, '-', label='original line')
    plt.grid()
    plt.axis('equal')
    plt.legend()
    plt.show()

def question4():
    n = 8
    u = np.linspace(0, 1.0, n)
    x = x_func(u)
    y = y_func(u)
    # fft is wrong?
    wx = np.fft.fft(x)
    wy = np.fft.fft(y)
    def genP(w):
        def P(u):
            # NOTE: n must be even 
            a = w.real
            b = w.imag
            return 1.0/np.sqrt(n) * \
                (a[0] + np.sum([(a[k] + a[n-k])*np.cos(2*np.pi*k*u) - \
                (b[k] - b[n-k])*np.sin(2*np.pi*k*u) for k in range(1, int(n/2)-1+1)]) \
                + a[int(n/2)]*np.cos(2*np.pi*n/2*u))
        return P
    Px = genP(wx)
    Py = genP(wy)

    numsteps = 10000
    uplot = np.linspace(0, 1.0, numsteps)
    x = x_func(uplot)
    y = y_func(uplot)
    xp = Px(uplot)
    yp = Py(uplot)
    plt.plot(x, y, '-')
    plt.plot(xp, yp, '-')
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    #question1()
    #question2()
    #question3()
    question4()

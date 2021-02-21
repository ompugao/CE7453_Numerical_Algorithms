from bspline import *

import logging
logging.basicConfig(format='[%(levelname)s][%(name)s:%(funcName)s]|%(filename)s:%(lineno)d| %(message)s')

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def test_write_bsplinecurve():
    # write test
    curve = BSplineCurve()
    curve.knot_vector = [0,0,0,0,1,1,1,1]
    curve.control_points = [[0,0], [1,1], [2,2],[3,4]]
    curve.write_file('test_out.txt')

def test_bspline_interpolation_solver():
    solver = BSplineInterpolationSolver()
    waypoints = Waypoints(filename='./waypoints.txt')
    ts, knots = solver._compute_parameterization(waypoints)
    assert(len(ts) == len(waypoints))
    assert(len(knots) == len(waypoints)+6)
    print(ts)
    print(knots)

import scipy
from matplotlib import pyplot as plt

def plot_waypoints(waypoints):
    plt.plot(waypoints.points[:, 0], waypoints.points[:, 1], 'x', color='blue')

def plot_control_points(curve):
    plt.plot(curve.control_points[:, 0], curve.control_points[:, 1], 'o', color='orange')

def test_plot_curve():
    solver = BSplineInterpolationSolver()
    waypoints = Waypoints(filename='./waypoints.txt')
    curve = solver.solve(waypoints)
    plot_waypoints(waypoints)
    plot_control_points(curve)
    x, y = scipy.interpolate.splev(np.arange(0, 1.01, 0.01), (curve.knot_vector, curve.control_points.T, 3))
    plt.plot(x, y, linestyle='-', label='b-spline interpolation')
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #test_write_bsplinecurve()
    #test_bspline_interpolation_solver()
    test_plot_curve()

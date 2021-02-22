# -*- coding: utf-8 -*-
import numpy as np
from enum import Enum
from exception import CBIException
from waypoints import Waypoints
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class BSplineCurve(object):
    def __init__(self, control_points = None, knot_vector = None):
        self.degree = 3
        self.control_points = control_points
        if self.control_points is None:
            self.control_points = []
        self.knot_vector = knot_vector
        if self.knot_vector is None:
            self.knot_vector = []

    def write_file(self, filename):
        with open(filename, 'w') as f:
            f.write('%d\n'%self.degree)
            f.write('%d\n'%len(self.control_points))
            for u in self.knot_vector:
                f.write('%f '%u)
            f.write('\n')
            for xy in self.control_points:
                f.write('%f %f\n'%(xy[0], xy[1]))

class InterpolationParameterizationType(Enum):
    UNIFORM = 1
    CHORDLENGTH = 2

class BSplineInterpolationSolver(object):
    """
    Cubic B-Spline Interpolation Solver
    """
    def __init__(self, ):
        pass

    def solve(self, waypoints):
        ts, knots = self._compute_parameterization(waypoints)
        A = self._compute_A(ts, knots)
        n = len(waypoints) - 1
        np.set_printoptions(suppress=True,precision=2)
        log.debug("A=")
        log.debug(A)
        b = np.zeros(n+3)
        b[1:n+2] = waypoints.points[:,0] #x
        log.debug("bx=")
        log.debug(b)
        px = np.linalg.solve(A, b)
        b[1:n+2] = waypoints.points[:,1] #y
        log.debug("by=")
        log.debug(b)
        py = np.linalg.solve(A, b)
        control_points = np.hstack([px.reshape([len(px), 1]), py.reshape([len(py), 1])])
        curve = BSplineCurve(control_points=control_points, knot_vector=knots)
        return curve

    def _compute_parameterization(self, waypoints, type = InterpolationParameterizationType.CHORDLENGTH):
        ts = []
        n = len(waypoints) - 1
        if type == InterpolationParameterizationType.CHORDLENGTH:
            ts = np.linspace(0, 1, n+1)
        elif type == InterpolationParameterizationType.UNIFORM:
            ts.append(0)
            z = waypoints.points.copy()
            z[1:] -= z[:-1].copy()
            dists = np.sqrt(np.sum(z[1:]**2, axis=1))
            normalized_cumdists = np.cumsum(dists/np.sum(dists))
            ts.extend(normalized_cumdists.tolist())
        else:
            raise ValueError('invalid interpolation type')
        ts = np.array(ts)
        knots = np.hstack([np.zeros(3), ts, np.ones(3)])
        return ts, knots

    def _compute_A(self, ts, knots):
        n = len(ts) - 1
        A = np.zeros([n+3, n+3])
        for i in range(1, n+2):
            for j in range(i-1, i+2):
                try:
                    A[i, j] = self._basis_func(j, ts[i-1], knots)
                except Exception as e:
                    from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

        for j in range(0, 3):
            A[0, j] = self._basis_func_2diff(j, ts[0], knots)
        for j in range(n, n+3):
            A[n+2, j] = self._basis_func_2diff(j, ts[n], knots)

        return A

    def _basis_func(self, i, u, knots):
        us = knots
        if us[i] <= u <= us[i+1] and (us[i] != u or u != us[i+1]):
            value = (u - us[i]) ** 3 / ((us[i+1] - us[i]) * (us[i+2]-us[i]) * (us[i+3] - us[i]))
        elif us[i+1] <= u <= us[i+2] and (us[i+1] != u or u != us[i+2]):
            value = ((u - us[i])**2 * (us[i+2] - u)) / ((us[i+2] - us[i+1])*(us[i+3] - us[i])*(us[i+2] - us[i])) \
                    + ((us[i+3] - u) * (u - us[i]) * (u - us[i+1])) / ((us[i+2] - us[i+1])*(us[i+3] - us[i+1])*(us[i+3] - us[i])) \
                    + (us[i+4] - u) * (u - us[i+1])**2 / ((us[i+2] - us[i+1])*(us[i+4] - us[i+1])*(us[i+3] - us[i+1]))
        elif us[i+2] <= u <= us[i+3] and (us[i+2] != u or u != us[i+3]):
            value = ((u - us[i]) * (us[i+3] - u)**2) / ((us[i+3] - us[i+2])*(us[i+3] - us[i+1])*(us[i+3] - us[i+1])) \
                    + ((us[i+4] - u) * (us[i+3] - u) * (u - us[i+1])) / ((us[i+3] - us[i+2])*(us[i+4] - us[i+1])*(us[i+3] - us[i+1])) \
                    + (us[i+4] - u)**2 * (u - us[i+2]) / ((us[i+3] - us[i+2])*(us[i+4] - us[i+2])*(us[i+4] - us[i+1]))
        elif us[i+3] <= u <= us[i+4] and (us[i+3] != u or u != us[i+4]):
            value = (us[i+4] - u) ** 3 / ((us[i+4] - us[i+3]) * (us[i+4]-us[i+2]) * (us[i+4] - us[i+1]))
        else:
            value = 0
            log.warn(('u is not in the domain (knot[%d]%f <= u: %f < knot[%d]%f)'%(i, us[i], u, i+4, us[i+4])))
            #from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())
            #raise ValueError('u is not in the domain (knot[%d]%f <= u: %f < knot[%d]%f)'%(i, us[i], u, i+4, us[i+4]))
        return value

    def _basis_func_2diff(self, i, u, knots):
        us = knots
        u0 = us[i]
        u1 = us[i+1]
        u2 = us[i+2]
        u3 = us[i+3]
        u4 = us[i+4]
        if us[i] <= u <= us[i+1] and (us[i] != u or u != us[i+1]):
            value = 6*(-u + u0)/((u0 - u1)*(u0 - u2)*(u0 - u3))
        elif us[i+1] <= u <= us[i+2] and (us[i+1] != u or u != us[i+2]):
            value = 2*((u0 - u2)*(u0 - u3)*(3*u - 2*u1 - u4) + (u0 - u2)*(u1 - u4)*(3*u - u0 - u1 - u3) + (u1 - u3)*(u1 - u4)*(3*u - 2*u0 - u2))/((u0 - u2)*(u0 - u3)*(u1 - u2)*(u1 - u3)*(u1 - u4))
        elif us[i+2] <= u <= us[i+3] and (us[i+2] != u or u != us[i+3]):
            value = 2*((u1 - u3)**2*(-3*u + u2 + 2*u4) + (u1 - u3)*(u2 - u4)*(-3*u + u1 + u3 + u4) + (u1 - u4)*(u2 - u4)*(-3*u + u0 + 2*u3))/((u1 - u3)**2*(u1 - u4)*(u2 - u3)*(u2 - u4))
        elif us[i+3] <= u <= us[i+4] and (us[i+3] != u or u != us[i+4]):
            value = 6*(u - u4)/((u1 - u4)*(u2 - u4)*(u3 - u4))
        else:
            log.warn(('u is not in the domain (knot[%d]%f <= u: %f < knot[%d]%f)'%(i, us[i], u, i+4, us[i+4])))
            return 0

        return value


if __name__ == '__main__':
    import argparse
    from utils import plot_waypoints, plot_control_points
    from scipy import interpolate
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputfile', type=str, default='waypoints.txt')
    args = parser.parse_args()
    solver = BSplineInterpolationSolver()
    waypoints = Waypoints(filename=args.inputfile)
    curve = solver.solve(waypoints)
    plot_waypoints(waypoints)
    plot_control_points(curve)
    x, y = interpolate.splev(np.arange(0, 1.01, 0.01), (curve.knot_vector, curve.control_points.T, 3))
    plt.plot(x, y, linestyle='-', label='b-spline interpolation')
    plt.grid()
    plt.axis('equal')
    plt.legend()
    plt.show()

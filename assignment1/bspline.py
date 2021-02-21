# -*- coding: utf-8 -*-
import numpy as np
from enum import Enum
from exception import CBIException
from waypoints import Waypoints

class BSplineCurve(object):
    def __init__(self, ):
        self.degree = 3
        self.control_points = []
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

    def solve(self,):
        pass

    def _compute_parameterization(self, waypoints: Waypoints,
            type: InterpolationParameterizationType = InterpolationParameterizationType.CHORDLENGTH):
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
        for i in range(1, n+3):
            for j in range(i-1, i+2):
                A[i, j] = self._basis_func(j, ts[i-1], knots)

    def _basis_func(self, i, u, knots):
        us = knots
        if us[i] <= u < us[i+1]:
            value = (u - us[i]) ** 3 / ((us[i+1] - us[i]) * (us[i+2]-us[i]) * (us[i+3] - us[i]))
        elif us[i+1] <= u < us[i+2]:
            value = ((u - us[i])**2 * (us[i+2] - u)) / ((us[i+2] - us[i+1])*(us[i+3] - us[i])*(us[i+2] - us[i])) \
                    + ((us[i+3] - u) * (u - us[i]) * (u - us[i+1])) / ((us[i+2] - us[i+1])*(us[i+3] - us[i+1])*(us[i+3] - us[i])) \
                    + (us[i+4] - u) * (u - us[i+1])**2 / ((us[i+2] - us[i+1])*(us[i+4] - us[i+1])*(us[i+3] - us[i+1]))
        elif us[i+2] <= u < us[i+3]:
            value = ((u - us[i]) * (us[i+3] - u)**2) / ((us[i+3] - us[i+2])*(us[i+3] - us[i+1])*(us[i+3] - us[i+1])) \
                    + ((us[i+4] - u) * (us[i+3] - u) * (u - us[i+1])) / ((us[i+3] - us[i+2])*(us[i+4] - us[i+1])*(us[i+3] - us[i+1])) \
                    + (us[i+4] - u)**2 * (u - us[i+2]) / ((us[i+3] - us[i+2])*(us[i+4] - us[i+2])*(us[i+4] - us[i+1]))
        elif us[i+3] <= u < us[i+4]:
            value = (us[i+4] - u) ** 3 / ((us[i+4] - us[i+3]) * (us[i+4]-us[i+2]) * (us[i+4] - us[i+1]))
        else:
            raise ValueError('u is not in the domain knot[%d]%f <= u: %f < knot[%d]%f'%(i, us[i], u, i+4, us[i+4]))
        return value

if __name__ == '__main__':
    solver = BSplineInterpolationSolver()
    

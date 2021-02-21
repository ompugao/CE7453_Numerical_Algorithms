from bspline import *

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

if __name__ == '__main__':
    test_write_bsplinecurve()
    test_bspline_interpolation_solver()

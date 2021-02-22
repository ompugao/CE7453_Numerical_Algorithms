# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt

def plot_waypoints(waypoints):
    plt.plot(waypoints.points[:, 0], waypoints.points[:, 1], 'x', color='blue')

def plot_control_points(curve):
    plt.plot(curve.control_points[:, 0], curve.control_points[:, 1], 'o', color='orange')


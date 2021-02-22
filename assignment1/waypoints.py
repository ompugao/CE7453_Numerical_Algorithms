# -*- coding: utf-8 -*-
import numpy as np

class Waypoints(object):
    def __init__(self, filename=None):
        self.points = np.array([])

        if filename is not None:
            self.read_file(filename)

    def __len__(self,):
        return len(self.points)

    def read_file(self, filename):
        with open(filename, 'r') as f:
            points = []
            for line in f.readlines():
                if len(line.rstrip()) == 0:
                    continue
                if line.lstrip().startswith('#'):
                    continue
                x, y = map(float, line.rstrip().split(' '))
                points.append([x, y])
            self.points = np.array(points)

    def write_file(self, filename):
        np.savetxt(filename, self.points, delimiter=' ')


if __name__ == '__main__':
    waypoints = Waypoints(filename='./waypoints.txt')
    print(waypoints.points)
    from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())


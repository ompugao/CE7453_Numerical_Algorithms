#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sympy
import matplotlib.pyplot as plt

n = 4
t = sympy.symbols('t')
c = sympy.symbols('c')
d = sympy.symbols('d:%d'%n)
s = sympy.symbols('s:%d'%n)

t_true = 10
d_vals = [5, 2, 3, 3]
s_vals = [3, 2, 6, 1]

def f_sympy():
    t = 0
    for di, si, in zip(d, s):
        t += di * 1.0 / (si + c)
    return t - t_true

if __name__ == '__main__':
    f = f_sympy()

    f = f.subs(dict(zip(d, d_vals))).subs(dict(zip(s, s_vals)))
    df = f.diff(c)

    xs = np.linspace(-0.8, 1.0, 1000)
    ys = np.array([f.subs(c, x) for x in xs])
    dfs = np.array([df.subs(c, x) for x in xs])

    plt.xlabel('c')
    plt.ylabel('value')
    plt.plot(xs, ys, label='f')
    plt.plot(xs, dfs, label='df')
    plt.grid()
    plt.legend()
    plt.show()




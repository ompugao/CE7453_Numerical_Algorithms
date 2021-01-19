#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

n = 4
t_true = 10
d = [5, 2, 3, 3]
s = [3, 2, 6, 1]

def newton_method(f, df, x0, tol=1e-8):
    c = x0
    prevc = 1e10
    #while np.abs(c - prevc) / c > tol:
    while np.abs(c - prevc) > tol:
        prevc = c
        c = c - f(c) / df(c)
        #print(c, prevc)
    return c


def f(c):
    t = 0
    for di, si, in zip(d, s):
        t += di * 1.0 / (si + c)
    return t - t_true

def df(c):
    a = 0
    for di, si, in zip(d, s):
        a += - di * 1.0 / ((si + c)**2)
    return a


if __name__ == '__main__':
    c = newton_method(f, df, 0.0)
    from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())




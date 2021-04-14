import sys
import math
import numpy as np
import matplotlib.pyplot as plt

def e_complex_pow(power):
    return complex(np.cos(power), np.sin(power))

def get_complex_representation(points):
    N = len(points)
    xs = np.array([(2*np.pi * i) / N for i in range(N)])
    ys = np.array(points, dtype=np.complex_)

    w = e_complex_pow(2*np.pi / N)
    
    # Fourier-Matrix
    F_N = np.array([ [w ** (j * k) for j in range(N)] for k in range(-(N-1)//2, (N-1)//2 + 1) ])

    Ck_s = (1 / N) * (np.conj(F_N) @ ys)  # Fourier-Coefficients    
    return Ck_s


def get_sin_cos_representation(points):
    Ck_s = get_complex_representation(points)
    Ck_zero_idx = (len(Ck_s)-1) // 2
    
    Ak_s = [2 * Ck_s[Ck_zero_idx]] + [Ck_s[Ck_zero_idx + n] + Ck_s[Ck_zero_idx - n] for n in range(1, Ck_zero_idx+1)]  # cosine coefficients
    Bk_s = [0] + [complex(0, Ck_s[Ck_zero_idx + n] - Ck_s[Ck_zero_idx - n]) for n in range(1, Ck_zero_idx+1)]  # sine coefficients

    for n in range(len(Ak_s)):
        if Ak_s[n].imag < 1e-3:
            Ak_s[n] = Ak_s[n].real
        
        if Bk_s[n].imag < 1e-3:
            Bk_s[n] = Bk_s[n].real

    return np.array(Ak_s), np.array(Bk_s)


def eval_sin_cos_representation(t, A, B):
    return A[0]/2 + sum(A[n] * np.cos(n * t) + B[n] * np.sin(n * t) for n in range(1, len(A)))


def plot_sin_cos_representation(A, B, y_points, start=-10, end=10):
    Xs = np.linspace(start, end, 5000)
    Ys = [eval_sin_cos_representation(t, A, B) for t in Xs]

    N = len(points)
    x_points = np.array([(2*np.pi * i) / N for i in range(N)])

    plt.figure(figsize=(14,7))
    plt.plot(Xs, Ys)
    plt.scatter(x_points, y_points, c='black')
    plt.show()


if __name__ == '__main__':
    points = list(map(float, sys.argv[1:]))
    A, B = get_sin_cos_representation(points)

    plot_sin_cos_representation(A, B, points, start=-4*np.pi, end=4*np.pi)
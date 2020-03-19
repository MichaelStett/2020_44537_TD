import matplotlib.pyplot as plt

from math import sin, cos, pi, log10, sqrt

(F, E, D, C, B, A) = (0, 4, 4, 5, 3, 7)

Amp = 1.0

def plot(x, y, title, description = 'empty', xlabel ='time', ylabel = 'values',  color = 'r', size = 0.25):
    plt.scatter(x, y, size, color)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(title)
    plt.title(title + "\n" + description)
    plt.show()

def func(t):
    f = B
    fi = C * pi
    return Amp*sin(2*pi*f*t + fi)

def quant(val, q):
    return (val + Amp) / (2 * Amp) * 2**q

def rangewithstep(start, stop, step):
    x = [start]
    i = 0
    while x[i] < stop:
        x.append(x[i] + step)
        i += 1
    return x

fs = 2500.0

x = rangewithstep(0, A, fs**-1)

y = [func(val) for i, val in enumerate(x)]

plot(x, y, f's(t)', f't:<{0},{A}> fs: {fs}Hz')

# 2
q = 16.0

y = [ quant(val, q) for i, val in enumerate(y)]

plot(x, y, f's(t)', f'fs: {fs}Hz q: {q}')

# 3
q = q / 2
fs = fs / 2

x = rangewithstep(0, A, fs**-1)

y = [ quant(func(val), q) for i, val in enumerate(x)]

plot(x, y, f's(t)', f'fs: {fs}Hz q: {q}')

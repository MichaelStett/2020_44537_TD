import matplotlib.pyplot as plt

from math import sin, cos, pi, log10, sqrt

(F, E, D, C, B, A) = (0, 4, 4, 5, 3, 7)

def plot(x, y, title, xlabel ='time', ylabel = 'values',  color = 'r', size = 0.25):
    plt.scatter(x, y, size, color)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def func_x(t):
    return A*t**2 + B*t + C

def func_y(t):
    return 2*func_x(t) ** 2 + 12*cos(t)

def func_z(t):
    return sin(2*pi * 7 * t) * func_x(t) - 0.2*log10(abs(func_y(t) + pi))

def func_u(t):
    return sqrt(abs(func_y(t)*func_y(t)*func_z(t))) - 1.8*sin(0.4 * t * func_z(t) * func_x(t))

def func_v(t):
    if t >= 0 and t < 0.22:
        return (1 - 7*t) * sin((2*pi*t*10)/(t+0.04))
    if t >= 0.22 and t < 0.7:
        return 0.63*t*sin(125*t)
    if t <= 1 and t >= 0.7:
        return t**(-0.662) + 0.77*sin(8*t)

def func_p(t, n):
    for i in range(1, n):
        yield ((cos(12*t*(i**2)) + cos(16*t*i)) / (i**2))

def rangewithstep(start, stop, step):
    x = [start]
    i = 0
    while x[i] < stop:
        x.append(x[i] + step)
        i += 1
    return x

d = B**2 - 4*A*C
if d < 0:
    print('brak miejsc')
elif d == 0:
    print('jedno miejsce zerowe: ' + str(-B / 2*A))
else:
    print('dwa miejsca zerowe: ' + str((-B + sqrt(d))/2*A) + ' & ' + str((-B - sqrt(d))/2*A))

x = rangewithstep(-10, 10, 1 / 100)

y = [func_x(val) for i, val in enumerate(x)]

plot(x, y, 'x(t)', ylabel='x(t)')

x = rangewithstep(0, 1, 1 / 22050)

y = [func_y(val) for i, val in enumerate(x)]

plot(x, y, 'y(t)', ylabel='y(t)')

y = [func_z(val) for i, val in enumerate(x)]

plot(x, y, 'z(t)', ylabel='z(t)')

y = [func_u(val) for i, val in enumerate(x)]

plot(x, y, 'u(t)', ylabel='u(t)')

y = [func_v(val) for i, val in enumerate(x)]

plot(x, y, 'v(t)', ylabel='v(t)')

for n in [2, 4, A*10 + B]:
    y = [sum(func_p(val, n)) for i, val in enumerate(x)]
    plot(x, y, 'p(t,' + str(n) + ')', ylabel='p(t,' + str(n) + ')')

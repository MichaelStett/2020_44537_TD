
import matplotlib.pyplot as plt
from math import floor, ceil, sqrt
from cmath import cos, sin, pi, exp, log10

(F, E, D, C, B, A) = (0, 4, 4, 5, 3, 7)

(Amp, T0, TN, fs, f) = (1.0, 0, 0.01, 100000, 100)

def plot(x, y, title, description = '',
         xlabel ='time', ylabel = 'values',
         xscale='linear', yscale='linear',
         color = 'r', size = 0.25, plottype='scatter'):
    if plottype == 'scatter':
        plt.scatter(x, y, size, color)
    elif plottype == 'stem':
        markerline, stemline, baseline = plt.stem(x, y, markerfmt='o', use_line_collection=True)
        markerline.set_markerfacecolor('none')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(1,4), useMathText=True)
    elif plottype == 'linear':
        plt.plot(x, y)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title + "\n" + description)
    plt.show()

def frange(start, stop, step):
    x = [start]
    i = 0
    while x[i] < stop:
        x.append(x[i] + step)
        i += 1
    return x

def wsp(N):
    return cos(2*pi/N) + 1j*sin(2*pi/N)

def DFT(x: list, N: int):
    # or wsp(N)**(-k*n)
    return [sum([ x[n] * (exp(1j*2*pi/N)**(-k*n)) for n in range(N) if x[n] is not None]) for k in range(N)]

def DFTInvert(x: list, N: int):
    return [sum([ val * (exp(1j*2*pi/N)**(k*i)) for i, val in enumerate(x)]) / N for k in range(N)]

def signal(t):
    return (4*Amp/pi)*sum([sin(i*2*pi*f*t)/i for i in range(1, 49, 2)])

def AmpSpectrum(x: list, N: int):
    # *(2/N) zmienia wysokość na amplitude
    return [sqrt(x[n].real**2 + x[n].imag**2)*(2/N) for n in range(N)]

def dBAmpspectrum(x: list, N: int):
    x = [i.real for i in x]
    threshold = max(x) / 10000
    # print(f'max: {max(x)}')
    # print(f'th: {threshold}')
    y = []
    for n in range(N):
        if x[n].real < threshold:
            y.append(0.0)
        else:
            y.append(10 * log10(x[n].real).real)
    return y

def frequencyScale(fs: float, N: int):
    return [(k * fs / N) for k in range(N)]

# funkcje z lab01
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

# main
x = frange(T0, TN - fs**-1, fs**-1)
N = len(x)
y = [signal(x[n]) for n in range(N)]

dft = DFT(y, N)
idft = DFTInvert(dft, N)
mid = int(N/2)

ampSpectrum = AmpSpectrum(dft, N)[:mid]
dBAmpSpectrum = dBAmpspectrum(ampSpectrum, mid)

# Wykres 1. Testowy sygnał w dziedzinie czasu
plot(x, y, 'wykres podstawowy s(t)')
plot(x, idft, 'wykres s(t) IDFT')

# Wykres 2. Widmo w skali częstotliwości
freqX = frequencyScale(fs, N)[:mid]
plot(freqX, ampSpectrum, "widmo-s  w dziedzinie częst. ", plottype='stem', xlabel='f [Hz]', ylabel='A')

# Wykres 3. Widmo w skali decybelowej
plot(freqX, dBAmpSpectrum, "widmo-s w dziedzinie częst. skala dB", plottype='stem', xlabel='f [Hz]', ylabel='A')

# Wykresy dla funkcji z lab01

# x
fs = 50
x_x = frange(-10, 10, fs**-1)
N = len(x_x)
mid = int(N / 2)

y_x = [func_y(x_x[n]) for n in range(N)]
amp_x = AmpSpectrum(DFT(y_x, N), N)[:mid]
db_x = dBAmpspectrum(amp_x, mid)

freqX = frequencyScale(fs, N)[:mid]
plot(x_x, y_x, 'x(t)')
plot(freqX, amp_x, 'widmo-x w dziedzinie częst.', plottype='stem', xlabel='f [Hz]', ylabel='A')
plot(freqX, db_x, 'widmo-x w dziedzinie częst. skala dB', plottype='stem', xlabel='f [Hz]', ylabel='A')

# yzuvp
fs = 1000
x_yzuvp = frange(0, 1, fs**-1)
N = len(x_yzuvp)
mid = int(N / 2)

y_y = [func_y(x_yzuvp[n]) for n in range(N)]
y_z = [func_z(x_yzuvp[n]) for n in range(N)]
y_u = [func_u(x_yzuvp[n]) for n in range(N)]
y_v = [func_v(x_yzuvp[n]) for n in range(N)]
n = A*10 + B
y_p = [sum(func_p(val, n)) for i, val in enumerate(x_yzuvp)]


amp_y = AmpSpectrum(DFT(y_y, N), N)[:mid]
amp_z = AmpSpectrum(DFT(y_z, N), N)[:mid]
amp_u = AmpSpectrum(DFT(y_u, N), N)[:mid]
amp_v = AmpSpectrum(DFT(y_v, N), N)[:mid]
amp_p = AmpSpectrum(DFT(y_p, N), N)[:mid]

db_y = dBAmpspectrum(amp_y, mid)
db_z = dBAmpspectrum(amp_z, mid)
db_u = dBAmpspectrum(amp_u, mid)
db_v = dBAmpspectrum(amp_v, mid)
db_p = dBAmpspectrum(amp_p, mid)

freqX = frequencyScale(fs, N)[:mid]

plot(x_yzuvp, y_y, 'y(t)')
plot(x_yzuvp, y_z, 'z(t)')
plot(x_yzuvp, y_u, 'u(t)')
# plot(x_yzuvp, y_v, 'v(t)') TypeError: can't convert complex to float???
plot(x_yzuvp, y_p, f'p(t, {n})')

plot(freqX, amp_y, 'widmo-y w dziedzinie częst.', plottype='stem', xlabel='f [Hz]', ylabel='A')
plot(freqX, amp_z, 'widmo-z w dziedzinie częst.', plottype='stem', xlabel='f [Hz]', ylabel='A')
plot(freqX, amp_u, 'widmo-u w dziedzinie częst.', plottype='stem', xlabel='f [Hz]', ylabel='A')
plot(freqX, amp_v, 'widmo-v w dziedzinie częst.', plottype='stem', xlabel='f [Hz]', ylabel='A')
plot(freqX, amp_p, f'widmo-p({n}) w dziedzinie częst.', plottype='stem', xlabel='f [Hz]',  ylabel=f'A')

plot(freqX, db_y, 'widmo-y w dziedzinie częst. skala dB', plottype='stem', xlabel='f [Hz]', ylabel='A')
plot(freqX, db_z, 'widmo-z w dziedzinie częst. skala dB', plottype='stem', xlabel='f [Hz]', ylabel='A')
plot(freqX, db_u, 'widmo-u w dziedzinie częst. skala dB', plottype='stem', xlabel='f [Hz]', ylabel='A')
plot(freqX, db_v, 'widmo-v w dziedzinie częst. skala dB', plottype='stem', xlabel='f [Hz]', ylabel='A')
plot(freqX, db_p, f'widmo-p({n}) w dziedzinie częst. skala dB', plottype='stem',  xlabel='f [Hz]', ylabel=f'A')

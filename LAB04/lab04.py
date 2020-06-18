import matplotlib.pyplot as plt
from math import floor, ceil, sqrt
from cmath import cos, sin, pi, exp, log10

(F, E, D, C, B, A) = (0, 4, 4, 5, 3, 7);

(Amp, T0, TN, fs, f) = (1.0, 0, 1.0, 1000, B)

(fm, fn) = (B, B * 10);

def plot(x, y, title, description = '',
         xlabel ='time', ylabel = 'values',
         xscale='linear', yscale='linear',
         color = 'b', size = 0.25, plottype='scatter'):
    if plottype == 'scatter':
        plt.scatter(x, y, size)
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

def DFT(x: list, N: int):
    return [sum([ x[n] * (exp(1j*2*pi/N)**(-k*n)) for n in range(N) if x[n] is not None]) for k in range(N)]

def DFTInvert(x: list, N: int):
    return [sum([ val * (exp(1j*2*pi/N)**(k*i)) for i, val in enumerate(x)]) / N for k in range(N)]

def signal(t):
    return (4*Amp/pi)*sum([sin(i*2*pi*f*t)/i for i in range(1, 49, 2)])

def AmpSpectrum(x: list, N: int):
    return [sqrt(x[n].real**2 + x[n].imag**2)*(2/N) for n in range(N)]

def dBAmpspectrum(x: list, N: int):
    x = [i.real for i in x]
    threshold = max(x) / 10000
    y = []
    for n in range(N):
        if x[n].real < threshold:
            y.append(0.0)
        else:
            y.append(10 * log10(x[n].real).real)
    return y

def frequencyScale(fs: float, N: int):
    return [(k * fs / N) for k in range(N)]   

def bandwidth(dBx: list, fScale: list):
    N = len(fScale)
    
    fmin = fScale[N - 1]
    fmax = fScale[0]
    
    for i in range(N):
        if dBx[i] >= -3 and dBx[i] != 0:
            if fScale[i] < fmin:
                fmin = fScale[i]

            if fScale[i] > fmax:
                fmax = fScale[i];
    
    if (fmax - fmin) >= 0:
        return fmax - fmin;
    else:
        return 0; 
    
def func_m(Am: float, t: float):
    return Am*sin(2*pi*fm*t);

def modamp(ka: float, ts: float, ms: list, i: int):
    return (ka * ms[i] + 1) * cos(2 * pi * fn * ts[i]);

def modfaz(kp: float, ts: float, ms: list, i: int):
    return cos(2*pi*fn*ts[i] + kp*ms[i]);

def plotAll(ts: list, ys: list, N: int, title):
    mid = int(N/2)
    dft = DFT(ys, N)[:mid]

    ampSpectrum = AmpSpectrum(dft, len(dft))
    dBAmpSpectrum = dBAmpspectrum(ampSpectrum, len(ampSpectrum))

    fScale = frequencyScale(fs, N)[:mid]

    plot(ts, ys, f'{title}', plottype="linear")
    plot(fScale, ampSpectrum, f'{title} widmo w dziedzinie częst. ', plottype='stem', xlabel='f [Hz]', ylabel='A')
    plot(fScale, dBAmpSpectrum, f'{title} widmo w dziedzinie częst. skala dB', plottype='stem', xlabel='f [Hz]', ylabel='A')
    
    width = bandwidth(dBAmpSpectrum, fScale)
    print(f"Bandwidth: {width}")

# Lab 
# 1.

ts = frange(T0, TN - fs**-1, fs**-1)
N = len(ts)

ms = [func_m(1.0, ts[i]) for i in range(N)]

plot(ts, ms, 'm(t) sygnał informacyjny', ylabel="m(t)")

# 2.

# a
(ka, kp) = (0.5, 1.5)

ys = [modamp(ka, ts, ms, i) for i in range(N)]; plotAll(ts, ys, N, f'modulacja amplitudy ka={ka}');

ys = [modfaz(kp, ts, ms, i) for i in range(N)]; plotAll(ts, ys, N, f'modulacja fazy kp={kp}');

# b 
(ka, kp) = (9.5, sqrt(pi / 3))

ys = [modamp(ka, ts, ms, i) for i in range(N)]; plotAll(ts, ys, N, f'modulacja amplitudy ka={ka}');

ys = [modfaz(kp, ts, ms, i) for i in range(N)]; plotAll(ts, ys, N, f'modulacja fazy kp={kp}');

# c 
(ka, kp) = (B*10 + A + 5, A*10 + B + C) 

ys = [modamp(ka, ts, ms, i) for i in range(N)]; plotAll(ts, ys, N, f'modulacja amplitudy ka={ka}');

ys = [modfaz(kp, ts, ms, i) for i in range(N)]; plotAll(ts, ys, N, f'modulacja fazy kp={kp}');

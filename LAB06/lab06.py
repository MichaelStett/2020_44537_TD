import matplotlib.pyplot as plt
from math import floor, ceil, floor, sqrt
from cmath import cos, sin, pi, exp, log10

(F, E, D, C, B, A) = (0, 4, 4, 5, 3, 7)

(N, Tb, fi) = (2, 0.1, 0)

(Amp, T0, TN, fs, f) = (1.0, 0, 1.0, 8 * (N * Tb**-1), N * Tb**-1)

ASK = [0.0, 1.0]
FSK = [(N + 1) / Tb, (N + 2) / Tb]
PSK = [0.0, pi]

(fm, fn) = (B, B * 10)

def plot(x, y, title, description='',
         xlabel='time', ylabel='values',
         xscale='linear', yscale='linear',
         color='b', size=0.25, plottype='scatter'):
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
    return [sum([ x[n] * (exp(1j * 2 * pi / N) ** (-k * n)) for n in range(N) if x[n] is not None]) for k in range(N)]

def DFTInvert(x: list, N: int):
    return [sum([ val * (exp(1j * 2 * pi / N) ** (k * i)) for i, val in enumerate(x)]) / N for k in range(N)]

def signal(t):
    return (4 * Amp / pi) * sum([sin(i * 2 * pi * f * t) / i for i in range(1, 49, 2)])

def AmpSpectrum(x: list, N: int):
    return [sqrt(x[n].real ** 2 + x[n].imag ** 2) * (2 / N) for n in range(N)]

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
                fmax = fScale[i]
    
    if (fmax - fmin) >= 0:
        return fmax - fmin
    else:
        return 0 
    
def func_m(Am: float, t: float):
    return Am * sin(2 * pi * fm * t)

def modamp(ka: float, ts: float, ms: list, i: int):
    return (ka * ms[i] + 1) * cos(2 * pi * fn * ts[i])

def modfaz(kp: float, ts: float, ms: list, i: int):
    return cos(2 * pi * fn * ts[i] + kp * ms[i])

def plotSpectrum(ts: list, ys: list, name):
    N = len(ts)

    mid = int(N / 2)

    dft = DFT(ys, N)[:mid]

    ampSpectrum = AmpSpectrum(dft, len(dft))
    dBAmpSpectrum = dBAmpspectrum(ampSpectrum, len(ampSpectrum))

    fScale = frequencyScale(fs, N)[:mid]

    plot(fScale, ampSpectrum, f'{name}: widmo w dziedzinie częst. ', plottype='stem', xlabel='f [Hz]', ylabel='A')
    plot(fScale, dBAmpSpectrum, f'{name}: widmo w dziedzinie częst. skala dB', plottype='stem', xlabel='f [Hz]', ylabel='A')
    
    width = bandwidth(dBAmpSpectrum, fScale)
    print(f"Bandwidth: {width}")

def func_ASK(bits: str, t, i):
    if bits[i] == '0':
        return (ASK[0] * sin(2 * pi * f * t[i] + 0.0))
    elif bits[i] == '1':
        return (ASK[1] * sin(2 * pi * f * t[i] + 0.0))

def func_FSK(bits: str, t, i):
    if bits[i] == '0':
        return (1.0 * sin(2 * pi * FSK[0] * t[i] + 0.0))
    elif bits[i] == '1':
        return (1.0 * sin(2 * pi * FSK[1] * t[i] + 0.0))

def func_PSK(bits: str, t, i):
    if bits[i] == '0':
        return (1.0 * sin(2 * pi * f * t[i] + PSK[0]))
    elif bits[i] == '1':
        return (1.0 * sin(2 * pi * f * t[i] + PSK[1]))

def text_to_bits(text: str, switch: bool):
    bits = list(map(bin, bytearray(text, 'utf-8')))

    if switch == 'big':
        return [item[:1:-1] for item in bits]
    elif switch == 'little':
        return [item[2:] for item in bits]

def bits_to_string(bits: list):
    return ''.join(bits)

def repeat(signal: list, times: int):
    return [item for item in signal for i in range(times)]

def plotSeparated(signals, ts):
    for i, signal in enumerate(signals):
        plt.plot(ts, signal['values'])
        name = signal['name']
        plt.title(f'{name}: Tb={Tb}, f={f}, fs={fs}, N={N}')
        plt.show()

def plotSpectrums(signals, ts):
    for i, signal in enumerate(signals):
        name = signal['name']
        plotSpectrum(ts, signal['values'], name)

def plotCombined(signals, ts, string):
    x_ticks = frange(T0, TN - Tb, Tb)
    y_ticks = []

    fig, axs = plt.subplots(len(signals), sharex='col', sharey='row')
    fig.suptitle(f'{string}: Tb={Tb}, f={f}, fs={fs}, N={N}')

    plt.subplots_adjust(hspace=.0)
    for i, signal in enumerate(signals):
        axs[i].plot(ts, signal['values'], color=signal['color'])
        axs[i].grid(True, which='both')
        axs[i].axhline(y=signal['hline'], color='k', linestyle=':', linewidth='0.75')
        axs[i].text(signal['label']['x'], signal['label']['y'], signal['name'])
        axs[i].set_xticks(x_ticks)
        axs[i].set_yticks(y_ticks)

    plt.xlabel('t[s]')
    plt.show()

def func(ts, i, c_f = f):
    return (1.0 * sin(2 * pi * c_f * ts[i] + 0.0))

def integral(signal):
    ticks = [round(value, 2)*1000 for value in frange(T0 + Tb, TN - Tb, Tb)]
    ticks_iter = 0;

    result = []

    sum = 0;
    for i, val in enumerate(signal):
        sum += val.real;
        result.append(sum)
        if i == ticks[ticks_iter]:
            #print("zerowanie sumy")
            sum = 0;
            ticks_iter += 1

    return result;

def outputSignal(pt, h):
    mt = []
    for i, val in enumerate(pt):
        if val >= h:
            mt.append('1')
        else:
            mt.append('0')

    return mt


text = "KOT"

# FOR LITTLE ENDIAN
bitsL = text_to_bits(text, 'little')
string = bits_to_string(bitsL)[0:10]

rstring = repeat(string, 100)

(rstring, length) = (rstring, len(rstring))

step = (TN - T0) / float(length)
ts = frange(T0, TN - step, step)

signalBase = [int(val).real for i, val in enumerate(rstring)]
signalA = [func_ASK(rstring, ts, i).real for i in range(length)]
signalF = [func_FSK(rstring, ts, i).real for i in range(length)]
signalP = [func_PSK(rstring, ts, i).real for i in range(length)]

signals = [
    { 'values': signalBase, 'name': 'Base',   'label': {'x': -0.1, 'y': 0.5 }, 'hline': 0.5, 'color': 'darkslategray' },
    { 'values': signalA,    'name': 'ASK(t)', 'label': {'x': -0.1, 'y': 0.0 }, 'hline': 0.0, 'color': 'royalblue' },
    { 'values': signalF,    'name': 'FSK(t)', 'label': {'x': -0.1, 'y': 0.0 }, 'hline': 0.0, 'color': 'lime' },
    { 'values': signalP,    'name': 'PSK(t)', 'label': {'x': -0.1, 'y': 0.0 }, 'hline': 0.0, 'color': 'tomato' }
]

# demodulacja: ASK
signal = [func(ts, i).real for i in range(length)]

xA = [ a*b for a, b in zip(signal, signalA)]

ptA = integral(xA)

mtA = outputSignal(ptA, h=0.001)

signalAdemo = [func_ASK(mtA, ts, i).real for i in range(length)];

signals = [
    { 'values': signalBase,  'name': 'Base',        'label': {'x': -0.10, 'y': 0.5 }, 'hline': 0.5, 'color': 'darkslategray' },
    { 'values': signalA,     'name': 'ASK(t)',      'label': {'x': -0.10, 'y': 0.0 }, 'hline': 0.0, 'color': 'royalblue' },
    { 'values': xA,          'name': 'x(t)',        'label': {'x': -0.10, 'y': 0.0 }, 'hline': 0.0, 'color': 'royalblue' },
    { 'values': ptA,         'name': 'p(t)',        'label': {'x': -0.10, 'y': 0.0 }, 'hline': 0.0, 'color': 'royalblue' },
    { 'values': mtA,         'name': 'm(t)',        'label': {'x': -0.10, 'y': 0.5 }, 'hline': 0.5, 'color': 'royalblue' },
    { 'values': signalAdemo, 'name': 'demo-ASK(t)', 'label': {'x': -0.15, 'y': 0.0 }, 'hline': 0.0, 'color': 'royalblue' }
]

plotCombined(signals, ts, string)

# demodulacja PSK
signal = [func(ts, i).real for i in range(length)]

xP = [ -(a*b) for a, b in zip(signal, signalP)]

ptP = integral(xP)

mtP = outputSignal(ptP, h=0.001)

signalPdemo = [func_PSK(mtP, ts, i).real for i in range(length)];

signals = [
    { 'values': signalBase,  'name': 'Base',        'label': {'x': -0.10, 'y': 0.5 }, 'hline': 0.5, 'color': 'darkslategray' },
    { 'values': signalP,     'name': 'PSK(t)',      'label': {'x': -0.10, 'y': 0.0 }, 'hline': 0.0, 'color': 'lime' },
    { 'values': xP,          'name': 'x(t)',        'label': {'x': -0.10, 'y': 0.0 }, 'hline': 0.0, 'color': 'lime' },
    { 'values': ptP,         'name': 'p(t)',        'label': {'x': -0.10, 'y': 0.0 }, 'hline': 0.0, 'color': 'lime' },
    { 'values': mtP,         'name': 'm(t)',        'label': {'x': -0.10, 'y': 0.5 }, 'hline': 0.5, 'color': 'lime' },
    { 'values': signalPdemo, 'name': 'demo-PSK(t)', 'label': {'x': -0.15, 'y': 0.0 }, 'hline': 0.0, 'color': 'lime' }
]

plotCombined(signals, ts, string)

# demodulacja FSK
signal1 = [func(ts, i, f).real for i in range(length)]
signal2 = [func(ts, i, f*2).real for i in range(length)]

xF1 = [ a*b for a, b in zip(signal1, signalF)]
xF2 = [ a*b for a, b in zip(signal2, signalF)]

xF = [b-a for a,b in zip(xF1, xF2)]

ptF = integral(xF)

mtF = outputSignal(ptF, h = 2.7)

signalFdemo = [func_FSK(mtF, ts, i).real for i in range(length)];

signals = [
    { 'values': signalBase,   'name': 'Base',        'label': {'x': -0.10, 'y': 0.5 }, 'hline': 0.5, 'color': 'darkslategray' },
    { 'values': signalF,      'name': 'FSK(t)',      'label': {'x': -0.10, 'y': 0.0 }, 'hline': 0.0, 'color': 'tomato' },
    { 'values': xF1,          'name': 'x1(t)',       'label': {'x': -0.10, 'y': 0.0 }, 'hline': 0.0, 'color': 'tomato' },
    { 'values': xF2,          'name': 'x2(t)',       'label': {'x': -0.10, 'y': 0.0 }, 'hline': 0.0, 'color': 'tomato' },
    { 'values': xF,           'name': 'x(t)',        'label': {'x': -0.10, 'y': 0.0 }, 'hline': 0.0, 'color': 'tomato' },
    { 'values': ptF,          'name': 'p(t)',        'label': {'x': -0.10, 'y': 0.0 }, 'hline': 0.0, 'color': 'tomato' },
    { 'values': mtF,          'name': 'm(t)',        'label': {'x': -0.10, 'y': 0.5 }, 'hline': 0.5, 'color': 'tomato' },
    { 'values': signalFdemo,  'name': 'demo-FSK(t)', 'label': {'x': -0.15, 'y': 0.0 }, 'hline': 0.0, 'color': 'tomato' }
]

plotCombined(signals, ts, string)
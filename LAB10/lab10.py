import sys
import codecs
import random
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

from time import time
from iteration_utilities import deepflatten
from math import floor, ceil, floor, sqrt
from cmath import cos, sin, pi, exp, log10
from random import randint

SAVE_TO_PNG = True;
WITH_PLOTS = True;

fs = 1000 # Hz

T0 = 0.0; TN = 1.0; Tb = 0.1; # T0:Tb:Tn
N = 2;

# const ASK 
A0 = 0.0; A1 = 1.0; f = N / Tb;

# const PSK 
A = 1.0; f0 = (N + 1) / Tb; f1 = (N + 2) / Tb;

# const FSK 
phi0 = 0.0; phi1 = pi / 2;


def flatten(iterable):
    """
        Funkcja pomocnicza wyrównująca tablice
    """
    try:
        iterator = iter(iterable)
    except TypeError:
        yield iterable
    else:
        for element in iterator:
            yield from flatten(element)


def tile(value, count):
    return [value for _ in range(int(count))]


def linspace(start, stop, count):
    step = (start - stop) / (count - 1)
    values = [start + step * i for i in range(count)]
    return values


def frange(start, stop, step):
    x = [start]
    i = 0
    while x[i] < stop:
        x.append(x[i] + step)
        i += 1
    return x


def BS2S(bits: list):
    """
        Konwerter bitów na string
    """
    bits = ''.join([str(i) for i in bits])

    binary_int = int(bits, 2)
    byte_number = binary_int.bit_length() + 7 // 8
    binary_array = binary_int.to_bytes(byte_number, "little")

    try:
        ascii_text = binary_array.decode()
    except UnicodeDecodeError:
        return None

    return ascii_text


def S2BS(text: str):
    """
        Konwerter str na tablice ascii
    """
    byte_array = text.encode()

    binary_int = int.from_bytes(byte_array, "little")
    binary_string = bin(binary_int)

    x = list(binary_string) 
    x.pop(1) # usunięcie b

    return [int(i) for i in ''.join(x)]


def genCLK(length):
    half = int(fs / 2);

    clk = fs * length * []

    for i in range(length * 2):
        clk[(i * half) : ((i + 1) * half)] = tile((i % 2) == 0, half)

    return clk


def genTTL(ts, bits):
    ttl = fs * len(bits) * []

    for i, val in enumerate(bits):
        ttl[(i * fs) : ((i + 1) * fs)] = tile(val, fs)

    return ttl


def spoil(bits: list, index= -1):
    """ 
    Psuje bit dla podanego indeksu.
        index: 
            -1 -> losowy index
            <0, 6> -> podany index
    """
    if index == -1:
        n = randint(0, 6) # index!!! <0, 6>
        bits[n] = int(not bits[n])  # 0 -> 1 or 1 -> 0
        #print(f'Zepsuje bit: {n + 1}')
    elif index >= 0 and index <= 6:
        n = index
        bits[n] = int(not bits[n])
        #print(f'Zepsuje bit: {n + 1}')

    return bits


def ham74(bits: list):
    """ 
        Funkcja pomocnicza dla kodowania typu Hamming(7, 4)
    """
    (p1, p2, p3) = (
        (bits[1 - 1] + bits[2 - 1] + bits[4 - 1]) % 2,
        (bits[1 - 1] + bits[3 - 1] + bits[4 - 1]) % 2,
        (bits[2 - 1] + bits[3 - 1] + bits[4 - 1]) % 2,
    )
    bits = [p1, p2, bits[1 - 1], p3, bits[1:4]]

    bits = list(flatten(bits))

    return bits


def d_ham74(bits: list):
    """ 
        Funkcja pomocnicza dla dekodowania typu Hamming(7, 4)
    """
    (p1, p2, p3) = (
        (bits[1 - 1] + bits[3 - 1] + bits[5 - 1] + bits[7 - 1]) % 2,
        (bits[2 - 1] + bits[3 - 1] + bits[6 - 1] + bits[7 - 1]) % 2,
        (bits[4 - 1] + bits[5 - 1] + bits[6 - 1] + bits[7 - 1]) % 2,
    )

    n = (p1 * 2 ** 0 + p2 * 2 ** 1 + p3 * 2 ** 2) - 1

    #print(f'Zepsuty bit: {n + 1}')

    bits[n] = int(not bits[n])

    bits = [bits[3-1], bits[4:8]]

    return bits


def HAM74(bits, code = True):
    """
        Hamming 74 kodowanie lub dekodowanie
    """
    if code == True:
        pairs = [(i * 4, i * 4 + 4) for i in range(int(len(bits) / 4))]

        bits = [ham74(bits[x:y]) for (x, y) in pairs]

        bits = [spoil(bit, index=-1) for bit in bits]
    else:
        bits = [d_ham74(bit) for bit in bits]

    return bits;


def ASK(signal: list, ts: list):
    """
        Modulacja amplitudy danego sygnału
    """
    result = []

    for i, val in enumerate(signal):
        if val == 0:
            result.append((A0 * sin(2 * pi * f * ts[i] + 0.0)).real)
        elif val == 1:
            result.append((A1 * sin(2 * pi * f * ts[i] + 0.0)).real)

    return result;


def FSK(signal: list, ts: list):
    """
        Modulacja częstotliwości danego sygnału
    """
    result = []

    for i, val in enumerate(signal):
        if val == 0:
             result.append((A * sin(2 * pi * f0 * ts[i] + 0.0)).real)
        elif val == 1:
             result.append((A * sin(2 * pi * f1 * ts[i] + 0.0)).real)

    return result;
    

def PSK(signal: list, ts: list):
    """
        Modulacja fazy danego sygnału
    """
    result = []

    for i, val in enumerate(signal):
        if val == 0:
            result.append((A * sin(2 * pi * f * ts[i] + phi0)).real)
        elif val == 1:
            result.append((A * sin(2 * pi * f * ts[i] + phi1)).real)

    return result;


def defaultSignal(ts, c_f = f):
    """
        Funkcja generujący domyślny sygnał sinusoidalny
    """
    result = []

    for t in ts:
        result.append((1.0 * sin(2 * pi * c_f * t + 0.0)).real)

    return result;


def integral(signal):
    """
        Funkcja licząca całke w czasie
    """
    times = int(len(ts) / length)

    ticks = [times * (i + 2) for i in range(length)]
    ticks_iter = 0;

    result = []

    sum = 0;
    for i, val in enumerate(signal):
        sum += val.real;
        result.append(sum)
        #print(f'{i}: {ticks_iter}')
        if i >= ticks[ticks_iter]:
            sum = 0;
            ticks_iter += 1

    return result;


def outputSignal(pt, h):
    """
        Funkcja generująca odpowiedź (w zakresie [0,1]) na podstawie podanej całki
    """
    mt = []
    for i, val in enumerate(pt):
        if val >= h:
            mt.append(1)
        else:
            mt.append(0)

    return mt


def DFT(x: list, N: int):
    return [sum([ x[n] * (exp(1j * 2 * pi / N) ** (-k * n)) for n in range(N) if x[n] is not None]) for k in range(N)]


def DFTInvert(x: list, N: int):
    return [sum([ val * (exp(1j * 2 * pi / N) ** (k * i)) for i, val in enumerate(x)]) / N for k in range(N)]


def AmpSpectrum(x: list):
    N = len(x)
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

    if(SAVE_TO_PNG):
        plt.savefig(f"Plots/{title}.pdf")
        plt.close()
    else:
        plt.show()


def plotSpectrum(ts: list, ys: list, name):
    N = len(ts)

    mid = int(N / 2)

    dft = DFT(ys, N)[:mid]

    fScale = frequencyScale(fs, N)[:mid]

    ampSpectrum = AmpSpectrum(dft)
    plot(fScale, ampSpectrum, f'{name}', 'widmo w dziedzinie częst. ', plottype='stem', xlabel='f [Hz]', ylabel='A')
   
    dBAmpSpectrum = dBAmpspectrum(ampSpectrum, len(ampSpectrum))
    plot(fScale, dBAmpSpectrum, f'{name}-dB', 'widmo w dziedzinie częst. skala dB', plottype='stem', xlabel='f [Hz]', ylabel='A')
    
    width = bandwidth(dBAmpSpectrum, fScale)
    print(f"Bandwidth: {width}")


def plotCombined(signals, ts):
    x_ticks = frange(T0, TN, Tb*2)
    y_ticks = []

    fig, axs = plt.subplots(len(signals), sharex='col', sharey='row')

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


def getResult(m):
    skip = frange(len(ts)/length/2, len(ts), len(ts)/length)
    skip = [int(i) for i in skip]

    iter = 0;
    temp = []
    result = []

    for i in skip[:-1]:
        # print(f'{iter}: {i} -> {signal[i]}')

        temp.append(m[i])

        if iter < 6:
            iter += 1;
        else:
            iter = 0;
            result.append(temp)
            temp = []

    return result;


def getNoiseFromFile(fileName):

    _, data = wav.read(fileName)

    limits = [-2**15,  2**16]

    noise = [(item - limits[0])/limits[1] for item in list(data)]

    return noise


def addNoise(signal, noise, alpha, multiplier = 1):
    signalWithNoise = [multiplier*((val * alpha) + ((1 - alpha) * noise[i])) for i, val in enumerate(signal)]

    return signalWithNoise


def modulation(TTL, ts, name):
    if name == 'ASK':
        return ASK(TTL, ts)
    if name == 'PSK':
        return PSK(TTL, ts)
    if name == 'FSK':
        return FSK(TTL, ts)


def demodulation(signalMod, ts, name):
    if name == 'ASK':
        return ASK_demod(signalMod, ts)
    if name == 'PSK':
        return PSK_demod(signalMod, ts)
    if name == 'FSK':
        return FSK_demod(signalMod, ts)


def ASK_demod(signalMod, ts):
    default = defaultSignal(ts)

    x = [ a*b for a, b in zip(default, signalMod)]

    p = integral(x)

    #plt.plot(ts, p)
    #plt.show()
    m = outputSignal(p, h=0.01)

    signalDemod = ASK(m, ts)

    return m, signalDemod


def PSK_demod(signalMod, ts):
    default = defaultSignal(ts)

    x = [ -(a*b) for a, b in zip(default, signalMod)]

    p = integral(x)

    m = outputSignal(p, h=-40)

    signalDemod = PSK(m, ts)

    return m, signalDemod


def FSK_demod(signalMod, ts):
    default1 = defaultSignal(ts)
    default2 = defaultSignal(ts, f*2)

    xF1 = [ a*b for a, b in zip(default1, signalMod)]
    xF2 = [ a*b for a, b in zip(default2, signalMod)]

    x = [b-a for a,b in zip(xF1, xF2)]

    p = integral(x)

    m = outputSignal(p, h=40)

    signalDemod = FSK(m, ts)

    return m, signalDemod


def BER(original, after):
    count = 0

    for o_bit, a_bit in zip(original, after):
        if o_bit != a_bit:
            count += 1

    return count / len(original)


outputs = [
    sys.__stdout__, 
    #codecs.open('lab10.txt', 'w', "utf-8")
]


for output in outputs:
    sys.stdout = output

    sumTime = 0;

    start = time()

    noise = getNoiseFromFile("noise.wav");

    # Początek
    text = "KOT"
    print(f"0. Dane wejściowe: {text}")

    # 1
    bits = S2BS(text)
    print(f"1. Strumień S2BS('{text}') \n{bits}")

    ber_first = bits; # BER #

    # 2
    bits = HAM74(bits) 
    print(f"2. Strumień zakodowany \n{bits}")

    # 3
    bits = list(deepflatten(bits))

    length = len(bits)
    
    ts = linspace(0, Tb * length, fs * length)

    signalCLK = genCLK(length)
    signalTTL = genTTL(ts, bits)

    signalASK = modulation(signalTTL, ts, 'ASK')
    signalPSK = modulation(signalTTL, ts, 'PSK')
    signalFSK = modulation(signalTTL, ts, 'FSK')

    for signal, name in zip([signalASK, signalPSK, signalFSK], ['ASK', 'PSK', 'FSK']):
        # widma przed dodaniem szumu
        if (WITH_PLOTS):
            plotSpectrum(ts, signal, name)

    mods = [
       { 'name': 'ASK', 'values': signalASK, 'alpha': 0.9970 },
       { 'name': 'ASK', 'values': signalASK, 'alpha': 0.9955 }, #  HOT ~8%
       { 'name': 'ASK', 'values': signalASK, 'alpha': 0.95 },
       { 'name': 'PSK', 'values': signalPSK, 'alpha': 0.352 }, # KOt ~4%
       { 'name': 'PSK', 'values': signalPSK, 'alpha': 0.335 }, # kOt ~8%  lub kO ~12%
       { 'name': 'PSK', 'values': signalPSK, 'alpha': 0.325 }, #   ~33% lub [f 25%
       { 'name': 'FSK', 'values': signalFSK, 'alpha': 0.345 }, # COT/NOT/IOT - ~4%
       { 'name': 'FSK', 'values': signalFSK, 'alpha': 0.335 }, # IOP ~8%
       { 'name': 'FSK', 'values': signalFSK, 'alpha': 0.330 }, #  V 25%
    ]

    end = time()
    sumTime += end - start;

    for i, mod in enumerate(mods):
        start = time()
        plotName = f"{mod['name']}_{mod['alpha']}"
        
        print()
        print(f"3.{i + 1} {plotName}")

        # dodanie szumu
        signalMod = addNoise(mod['values'], noise, mod['alpha'])

        # widma po dodaniu szumu
        if (WITH_PLOTS):
            plotSpectrum(ts, signalMod, f"{plotName}")

        m, signalDemod = demodulation(signalMod, ts, mod['name'])

        result = getResult(m)

        print(f"4. Strumień po demodulacji \n{result}")

        bits = HAM74(result, code=False)

        bits = list(deepflatten(bits))
    
        ber_second = bits; # BER #

        print(f"5. Strumień zdekodowany: \n{bits}")

        ber = BER(ber_first, ber_second)
        print(f"BER: {ber}")

        # Koniec
        text = BS2S(bits)

        if (text == None):
            print(f"Nie udało sie zdekodowac sygnału na tekst.")
        else:
            print(f"Dane wyjściowe: {text}")

        end = time()
        sumTime += end - start;

        print(f"Czas obliczeń dla danego przypadku: {end - start}")

    print(f"Czas obliczeń: {sumTime}")

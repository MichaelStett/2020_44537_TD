import sys
import codecs
import matplotlib.pyplot as plt

from iteration_utilities import deepflatten
from math import floor, ceil, floor, sqrt
from cmath import cos, sin, pi, exp, log10
from random import randint

fs = 100 # Hz

T0 = 0.0; TN = 1.0; Tb = 0.1; # T0:Tb:Tn
N = 2;

f = N / Tb;

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





def frange(start, stop, step):
    x = [start]
    i = 0
    while x[i] < stop:
        x.append(x[i] + step)
        i += 1
    return x


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

    bits = [bits[3 - 1], bits[4:8]]

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


def plotCombined(signals):
    x_ticks = frange(T0, Tb * len(bits), Tb*2)
    y_ticks = []

    fig, axs = plt.subplots(len(signals), sharex='col', sharey='row')

    plt.subplots_adjust(hspace=.1)
    for i, signal in enumerate(signals):
        axs[i].plot(signal['x'], signal['y'], color=signal['color'])
        axs[i].grid(True, which='both')
        axs[i].axhline(y=signal['hline'], color='k', linestyle=':', linewidth='0.75')
        axs[i].text(signal['label']['x'], signal['label']['y'], signal['name'])
        #axs[i].set_xticks(x_ticks)
        axs[i].set_yticks(y_ticks)

    plt.xlabel('t[s]')
    plt.show()


def getResult(mX, ts):
    skip = frange(len(ts)/length/2, len(ts), len(ts)/length)
    skip = [int(i) for i in skip]
    iter = 0;
    temp = []
    result = []

    for i in skip[:-1]:
        if mX[i] == True:
            temp.append(1)
        else:
            temp.append(0)

        if iter < 6:
            iter += 1;
        else:
            iter = 0;
            result.append(temp)
            temp = []

    return result;


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


def NRZI(CLK, TTL):
    nrzi = [0]

    value = 0

    for i in range(len(CLK) - 1):
        if (CLK[i] == 1 and CLK[i + 1] == 0):
            if value == 0:
                value = -1
            else:
                if (TTL[i]):
                    value *= -1;

        nrzi.append(value)

    return nrzi


def NRZI_Decode(nrzi):
    tf = [i == 1 for i in nrzi]

    length = len(nrzi)

    (start, stop, step) = (
        Tb / 2,
        (length / fs) * (Tb) - Tb / 2,
        length - fs
    )

    time = linspace(start, stop, step)

    ttl = [ tf[i - fs] ^ tf[i] for i in range(int(fs), length)]

    return time, ttl


def BAMI(CLK, TTL):
    bami = [0]

    value = 0
    high = -1

    for i in range(len(CLK) - 1):
        if (CLK[i] == 0 and CLK[i + 1] == 1):
            if (TTL[i + 1]):
                value = high
                high *= -1
            else:
                value = 0

        bami.append(value)

    return bami


def BAMI_Decode(bami):
    ttl = [not (sample == 0) for sample in bami]

    return ttl;


def MANCHESTER(CLK, TTL):
    man = [0]

    value = 0

    for i in range(len(CLK) - 1):
        if (CLK[i] == 1 and CLK[i + 1] == 0):
            if (TTL[i] == 0):
                value = 1
            else:
                value = -1
        elif (CLK[i] == 0 and CLK[i + 1] == 1):
            if (TTL[i] == TTL[i + 1]):
                value *= -1

        man.append(value)

    return man


def MANCHESTER_Decode(CLK, man, ts):
    quarter = int(fs / 4)

    CLK = tile(1, quarter) + CLK
    CLK = CLK[0:int(len(CLK) - quarter)]

    bits = []

    for i in range(len(CLK) - 1):
        if (CLK[i] == 1 and CLK[i + 1] == 0):
            bits.append(man[i])

    ttl = [-i for i in genTTL(ts, bits)]

    return ttl

outputs = [
    #sys.__stdout__, 
    codecs.open('lab9.txt', 'w', "utf-8")
]


for output in outputs:
    sys.stdout = output

    print(f'LAB 9: Tor transmisyjny - Michał Tymejczyk 44537\n')

    # Początek
    text = "ABC"
    print(f"Dane wejściowe: {text}")

    # 1
    bits = S2BS(text)
    print(f"Strumień 1 - S2BS({text}) \n{bits}")

    ## 2
    bits = HAM74(bits)
    print(f"Strumień 2 - zakodowany \n{bits}")
    
    # 3
    bits = list(deepflatten(bits))

    ts = [-i for i in linspace(0, Tb * len(bits), fs * len(bits))]

    length = len(bits)

    signalCLK = genCLK(length)
    signalTTL = genTTL(ts, bits)

    signalNRZI = NRZI(signalCLK, signalTTL)
    signalBAMI = BAMI(signalCLK, signalTTL)
    signalMAN =  MANCHESTER(signalCLK, signalTTL)

    time, signalNRZI_Decoded = NRZI_Decode(signalNRZI)
    time = [-i for i in time]

    signalBAMI_Decoded = BAMI_Decode(signalBAMI)
    signalMAN_Decoded = MANCHESTER_Decode(signalCLK, signalMAN, ts)

    # podsumowanie
    signals = [
        #{ 'x': ts, 'y': signalCLK,  'name': 'CLK',  'label': {'x': -0.1, 'y': 0.5 }, 'hline': 0.5, 'color': 'darkslategray' },
        { 'x': ts, 'y': signalTTL,  'name': 'TTL',  'label': {'x': -0.2, 'y': 0.5 }, 'hline': 0.0, 'color': 'darkslategray' },
        #{ 'x': ts, 'y': signalNRZI, 'name': 'NRZI', 'label': {'x': -0.1, 'y': 0.0 }, 'hline': 0.0, 'color': 'royalblue' },
        { 'x': time, 'y': signalNRZI_Decoded, 'name': 'NRZI', 'label': {'x': -0.2, 'y': 0.5 }, 'hline': 0.0, 'color': 'royalblue' },
        #{ 'x': ts, 'y': signalBAMI, 'name': 'BAMI', 'label': {'x': -0.1, 'y': 0.0 }, 'hline': 0.0, 'color': 'lime' },
        { 'x': ts, 'y': signalBAMI_Decoded, 'name': 'BAMI', 'label': {'x': -0.2, 'y': 0.5 }, 'hline': 0.0, 'color': 'lime' },
        #{ 'x': ts, 'y': signalMAN,  'name': 'MAN',  'label': {'x': -0.1, 'y': 0.0 }, 'hline': 0.0, 'color': 'tomato' },
        { 'x': ts, 'y': signalMAN_Decoded,  'name': 'MAN',  'label': {'x': -0.2, 'y': 0.0 }, 'hline': -1.0, 'color': 'tomato' },
    ]

    plotCombined(signals)

    # 4 z ASK
    result = getResult(signalMAN_Decoded, ts)

    print(f"Strumień 3 - po demodulacji \n{result}")

    bits = HAM74(result, code=False)

    bits = list(deepflatten(bits))
    
    print(f"Strumień 4 - zdekodowany: \n{bits}")

    # Koniec
    text = BS2S(bits)
    print(f"Dane wyjściowe: {text}")
    
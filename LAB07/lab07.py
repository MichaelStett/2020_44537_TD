import matplotlib.pyplot as plt
from math import floor, ceil, floor, sqrt
from cmath import cos, sin, pi, exp, log10

fs = 250 # Hz

T0 = 0.0; TN = 1.0; Tb = 0.1; # T0:Tb:Tn

N = 2;

def tile(value, count):
    return [value for _ in range(int(count))]

def linspace(start, stop, count):
    step = (start - stop) / (count - 1)
    values = [start + step * i for i in range(count)]
    return values


def BS2S(bits: list):
    bits = ''.join([str(i) for i in bits])

    binary_int = int(bits, 2)
    byte_number = binary_int.bit_length() + 7 // 8
    binary_array = binary_int.to_bytes(byte_number, "big")

    ascii_text = binary_array.decode()

    return ascii_text


def S2BS(text: str):
    byte_array = text.encode()

    binary_int = int.from_bytes(byte_array, "big")
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


def plotCombined(signals):

    x_lim = [ min(signals[0]['x']) - Tb,  max(signals[0]['x']) + Tb]
    y_ticks = []

    fig, axs = plt.subplots(len(signals), sharex='col', sharey='row')

    plt.subplots_adjust(hspace=.0)

    for i, signal in enumerate(signals):
        axs[i].plot(signal['x'], signal['y'], color=signal['color'])
        axs[i].grid(True, which='both')
        axs[i].axhline(y=signal['hline'], color='k', linestyle=':', linewidth='0.75')
        axs[i].text(signal['label']['x'], signal['label']['y'], signal['name'])
        #axs[i].set_xlim(x_lim)
        axs[i].set_yticks(y_ticks)

    plt.xlabel('t[s]')
    plt.show()


text = "KOT"

bits = S2BS(text)

ts = linspace(0, Tb * len(bits), fs * len(bits))

length = len(bits)

signalCLK = genCLK(length)
signalTTL = genTTL(ts, bits)

signalNRZI = NRZI(signalCLK, signalTTL)
time, signalNRZI_Decoded = NRZI_Decode(signalNRZI)

signalBAMI = BAMI(signalCLK, signalTTL)
signalBAMI_Decoded = BAMI_Decode(signalBAMI)

signalMAN = MANCHESTER(signalCLK, signalTTL)
signalMAN_Decoded = MANCHESTER_Decode(signalCLK, signalMAN, ts)


signals = [
    { 'x': ts,   'y': signalCLK,          'name': 'CLK',      'label': {'x': -0.25, 'y': 0.5 }, 'hline':  0.0, 'color': 'darkslategray' },
    { 'x': ts,   'y': signalTTL,          'name': 'TTL',      'label': {'x': -0.25, 'y': 0.5 }, 'hline':  0.0, 'color': 'royalblue' },
    { 'x': ts,   'y': signalNRZI,         'name': 'NRZI',     'label': {'x': -0.25, 'y': 0.0 }, 'hline':  0.0, 'color': 'darkslategray' },
    { 'x': time, 'y': signalNRZI_Decoded, 'name': 'TTL-NRZI', 'label': {'x': -0.25, 'y': 0.5 }, 'hline':  0.0, 'color': 'royalblue' },
    { 'x': ts,   'y': signalBAMI,         'name': 'BAMI',     'label': {'x': -0.25, 'y': 0.0 }, 'hline':  0.0, 'color': 'fuchsia' },
    { 'x': ts,   'y': signalBAMI_Decoded, 'name': 'TTL-BAMI', 'label': {'x': -0.25, 'y': 0.5 }, 'hline':  0.0, 'color': 'royalblue' },
    { 'x': ts,   'y': signalMAN,          'name': 'MAN',      'label': {'x': -0.25, 'y': 0.0 }, 'hline':  0.0, 'color': 'lime' },
    { 'x': ts,   'y': signalMAN_Decoded,  'name': 'TTL-MAN',  'label': {'x': -0.25, 'y': 0.0 }, 'hline': -1.0, 'color': 'royalblue' },
]


plotCombined(signals)

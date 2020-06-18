import sys
import codecs

from random import randint

# helper
def flatten(iterable):
    try:
        iterator = iter(iterable)
    except TypeError:
        yield iterable
    else:
        for element in iterator:
            yield from flatten(element)


def text_to_bits(text: str, switch: bool):
    bits = list(map(bin, bytearray(text, 'utf-8')))

    if switch == 'big':
        ret = [item[:1:-1] for item in bits]
    elif switch == 'little':
        ret = [item[2:] for item in bits]
    else: 
        exit(-1)

    return [item for item in ret]


def bits_to_list(bits: list, length: int):
    if length <= 0 or length > 8*len(bits):
        print(f'Index {length} out of range. List length: {8*len(bits)}')
        exit(-1)

    return [ int(item) for item in ''.join(bits)[0:length]]


def spoil(bits: list, index= -1):
    """ 
    Spoils index in list.
        index: 
            -1 -> random index
            <0, 6> -> exact index
    """
    if index == -1:
        n = randint(0, 6) # index!!! <0, 6>
        print(f'Zepsuje bit: {n + 1}')
        bits[n] = int(not bits[n])  # 0 -> 1 or 1 -> 0
    elif index >= 0 and index < 7:
        n = index
        print(f'Zepsuje bit: {n + 1}')
        bits[n] = int(not bits[n])

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

    bits = list(flatten([p1, p2, bits[1 - 1], p3, bits[1:4]]))

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

    print(f'Zepsuty bit: {n + 1}')

    bits[n] = int(not bits[n])

    bits = [bits[3 - 1], bits[4:8]]

    return bits


def ham84(bits: list):
    """ 
        Funkcja pomocnicza dla kodowania typu Hamming(8, 4) / SECDED
    """
    (p1, p2, p3) = (
        (bits[1 - 1] + bits[2 - 1] + bits[4 - 1]) % 2,
        (bits[1 - 1] + bits[3 - 1] + bits[4 - 1]) % 2,
        (bits[2 - 1] + bits[3 - 1] + bits[4 - 1]) % 2,
    )

    bits = list(flatten([p1, p2, bits[1 - 1], p3, bits[1:]]))

    p4 = sum(bits) % 2

    print(f'bit parzystości: {p4}')

    bits = list(flatten([bits, p4]))

    return bits


def d_ham84(bits: list, repair=True):
    print()

    p4 = bits[-1]
    expected_p4 = sum(bits[:-1]) % 2
    # print(f'bf: {p4} <=> {expected_p4}')

    if expected_p4 != p4 and repair == False:
        print("\nZepsuty p4!")
        return [] * 4
    else: 
        # Jeżeli p4 zgodne wylicz p1, p2, p3
        (p1, p2, p3) = (
            (bits[1 - 1] + bits[3 - 1] + bits[5 - 1] + bits[7 - 1]) % 2,
            (bits[2 - 1] + bits[3 - 1] + bits[6 - 1] + bits[7 - 1]) % 2,
            (bits[4 - 1] + bits[5 - 1] + bits[6 - 1] + bits[7 - 1]) % 2,
        )

        # Krok 3. Wyznaczenie indeksu korekty
        n = (p1 * 2 ** 0 + p2 * 2 ** 1 + p3 * 2 ** 2)

        print(f'\nZepsuty bit: {n + 1}')

        # Krok 4 Negacja bitu
        bits[n] = int(not bits[n])

        print(f'Przed odpsuciem: {bits}')

        bits = list(flatten([bits[3 - 1], bits[4:7]]))

        # Krok 5 ponowna weryfikacja
        p4 = bits[-1]
        expected_p4 = sum(bits[:-1]) % 2
        # print(f'af: {p4} <=> {expected_p4}')

        if expected_p4 != p4:
            print("Zepsuty p4! Trzeba przesłać ponownie")
            return [] * 4

        print(f'Po odpsuciu: {bits}')
        return bits


def scopePairs(bits: list):
    step = int(len(bits) / 4)
    return [(i * 4, i * 4 + 4) for i in range(step)]


def HAM74(text: str):
    print("HAM(7, 4): ")

    print(f'Tekst: {text}.')
    print(f'Liczba bitów: {8*len(text)}.')

    length = 12

    print(f'Wybrana liczba bitów: {length}.')

    bits = text_to_bits(text, 'little')
    bits = bits_to_list(bits, length)

    print(f'\nPrzed kodowaniem: {bits}')

    pairs = scopePairs(bits)

    print(f'\nZbiór indexów podziału bitów: {pairs}')

    print('\nBity podzielone: ')
    for (x, y) in pairs:
        print(f'{bits[x:y]}')

    hs = [ham74(bits[x:y]) for (x, y) in pairs]
    print(f'\nPo kodowaniu: {hs}')

    hs = [spoil(h, index=-1) for h in hs]
    print(f'\nPo zepsuciu: {hs}')

    ds = [d_ham74(h) for h in hs]
    ds = list(flatten(ds))
    print(f'\nPo dekodowaniu: {ds}')


def HAM84(text: str):
    print("SECDED: ")
    print(f'Tekst: {text}.')
    print(f'Podstawowa liczba bitów: {8*len(text)}.')

    length = 12

    print(f'Wybrana liczba bitów: {length}.')

    bits = text_to_bits(text, 'little')
    bits = bits_to_list(bits, length)

    print(f'\nPrzed kodowaniem: {bits}')

    pairs = scopePairs(bits)

    print(f'\nZbiór indexów podziału bitów: {pairs}')

    print('\nBity podzielone: ')
    for (x, y) in pairs:
        print(f'{bits[x:y]}')

    hs = [ham84(bits[x:y]) for (x, y) in pairs]
    print(f'\nPo kodowaniu: {hs}')

    hs = [spoil(h, index=-1) for h in hs]
    print(f'\nPo zepsuciu: {hs}')

    hs = [spoil(h, index=-1) for h in hs]
    print(f'\nPo zepsuciu: {hs}')

    ds = [d_ham84(h) for h in hs]
    ds = list(flatten(ds))
    print(f'\nPo dekodowaniu: {ds}')


outputs = [
    sys.__stdout__, 
    codecs.open('hamming.txt', 'w', "utf-8")
]

print(f'LAB 8: Kod Hamminga - Michał Tymejczyk 44537\n')


for output in outputs:
    sys.stdout = output

    text = "KOT"

    HAM74(text)
    print(" \n = = = = = = = = = = = = = = = = \n")
    HAM84(text)
 

import numpy as np


def _neighbors(image):
    padded = np.pad(image, 1, mode="constant")
    p2 = padded[:-2, 1:-1]
    p3 = padded[:-2, 2:]
    p4 = padded[1:-1, 2:]
    p5 = padded[2:, 2:]
    p6 = padded[2:, 1:-1]
    p7 = padded[2:, :-2]
    p8 = padded[1:-1, :-2]
    p9 = padded[:-2, :-2]
    return p2, p3, p4, p5, p6, p7, p8, p9


def zhang_suen(image, dark_foreground=True):

    values = np.asarray(image.convert("L"))
    skeleton = values == (0 if dark_foreground else 255)

    while True:
        changed = False
        for step in (0, 1):
            p2, p3, p4, p5, p6, p7, p8, p9 = _neighbors(skeleton)
            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = sum(neighbors)
            transitions = sum(
                (~neighbors[i] & neighbors[(i + 1) % 8]).astype(np.uint8)
                for i in range(8)
            )
            remove = skeleton & (count >= 2) & (count <= 6) & (transitions == 1)
            if step == 0:
                remove &= ~(p2 & p4 & p6)
                remove &= ~(p4 & p6 & p8)
            else:
                remove &= ~(p2 & p4 & p8)
                remove &= ~(p2 & p6 & p8)
            if np.any(remove):
                skeleton[remove] = False
                changed = True
        if not changed:
            break
    return np.where(skeleton, 0, 255).astype(np.uint8)


def calibrated_thinning(image, dark_foreground=True):
    values = np.asarray(image.convert("L"))
    skeleton = values == (0 if dark_foreground else 255)

    removable_patterns = np.array([
        5, 7, 13, 15, 20, 21, 22, 23, 28, 29, 30, 31, 52, 53, 54, 55,
        60, 61, 62, 63, 65, 67, 69, 71, 77, 79, 80, 81, 83, 84, 85, 86,
        87, 88, 89, 91, 92, 93, 94, 95, 97, 99, 101, 103, 109, 111, 112,
        113, 115, 116, 117, 118, 119, 120, 121, 123, 124, 125, 126, 127,
        133, 135, 141, 143, 149, 151, 157, 159, 181, 183, 189, 191, 193,
        195, 197, 199, 205, 207, 208, 209, 211, 212, 213, 214, 215, 216,
        217, 219, 220, 221, 222, 223, 225, 227, 229, 231, 237, 239, 240,
        241, 243, 244, 245, 246, 247, 248, 249, 251, 252, 253, 254, 255,
    ], dtype=np.uint16)

    pattern_lookup = np.zeros(256, dtype=bool)
    pattern_lookup[removable_patterns] = True

    interior = np.zeros_like(skeleton) #pixel wewnetrzny
    interior[1:-1, 1:-1] = True

    calibrations = (
        (
            lambda b, t: (t < 3) & (b < 5),
            lambda b, t: (t < 5) & (b > 4),
            lambda b, t: (t < 3) & (b > 2),
        ),
        (
            lambda b, t: (t < 3) & (b < 6),
            lambda b, t: (t < 4) & (b > 2) & (b < 6),
            lambda b, t: t == 2,
        ),
        (
            lambda b, t: (t < 4) & (b < 6),
            lambda b, t: (t < 4) & (b > 2) & (b < 6),
            lambda b, t: t < 3,
        ),
        (
            lambda b, t: b > 1,
            lambda b, t: t < 4,
            lambda b, t: np.ones_like(b, dtype=bool),
        ),
    )

    def metrics(image):
        neighbors = _neighbors(image)
        count = sum(neighbors)
        transitions = sum(
            (neighbors[index] != neighbors[(index + 1) % 8]).astype(np.uint8)
            for index in range(8)
        ) #przejscie z 0 na 1 w okolicy pixela srodkowego
        pattern = (
            neighbors[0].astype(np.uint16)
            + neighbors[1].astype(np.uint16) * 2
            + neighbors[2].astype(np.uint16) * 4
            + neighbors[3].astype(np.uint16) * 8
            + neighbors[4].astype(np.uint16) * 16
            + neighbors[5].astype(np.uint16) * 32
            + neighbors[6].astype(np.uint16) * 64
            + neighbors[7].astype(np.uint16) * 128
        )
        return count, transitions, pattern

    for first_condition, second_condition, third_condition in calibrations: # odpowiada to f1, f2 i f3
        while True:
            count, transitions, pattern = metrics(skeleton)
            gray = skeleton & interior & first_condition(count, transitions) #oznaczy to jako pixel szary
            if not np.any(gray):
                break

            marked_for_deletion = gray & second_condition(count, transitions) #dla tego szarego pixela jak spełni 2 warunek to usuniecie
            removed = marked_for_deletion & third_condition(count, transitions)
            # dla oznaczpnego pixela szarego, jeśli spełni 3 warunek to usuwamy go zmieniajac na biały
            if np.any(removed):
                skeleton[removed] = False

            remaining_gray = gray & ~removed #pozostałe szare usuwamy przez matrycę podana przez pana Patryka
            count, transitions, pattern = metrics(skeleton)

            removed_by_pattern = (
                remaining_gray & pattern_lookup[pattern.astype(np.uint8)]
            )
            if np.any(removed_by_pattern):
                skeleton[removed_by_pattern] = False

            if not np.any(removed) and not np.any(removed_by_pattern):
                break

    return np.where(skeleton, 0, 255).astype(np.uint8)
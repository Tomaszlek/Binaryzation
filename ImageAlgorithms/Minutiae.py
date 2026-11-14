import numpy as np


def detect_minutiae(image, min_distance=5): #pobawić się ze zmianami w min_distance
    padded = np.pad(image, 1, mode="constant")

    neighbors = (
        padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:],
        padded[2:, 2:], padded[2:, 1:-1], padded[2:, :-2],
        padded[1:-1, :-2], padded[:-2, :-2],
    )
    transitions = sum(
        (neighbors[index] != neighbors[(index + 1) % 8]).astype(np.uint8)
        for index in range(8)
    )

    crossing_number = transitions / 2
    endings = image & (crossing_number == 1)
    bifurcations = image & (crossing_number == 3)

    if min_distance > 1:
        for candidates in (endings, bifurcations):
            filtered = np.zeros_like(candidates)
            suppressed = np.zeros_like(candidates)

            for y, x in zip(*np.nonzero(candidates)):
                if suppressed[y, x]:
                    continue

                filtered[y, x] = True

                y0, y1 = max(0, y - min_distance), min(candidates.shape[0], y + min_distance + 1)
                x0, x1 = max(0, x - min_distance), min(candidates.shape[1], x + min_distance + 1)

                suppressed[y0:y1, x0:x1] = True

            candidates[:] = filtered

    points = []
    for kind, mask in (("ending", endings), ("bifurcation", bifurcations)):
        points.extend((int(x), int(y), kind) for y, x in zip(*np.nonzero(mask)))
    return points


def annotate_minutiae(image, minutiae):
    result = np.asarray(image.convert("RGB")).copy()

    if result.ndim == 2:
        result = np.repeat(result[:, :, None], 3, axis=2)

    for x, y, kind in minutiae:
        color = np.array([255, 0, 0] if kind == "ending" else [0, 0, 255])

        y0, y1 = max(0, y - 2), min(result.shape[0], y + 3)
        x0, x1 = max(0, x - 2), min(result.shape[1], x + 3)

        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= 4
        result[y0:y1, x0:x1][disk] = color
    return result

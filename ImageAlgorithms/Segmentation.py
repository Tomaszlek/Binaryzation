from collections import deque

import numpy as np
from scipy import ndimage


def color_selection_mask(image, x, y, max_pixels=1000, tolerance_min=0, tolerance_max=30, global_mode=False):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Wybór kolorem wymaga obrazu RGB.")
    
    height, width = image.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Punkt wyboru znajduje się poza obrazem.")
    if max_pixels <= 0:
        raise ValueError("Maksymalna liczba pikseli musi być większa od zera.")
    if not (0 <= tolerance_min <= tolerance_max <= 255):
        raise ValueError("Zakres tolerancji musi spełniać 0 <= min <= max <= 255.")

    target = image[y, x, :3].astype(np.int16)
    difference = np.abs(image[:, :, :3].astype(np.int16) - target)
    eligible = np.all(
        (difference >= tolerance_min) & (difference <= tolerance_max), axis=2
    )
    eligible[y, x] = True

    if global_mode:
        locations = np.flatnonzero(eligible)
        mask = np.zeros((height, width), dtype=bool)
        seed = y * width + x
        mask.flat[seed] = True
        other_locations = locations[locations != seed]
        mask.flat[other_locations[:max_pixels - 1]] = True
        return mask

    mask = np.zeros((height, width), dtype=bool)
    visited = np.zeros((height, width), dtype=bool)

    queue = deque([(x, y)])
    visited[y, x] = True
    selected = 0

    while queue and selected < max_pixels:
        cx, cy = queue.popleft()
        if not eligible[cy, cx]:
            continue
        mask[cy, cx] = True
        selected += 1
        for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
            if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((nx, ny))
    return mask


def resize_selection(mask, operation, iterations=1):
    if iterations < 1:
        raise ValueError("Liczba iteracji musi być większa od zera.")
    if operation not in ("dilate", "erode"):
        raise ValueError("Operacja musi być 'dilate' lub 'erode'.")
    structure = np.ones((3, 3), dtype=bool)
    function = ndimage.binary_dilation if operation == "dilate" else ndimage.binary_erosion
    return function(np.asarray(mask, dtype=bool), structure=structure, iterations=iterations)


def watershed_segmentation(image):
    gray = np.asarray(image.convert("L"))

    if gray.ndim != 2:
        raise ValueError("Watershed wymaga obrazu w odcieniach szarości.")
    
    gradient = np.hypot(
        ndimage.sobel(gray.astype(np.float32), axis=0),
        ndimage.sobel(gray.astype(np.float32), axis=1),
    )

    if gradient.max() > 0:
        gradient = (gradient * (255.0 / gradient.max())).astype(np.uint8)
    else:
        gradient = np.zeros_like(gray, dtype=np.uint8)

    markers = np.zeros(gray.shape, dtype=np.int32)
    markers[gray <= np.percentile(gray, 20)] = 1
    markers[gray >= np.percentile(gray, 80)] = 2
    labels = ndimage.watershed_ift(gradient, markers)
    return labels

from PIL import Image
import numpy as np
from collections import deque


class ImageEditor:
    def __init__(self,):
        pass
        #self.image = Image.open(image_path)
        #self.img_array = np.array(self.image)

    def save_image(self, path):
        result_image = Image.fromarray(self.img_array.astype(np.uint8))
        result_image.save(path)

    def magic_wand(self, x, y, tolerance=30, max_pixels=1000):
        target_color = self.img_array[y, x].copy()
        height, width, _ = self.img_array.shape
        visited = np.zeros((height, width), dtype=bool)
        queue = deque([(x, y)])
        visited[y, x] = True
        pixel_count = 0

        while queue and pixel_count < max_pixels:
            cx, cy = queue.popleft()
            if self._color_match(self.img_array[cy, cx], target_color, tolerance):
                self.img_array[cy, cx] = [255, 0, 0]  # Wypełnienie kolorem (czerwony dla wizualizacji)
                pixel_count += 1

                # Sprawdzenie sąsiednich pikseli
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))

    def pencil_draw(self, x, y, color=[0, 0, 0], size=1):
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                if 0 <= x + dx < self.img_array.shape[1] and 0 <= y + dy < self.img_array.shape[0]:
                    self.img_array[y + dy, x + dx] = color

    def fill(self, x, y, color, global_mode=False, tolerance=30, max_pixels=1000):
        target_color = self.img_array[y, x].copy()
        if np.array_equal(target_color, color):
            return

        height, width, _ = self.img_array.shape
        visited = np.zeros((height, width), dtype=bool)
        queue = deque([(x, y)])
        visited[y, x] = True
        pixel_count = 0

        while queue and pixel_count < max_pixels:
            cx, cy = queue.popleft()
            if global_mode or self._color_match(self.img_array[cy, cx], target_color, tolerance):
                self.img_array[cy, cx] = color
                pixel_count += 1

                # Sprawdzenie sąsiednich pikseli
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))

    def erode(self, iterations=1):
        self._morphological_operation(iterations, mode='erode')

    def dilate(self, iterations=1):
        self._morphological_operation(iterations, mode='dilate')

    def _morphological_operation(self, iterations, mode='erode'):
        for _ in range(iterations):
            new_img = self.img_array.copy()
            height, width, _ = self.img_array.shape

            for y in range(height):
                for x in range(width):
                    if mode == 'erode':
                        # Erozja
                        if np.all(self.img_array[y, x] == [255, 255, 255]):  # Biały piksel
                            new_img[y, x] = [255, 255, 255]
                        else:
                            new_img[y, x] = [0, 0, 0]
                    elif mode == 'dilate':
                        # Dylatacja
                        if any(self.img_array[max(y + dy, 0), max(x + dx, 0)] != [0, 0, 0] for dy in [-1, 0, 1] for dx
                               in [-1, 0, 1]):
                            new_img[y, x] = [0, 0, 0]
            self.img_array = new_img

    def _color_match(self, color1, color2, tolerance):
        return np.all(np.abs(color1 - color2) <= tolerance)

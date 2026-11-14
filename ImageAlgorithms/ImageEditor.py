from PIL import Image
import numpy as np
from collections import deque


class ImageEditor:
    def __init__(self, image_path=None):
        if image_path:
            self.image = Image.open(image_path)
            self.img_array = np.array(self.image, dtype=np.uint8)

    def save_image(self, path):
        result_image = Image.fromarray(self.img_array)
        result_image.save(path)

    def magic_wand(self, x, y, tolerance=30, max_pixels=1000):
        target_color = self.img_array[y, x]
        mask = self._get_flood_mask(x, y, target_color, tolerance, max_pixels)
        self.img_array[mask] = [255, 0, 0]  # wypełnienie kolorem czerwonym, potem sie zmieni na wybrany kolor w GUI

    def pencil_draw(self, x, y, color=[0, 0, 0], size=1):
        height, width = self.img_array.shape[:2]
        yy, xx = np.ogrid[-size:size + 1, -size:size + 1]
        
        mask = xx ** 2 + yy ** 2 <= size ** 2

        for dy, dx in zip(*np.where(mask)):
            px, py = x + dx - size, y + dy - size
            if 0 <= px < width and 0 <= py < height:
                self.img_array[py, px] = color

    def fill(self, x, y, color, global_mode=False, tolerance=30, max_pixels=1000):
        target_color = self.img_array[y, x]
        if np.array_equal(target_color, color):
            return

        if global_mode:
            mask = self._get_global_flood_mask(target_color, tolerance)
        else:
            mask = self._get_flood_mask(x, y, target_color, tolerance, max_pixels)
        self.img_array[mask] = color

    def erode(self, iterations=1):
        for _ in range(iterations):
            self.img_array = self._morphological_operation(self.img_array, mode='erode')

    def dilate(self, iterations=1):
        for _ in range(iterations):
            self.img_array = self._morphological_operation(self.img_array, mode='dilate')

    def _get_flood_mask(self, x, y, target_color, tolerance, max_pixels):
        height, width = self.img_array.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        queue = deque([(x, y)])
        mask[y, x] = True
        pixel_count = 0

        while queue and pixel_count < max_pixels:
            cx, cy = queue.popleft()
            if self._color_match(self.img_array[cy, cx], target_color, tolerance):
                pixel_count += 1
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height and not mask[ny, nx]:
                        mask[ny, nx] = True
                        queue.append((nx, ny))
        return mask

    def _get_global_flood_mask(self, target_color, tolerance):
        diff = np.abs(self.img_array - target_color)
        return np.all(diff <= tolerance, axis=-1)

    def _morphological_operation(self, image, mode='erode'):
        height, width, _ = image.shape
        padded_image = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        new_image = image.copy()

        for y in range(1, height + 1):
            for x in range(1, width + 1):
                region = padded_image[y - 1:y + 2, x - 1:x + 2]
                if mode == 'erode':
                    new_image[y - 1, x - 1] = np.min(region, axis=(0, 1))
                elif mode == 'dilate':
                    new_image[y - 1, x - 1] = np.max(region, axis=(0, 1))
        return new_image

    def _color_match(self, color1, color2, tolerance):
        return np.all(np.abs(color1 - color2) <= tolerance)

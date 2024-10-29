from PIL import Image
import numpy as np


class Binaryzation:
    @staticmethod
    def apply_otsu(image):
        if image is None:
            raise ValueError("Brak obrazu do binaryzacji.")

        img_gray = np.array(image.convert("L"))  # Konwertujemy obraz na odcienie szarości
        pixel_counts, bin_edges = np.histogram(img_gray, bins=256, range=(0, 256))

        total_pixels = img_gray.size
        current_max, threshold = 0, 0
        sum_total, sumB, weightB = 0, 0, 0

        # Oblicz całkowitą sumę pikseli
        for i in range(256):
            sum_total += i * pixel_counts[i]

        for i in range(256):
            weightB += pixel_counts[i]
            if weightB == 0:
                continue
            weightF = total_pixels - weightB
            if weightF == 0:
                break

            sumB += i * pixel_counts[i]
            meanB = sumB / weightB
            meanF = (sum_total - sumB) / weightF

            # Obliczamy wariancję międzyklasową
            var_between = weightB * weightF * (meanB - meanF) ** 2

            if var_between > current_max:
                current_max = var_between
                threshold = i

        # Zastosuj binaryzację
        binary_img = np.where(img_gray > threshold, 255, 0)
        return Image.fromarray(binary_img.astype(np.uint8))

    @staticmethod
    def apply_threshold_binaryzation(image, threshold, channel):
        if image is None:
            raise ValueError("Brak obrazu do binaryzacji.")

        img_array = np.array(image)
        if channel == "RGB":
            avg_channel = img_array.mean(axis=2)
            binary_img = np.where(avg_channel > threshold, 255, 0)
        else:
            channel_idx = {"R": 0, "G": 1, "B": 2}[channel]
            binary_img = np.where(img_array[:, :, channel_idx] > threshold, 255, 0)

        return Image.fromarray(binary_img.astype(np.uint8))

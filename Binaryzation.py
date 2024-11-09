from PIL import Image
import numpy as np
from scipy.ndimage import uniform_filter, minimum_filter, maximum_filter

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

        if len(img_array.shape) == 2:
            # Obraz w odcieniach szarości, binaryzacja na podstawie jednego progu
            binary_img = np.where(img_array > threshold, 255, 0)

        elif len(img_array.shape) == 3:
            if channel == "RGB":
                avg_channel = img_array.mean(axis=2)
                binary_img = np.where(avg_channel > threshold, 255, 0)
            else:
                channel_idx = {"R": 0, "G": 1, "B": 2}[channel]
                binary_img = np.where(img_array[:, :, channel_idx] > threshold, 255, 0)

        return Image.fromarray(binary_img.astype(np.uint8))

    @staticmethod
    def apply_niblack_binaryzation(image, window_size=15, k=-0.2):
        if image is None:
            raise ValueError("Brak obrazu do binaryzacji.")

        img_gray = np.array(image.convert("L"))  # Konwertujemy obraz na odcienie szarości

        # Obliczanie lokalnej średniej i odchylenia standardowego za pomocą filtrów
        mean = uniform_filter(img_gray, size=window_size)
        squared_mean = uniform_filter(img_gray ** 2, size=window_size)
        stddev = np.sqrt(squared_mean - mean ** 2)

        # Obliczanie binaryzacji Niblacka
        threshold = mean + k * stddev
        binary_image = (img_gray > threshold).astype(np.uint8) * 255

        return Image.fromarray(binary_image)

    @staticmethod
    def apply_sauvola_binaryzation(image, window_size=15, k=0.15):
        if image is None:
            raise ValueError("Brak obrazu do binaryzacji.")

        img_gray = np.array(image.convert("L"))

        # Obliczanie lokalnej średniej i odchylenia standardowego za pomocą filtrów
        mean = uniform_filter(img_gray, size=window_size)
        squared_mean = uniform_filter(img_gray ** 2, size=window_size)
        stddev = np.sqrt(squared_mean - mean ** 2)

        # Obliczanie binaryzacji Sauvoli
        threshold = mean * (1 + k * (stddev / 128 - 1))
        binary_image = (img_gray > threshold).astype(np.uint8) * 255

        return Image.fromarray(binary_image)

    @staticmethod
    def apply_bernsen_binaryzation(image, window_size=15, T1 = 0.15):
        if image is None:
            raise ValueError("Brak obrazu do binaryzacji.")

        img_gray = np.array(image.convert("L"))

        # Obliczanie minimalnej i maksymalnej wartości w lokalnym oknie
        min_img = minimum_filter(img_gray, size=window_size)
        max_img = maximum_filter(img_gray, size=window_size)

        # Obliczanie progu
        threshold = (min_img + max_img) / 2
        contrast = max_img - min_img

        # Binaryzacja zgodnie z metodą Bernsena
        binary_image = np.where(contrast < T1, 0, (img_gray > threshold).astype(np.uint8) * 255)

        return Image.fromarray(binary_image)

    @staticmethod
    def apply_custom_binaryzation(image):
        if image is None:
            raise ValueError("Brak obrazu do binaryzacji.")

        img_gray = np.array(image.convert("L"))

        # Własna metoda binaryzacji: średnia na lokalnym oknie 3x3
        mean = uniform_filter(img_gray, size=3)

        # Binaryzacja z użyciem progu na podstawie średniej
        binary_image = (img_gray > mean).astype(np.uint8) * 255

        return Image.fromarray(binary_image)




import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import exposure


class Histogram:
    """Klasa odpowiedzialna za wyświetlanie histogramów obrazu."""

    def show_histogram(self, image, channel):
        if image is None:
            raise ValueError("Brak obrazu do wyświetlenia histogramu.")

        img_array = np.array(image)

        plt.figure()
        if channel == "RGB":
            avg_channel = img_array.mean(axis=2)
            plt.hist(avg_channel.ravel(), bins=256, color='gray', alpha=0.5)
        else:
            channel_idx = {"R": 0, "G": 1, "B": 2}[channel]
            plt.hist(img_array[:, :, channel_idx].ravel(), bins=256, color=channel.lower(), alpha=0.5)

        plt.title(f"Histogram - Kanał: {channel}")
        plt.show()

    def stretch_histogram(self, image):
        if image is None:
            raise ValueError("Brak obrazu do rozciągnięcia histogramu.")

        img_array = np.array(image)
        p_min, p_max = np.min(img_array), np.max(img_array)
        stretched = (img_array - p_min) / (p_max - p_min) * 255
        return Image.fromarray(stretched.astype(np.uint8))

    def equalize_histogram(self, image):
        if image is None:
            raise ValueError("Brak obrazu do wyrównania histogramu.")

        img_gray = np.array(image.convert("L"))  # Konwertujemy na odcienie szarości
        img_equalized = exposure.equalize_hist(img_gray) * 255
        return Image.fromarray(img_equalized.astype(np.uint8))

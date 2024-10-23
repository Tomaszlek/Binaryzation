import matplotlib.pyplot as plt
import numpy as np


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
        
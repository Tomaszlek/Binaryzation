from PIL import Image
import numpy as np


class Binaryzation:
    """Klasa odpowiedzialna za binaryzację obrazu."""

    def apply_threshold(self, image, threshold, channel):
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
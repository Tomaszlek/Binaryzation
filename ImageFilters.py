from PIL import Image
import numpy as np


class ImageFilters:
    @staticmethod
    def apply_convolution(image, kernel):
        if image is None:
            raise ValueError("Brak obrazu do przetworzenia.")

        img_array = np.array(image)
        kernel_height, kernel_width = kernel.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        # Padding obrazu
        padded_image = np.pad(img_array, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='edge')
        result_image = np.zeros_like(img_array)

        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                # Wycinanie lokalnego regionu
                local_region = padded_image[i:i + kernel_height, j:j + kernel_width]
                result_image[i, j] = np.clip(np.sum(local_region * kernel[:, :, np.newaxis], axis=(0, 1)), 0, 255)

        return Image.fromarray(result_image.astype(np.uint8))

    @staticmethod
    def gaussian_blur(size=5, sigma=1.0):
        """Tworzy kernel rozmycia Gaussowskiego."""
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
            (size, size)
        )
        return kernel / np.sum(kernel)

    @staticmethod
    def prewitt_filter():
        """Kernel filtru Prewitta."""
        return np.array([[1, 1, 1],
                         [0, 0, 0],
                         [-1, -1, -1]])

    @staticmethod
    def sobel_filter():
        """Kernel filtru Sobela."""
        return np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])

    @staticmethod
    def laplacian_filter():
        """Kernel Laplacjana."""
        return np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]])

    @staticmethod
    def apply_median_filter(image, kernel_size=3):
        if image is None:
            raise ValueError("Brak obrazu do przetworzenia.")

        img_array = np.array(image)
        pad_size = kernel_size // 2
        padded_image = np.pad(img_array, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')
        result_image = np.zeros_like(img_array)

        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                # Wycinanie lokalnego regionu
                local_region = padded_image[i:i + kernel_size, j:j + kernel_size]
                result_image[i, j] = np.median(local_region, axis=(0, 1))

        return Image.fromarray(result_image.astype(np.uint8))

    @staticmethod
    def pixelation(image, pixel_size=10):
        if image is None:
            raise ValueError("Brak obrazu do przetworzenia.")

        img_array = np.array(image)
        h, w, c = img_array.shape
        result_image = img_array.copy()

        for i in range(0, h, pixel_size):
            for j in range(0, w, pixel_size):
                # Wycinanie bloku
                block = img_array[i:i + pixel_size, j:j + pixel_size]
                # Obliczanie średniego koloru
                average_color = block.mean(axis=(0, 1)).astype(np.uint8)
                result_image[i:i + pixel_size, j:j + pixel_size] = average_color

        return Image.fromarray(result_image)

    @staticmethod
    def apply_kuwahara_filter(image, window_size=5):
        if image is None:
            raise ValueError("Brak obrazu do przetworzenia.")

        img_array = np.array(image)
        pad_size = window_size // 2
        padded_image = np.pad(img_array, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')
        result_image = np.zeros_like(img_array)

        for i in range(img_array.shape[0]):
            for j in range(img_array.shape[1]):
                # Wycinanie lokalnego regionu
                local_region = padded_image[i:i + window_size, j:j + window_size]
                # Obliczanie lokalnych statystyk
                mean_1 = local_region[:pad_size + 1, :pad_size + 1].mean(axis=(0, 1))
                mean_2 = local_region[:pad_size + 1, pad_size:].mean(axis=(0, 1))
                mean_3 = local_region[pad_size:, :pad_size + 1].mean(axis=(0, 1))
                mean_4 = local_region[pad_size:, pad_size:].mean(axis=(0, 1))

                # Wybieranie średniego koloru z najmniejszym odchyleniem
                means = np.array([mean_1, mean_2, mean_3, mean_4])
                result_image[i, j] = means[np.argmin(np.var(means, axis=1))]

        return Image.fromarray(result_image.astype(np.uint8))


class PredatorFilter:
    @staticmethod
    def apply_predator_filter(image, pixel_size=10):
        if image is None:
            raise ValueError("Brak obrazu do przetworzenia.")

        # Pikselizacja
        pixelated_image = ImageFilters.pixelation(image, pixel_size)

        # MinRGB
        img_array = np.array(pixelated_image)
        min_rgb_image = np.min(img_array, axis=2)

        # Filtr Sobela
        sobel_kernel = ImageFilters.sobel_filter()
        sobel_result = ImageFilters.apply_convolution(Image.fromarray(min_rgb_image.astype(np.uint8)), sobel_kernel)

        return sobel_result
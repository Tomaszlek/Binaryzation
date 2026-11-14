from tkinter import filedialog, messagebox
from PIL import Image
import numpy as np


class ImageHandler:
    def __init__(self):
        self.image = None
        self.image_path = None

    def load_image(self, file_path):
        pil_image = Image.open(file_path)

        # zmienne na ścieżkę i obraz
        self.image_path = file_path
        self.image = pil_image

        # potrzebna konwersja do numpy zeby robic operacje na obrazie
        return np.array(pil_image)

    def save_image(self, image):
        try:
            if image is None:
                raise ValueError("Brak obrazu do zapisania.")
            file_path = filedialog.asksaveasfilename(defaultextension=".png")
            if not file_path:
                raise FileNotFoundError("Nie wybrano ścieżki zapisu.")
            image.save(file_path)
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Błąd", str(e))
            raise
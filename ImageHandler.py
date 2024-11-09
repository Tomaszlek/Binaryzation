from tkinter import filedialog, messagebox
from PIL import Image, UnidentifiedImageError


class ImageHandler:
    """Klasa do zarządzania wczytywaniem i zapisywaniem obrazów."""

    def __init__(self):
        self.image = None
        self.image_path = None

    def load_image(self):
        try:
            file_path = filedialog.askopenfilename()
            if not file_path:
                raise FileNotFoundError("Nie wybrano pliku.")
            self.image = Image.open(file_path)
            self.image_path = file_path
            return self.image
        except FileNotFoundError as e:
            messagebox.showerror("Błąd", str(e))
            raise
        except UnidentifiedImageError:
            messagebox.showerror("Błąd", "Nie można otworzyć tego pliku jako obrazu.")
            raise

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
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt


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


class ImageApp:
    """Główna klasa aplikacji zarządzająca interfejsem użytkownika."""

    def __init__(self, root):
        self.root = root
        self.root.title("Binaryzacja obrazu oraz histogram")

        self.image_handler = ImageHandler()
        self.binaryzation = Binaryzation()
        self.histogram = Histogram()

        # Menu
        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)
        file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Plik", menu=file_menu)
        file_menu.add_command(label="Wczytaj obraz", command=self.load_image)
        file_menu.add_command(label="Zapisz obraz", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Wyjdź", command=self.root.quit)

        # Main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Image display panel
        self.image_label = tk.Label(self.main_frame)
        self.image_label.pack()

        # Control panel
        self.control_panel = tk.Frame(self.root)
        self.control_panel.pack(side=tk.RIGHT, padx=10, pady=10)

        # Thresholding section
        tk.Label(self.control_panel, text="Binaryzacja progowa").pack(anchor=tk.W)
        self.threshold_slider = tk.Scale(self.control_panel, from_=0, to=255, orient=tk.HORIZONTAL, label="Próg")
        self.threshold_slider.pack(anchor=tk.W)

        self.channel_var = tk.StringVar(value="RGB")
        tk.Radiobutton(self.control_panel, text="Czerwony (R)", variable=self.channel_var, value="R").pack(anchor=tk.W)
        tk.Radiobutton(self.control_panel, text="Zielony (G)", variable=self.channel_var, value="G").pack(anchor=tk.W)
        tk.Radiobutton(self.control_panel, text="Niebieski (B)", variable=self.channel_var, value="B").pack(anchor=tk.W)
        tk.Radiobutton(self.control_panel, text="Średnia (RGB)", variable=self.channel_var, value="RGB").pack(
            anchor=tk.W)

        self.apply_button = tk.Button(self.control_panel, text="Zastosuj binaryzację", command=self.apply_threshold)
        self.apply_button.pack(anchor=tk.W, pady=5)

        # Histogram section
        tk.Label(self.control_panel, text="Wyświetlanie histogramu").pack(anchor=tk.W)
        self.histogram_button = tk.Button(self.control_panel, text="Pokaż histogram", command=self.show_histogram)
        self.histogram_button.pack(anchor=tk.W, pady=5)

        self.hist_channel_var = tk.StringVar(value="RGB")
        tk.Radiobutton(self.control_panel, text="Czerwony (R)", variable=self.hist_channel_var, value="R").pack(
            anchor=tk.W)
        tk.Radiobutton(self.control_panel, text="Zielony (G)", variable=self.hist_channel_var, value="G").pack(
            anchor=tk.W)
        tk.Radiobutton(self.control_panel, text="Niebieski (B)", variable=self.hist_channel_var, value="B").pack(
            anchor=tk.W)
        tk.Radiobutton(self.control_panel, text="Średnia (RGB)", variable=self.hist_channel_var, value="RGB").pack(
            anchor=tk.W)

        # Status bar
        self.status_bar = tk.Label(self.root, text="Brak załadowanego obrazu", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.image = None

    def load_image(self):
        try:
            self.image = self.image_handler.load_image()
            self.display_image(self.image)
            self.status_bar.config(text=f"Wczytano obraz: {self.image_handler.image_path}")
        except FileNotFoundError:
            self.status_bar.config(text="Nie udało się wczytać obrazu.")

    def save_image(self):
        try:
            self.image_handler.save_image(self.image)
            self.status_bar.config(text="Zapisano obraz.")
        except FileNotFoundError:
            self.status_bar.config(text="Nie udało się zapisać obrazu.")

    def display_image(self, img):
        img_resized = img.resize((512, 512))  # Resize to fit in the UI
        img_tk = ImageTk.PhotoImage(img_resized)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    def apply_threshold(self):
        try:
            threshold = self.threshold_slider.get()
            channel = self.channel_var.get()
            binary_img = self.binaryzation.apply_threshold(self.image, threshold, channel)
            self.display_image(binary_img)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def show_histogram(self):
        try:
            channel = self.hist_channel_var.get()
            self.histogram.show_histogram(self.image, channel)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()

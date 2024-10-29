import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk
from Image import ImageHandler
from Binaryzation import Binaryzation
from Plots import Histogram


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

        # Histogram section - rozszerzone
        tk.Label(self.control_panel, text="Histogram - operacje").pack(anchor=tk.W)

        # Wyświetlanie histogramu
        self.histogram_button = tk.Button(self.control_panel, text="Pokaż histogram", command=self.show_histogram)
        self.histogram_button.pack(anchor=tk.W, pady=5)

        # Rozciąganie histogramu
        self.stretch_hist_button = tk.Button(self.control_panel, text="Rozciągnij histogram",
                                             command=self.stretch_histogram)
        self.stretch_hist_button.pack(anchor=tk.W, pady=5)

        # Wyrównanie histogramu
        self.equalize_hist_button = tk.Button(self.control_panel, text="Wyrównaj histogram",
                                              command=self.equalize_histogram)
        self.equalize_hist_button.pack(anchor=tk.W, pady=5)

        # Binaryzacja Otsu
        self.otsu_button = tk.Button(self.control_panel, text="Binaryzacja Otsu", command=self.apply_otsu)
        self.otsu_button.pack(anchor=tk.W, pady=5)
        # Status bar
        self.status_bar = tk.Label(self.root, text="Brak załadowanego obrazu", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.image = None

        # Rozciąganie histogramu
    def stretch_histogram(self):
        try:
            self.image = self.histogram.stretch_histogram(self.image)
            self.display_image(self.image)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

        # Wyrównanie histogramu

    def equalize_histogram(self):
        try:
            self.image = self.histogram.equalize_histogram(self.image)
            self.display_image(self.image)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

        # Binaryzacja Otsu

    def apply_otsu(self):
        try:
            binary_img = self.binaryzation.apply_otsu(self.image)
            self.display_image(binary_img)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

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
            binary_img = self.binaryzation.apply_threshold_binaryzation(self.image, threshold, channel)
            self.display_image(binary_img)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def show_histogram(self):
        try:
            channel = self.hist_channel_var.get()
            self.histogram.show_histogram(self.image, channel)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

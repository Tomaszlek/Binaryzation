import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk
from ImageHandler import ImageHandler
from Binaryzation import Binaryzation
from Plots import Histogram
from ImageFilters import ImageFilters, PredatorFilter
from ImageEditor import ImageEditor


class ImageApp:
    """Główna klasa aplikacji zarządzająca interfejsem użytkownika."""

    def __init__(self, root):
        self.root = root
        self.root.title("Aplikacja biometryczna")

        self.image_handler = ImageHandler()
        self.binaryzation = Binaryzation()
        self.histogram = Histogram()
        self.image_filters = ImageFilters()
        self.image_edits = ImageEditor()

        # Menu
        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)

        # Plik menu
        file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Plik", menu=file_menu)
        file_menu.add_command(label="Wczytaj obraz", command=self.load_image)
        file_menu.add_command(label="Zapisz obraz", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Wyjdź", command=self.root.quit)

        # Binaryzacja menu
        binaryzation_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Binaryzacja", menu=binaryzation_menu)
        binaryzation_menu.add_command(label="Binaryzacja progowa", command=self.apply_threshold)
        binaryzation_menu.add_separator()
        binaryzation_menu.add_command(label="Binaryzacja Otsu", command=self.apply_otsu)
        binaryzation_menu.add_command(label="Binaryzacja Niblacka", command=self.apply_niblack)
        binaryzation_menu.add_command(label="Binaryzacja Sauvola", command=self.apply_sauvola)
        binaryzation_menu.add_command(label="Binaryzacja Bernsena", command=self.apply_bernsen)
        binaryzation_menu.add_command(label="Binaryzacja własna(OSTROŻNIE!!!)", command=self.apply_my_binaryzation)

        # Histogram menu
        histogram_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Histogram", menu=histogram_menu)
        histogram_menu.add_command(label="Pokaż histogram", command=self.show_histogram)
        histogram_menu.add_separator()
        histogram_menu.add_command(label="Rozciągnij histogram", command=self.stretch_histogram)
        histogram_menu.add_command(label="Wyrównaj histogram", command=self.equalize_histogram)

        # Filtry menu
        filters_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Filtry", menu=filters_menu)
        filters_menu.add_command(label="Filtr Prewitta", command=self.apply_prewitt_filter)
        filters_menu.add_command(label="Filtr Sobela", command=self.apply_sobel_filter)
        filters_menu.add_command(label="Filtr Laplacjana", command=self.apply_laplacian_filter)
        filters_menu.add_command(label="Rozmycie Gaussowskie", command=self.apply_gaussian_blur)
        filters_menu.add_command(label="Filtr Kuwahara", command=self.apply_kuwahara_filter)
        filters_menu.add_command(label="Pikselizacja", command=self.apply_pixelation)
        filters_menu.add_command(label="Filtr Predatora", command=self.apply_predator_filter)

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
        tk.Radiobutton(self.control_panel, text="Średnia (RGB)", variable=self.channel_var, value="RGB").pack(anchor=tk.W)

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

    def stretch_histogram(self):
        try:
            self.image = self.histogram.stretch_histogram(self.image)
            self.display_image(self.image)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def equalize_histogram(self):
        try:
            self.image = self.histogram.equalize_histogram(self.image)
            self.display_image(self.image)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def apply_otsu(self):
        try:
            binary_img = self.binaryzation.apply_otsu(self.image)
            self.display_image(binary_img)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def apply_niblack(self):
        try:
            binary_img = self.binaryzation.apply_niblack_binaryzation(self.image)
            self.display_image(binary_img)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def apply_sauvola(self):
        try:
            binary_img = self.binaryzation.apply_sauvola_binaryzation(self.image)
            self.display_image(binary_img)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def apply_bernsen(self):
        try:
            binary_img = self.binaryzation.apply_bernsen_binaryzation(self.image)
            self.display_image(binary_img)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def apply_my_binaryzation(self):
        try:
            binary_img = self.binaryzation.apply_custom_binaryzation(self.image)
            self.display_image(binary_img)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def apply_prewitt_filter(self):
        try:
            kernel = self.image_filters.prewitt_filter()
            filtered_image = self.image_filters.apply_convolution(self.image, kernel)
            self.display_image(filtered_image)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def apply_sobel_filter(self):
        try:
            kernel = self.image_filters.sobel_filter()
            filtered_image = self.image_filters.apply_convolution(self.image, kernel)
            self.display_image(filtered_image)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def apply_laplacian_filter(self):
        try:
            kernel = self.image_filters.laplacian_filter()
            filtered_image = self.image_filters.apply_convolution(self.image, kernel)
            self.display_image(filtered_image)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def apply_gaussian_blur(self):
        try:
            kernel = self.image_filters.gaussian_blur()
            blurred_image = self.image_filters.apply_convolution(self.image, kernel)
            self.display_image(blurred_image)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def apply_kuwahara_filter(self):
        try:
            filtered_image = self.image_filters.apply_kuwahara_filter(self.image)
            self.display_image(filtered_image)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def apply_pixelation(self):
        try:
            pixelated_image = self.image_filters.pixelation(self.image)
            self.display_image(pixelated_image)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def apply_predator_filter(self):
        try:
            predator_image = PredatorFilter.apply_predator_filter(self.image)
            self.display_image(predator_image)
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

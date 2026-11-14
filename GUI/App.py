import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import ImageTk, Image
import numpy as np
from ImageAlgorithms.Binaryzation import Binaryzation
from ImageAlgorithms.ImageFilters import ImageFilters, PredatorFilter
from ImageAlgorithms.Minutiae import annotate_minutiae, detect_minutiae
from ImageAlgorithms.Segmentation import color_selection_mask, resize_selection, watershed_segmentation
from Skeletization.Skeletonization import calibrated_thinning, zhang_suen

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikacja biometryczna")
        self.root.geometry("1100x720")

        self.image = None
        self.img_array = None
        self.selection_mask = None
        self.selection_source = None
        self.category_windows = {}
        self.max_pixels_var = tk.StringVar(master=self.root, value="1000")
        self.tolerance_min_var = tk.StringVar(master=self.root, value="0")
        self.tolerance_max_var = tk.StringVar(master=self.root, value="30")
        self.global_mode_var = tk.BooleanVar(master=self.root, value=False)
        self.tool_var = tk.StringVar(master=self.root, value="pencil")
        self.binary_method_var = tk.StringVar(master=self.root, value="Progowa")
        self.threshold_value = tk.IntVar(master=self.root, value=127)
        self.window_size_value = tk.IntVar(master=self.root, value=15)
        self.filter_method_var = tk.StringVar(master=self.root, value="Rozmycie Gaussa")
        self.filter_size_value = tk.IntVar(master=self.root, value=5)
        self.filter_sigma_value = tk.DoubleVar(master=self.root, value=1.0)

        workspace = tk.Frame(self.root)
        workspace.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(workspace, width=800, height=600, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<B1-Motion>", self.pencil_draw)
        self.canvas.bind("<Button-1>", self.magic_wand_or_fill)

        sidebar = tk.Frame(workspace, width=190, padx=12, pady=12, relief=tk.GROOVE, bd=1)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)
        tk.Label(sidebar, text="NARZĘDZIA", font=("TkDefaultFont", 12, "bold")).pack(
            anchor=tk.W, pady=(0, 12)
        )
        tk.Button(sidebar, text="Wczytaj obraz", command=self.load_image).pack(fill=tk.X, pady=3)
        tk.Button(sidebar, text="Zapisz obraz", command=self.save_image).pack(fill=tk.X, pady=3)
        tk.Label(sidebar, text="Przetwarzanie", anchor=tk.W).pack(fill=tk.X, pady=(16, 4))
        for title, category in (
            ("Filtry", "filters"),
            ("Binaryzacja", "binaryzation"),
            ("Segmentacja", "segmentation"),
            ("Szkieletyzacja", "skeletonization"),
            ("Minucje", "minutiae"),
        ):
            tk.Button(
                sidebar,
                text=title,
                command=lambda selected=category: self.open_category(selected),
                height=2,
            ).pack(fill=tk.X, pady=4)

        self.tk_image = None
        self.status_bar = tk.Label(self.root, text="Brak załadowanego obrazu", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def open_category(self, category):
        existing = self.category_windows.get(category)
        if existing is not None and existing.winfo_exists():
            existing.deiconify()
            existing.lift()
            return

        titles = {
            "filters": "Filtry obrazu",
            "binaryzation": "Binaryzacja",
            "segmentation": "Segmentacja i zaznaczanie",
            "skeletonization": "Szkieletyzacja",
            "minutiae": "Ekstrakcja minucji",
        }
        window = tk.Toplevel(self.root)
        window.title(titles[category])
        window.transient(self.root)
        window.resizable(False, False)
        self.category_windows[category] = window
        window.protocol(
            "WM_DELETE_WINDOW",
            lambda selected=category, target=window: self.close_category(selected, target),
        )
        panel = tk.Frame(window, padx=16, pady=14)
        panel.pack(fill=tk.BOTH, expand=True)
        tk.Label(panel, text=titles[category], font=("TkDefaultFont", 12, "bold")).pack(
            anchor=tk.W, pady=(0, 10)
        )

        if category == "filters":
            self._build_filter_menu(panel)
        elif category == "binaryzation":
            self._build_binaryzation_menu(panel)
        elif category == "segmentation":
            self._build_segmentation_menu(panel)
        elif category == "skeletonization":
            self._build_skeleton_menu(panel)
        elif category == "minutiae":
            self._build_minutiae_menu(panel)

    def close_category(self, category, window):
        if category == "segmentation":
            self.tool_var.set("pencil")
        self.category_windows.pop(category, None)
        window.destroy()

    @staticmethod
    def _menu_button(parent, label, command):
        tk.Button(parent, text=label, command=command).pack(fill=tk.X, pady=3)

    def _build_filter_menu(self, panel):
        tk.Label(panel, text="Wybierz filtr").pack(anchor=tk.W)
        tk.OptionMenu(
            panel,
            self.filter_method_var,
            "Rozmycie Gaussa",
            "Filtr medianowy",
            "Pikselizacja",
            "Filtr Predatora",
        ).pack(fill=tk.X, pady=(2, 8))
        tk.Scale(
            panel, from_=3, to=15, resolution=2, orient=tk.HORIZONTAL,
            label="Rozmiar filtra", variable=self.filter_size_value
        ).pack(fill=tk.X)
        tk.Scale(
            panel, from_=0.2, to=5.0, resolution=0.1, orient=tk.HORIZONTAL,
            label="Sigma Gaussa", variable=self.filter_sigma_value
        ).pack(fill=tk.X)
        self._menu_button(panel, "Zastosuj filtr", self.apply_selected_filter)

    def _build_binaryzation_menu(self, panel):
        tk.Label(panel, text="Metoda").pack(anchor=tk.W)
        tk.OptionMenu(
            panel,
            self.binary_method_var,
            "Progowa",
            "Otsu",
            "Niblack",
            "Sauvola",
            "Phansalkar",
            "Własna",
        ).pack(fill=tk.X, pady=(2, 8))
        tk.Scale(
            panel, from_=0, to=255, orient=tk.HORIZONTAL, label="Próg",
            variable=self.threshold_value
        ).pack(fill=tk.X)
        tk.Scale(
            panel, from_=3, to=51, resolution=2, orient=tk.HORIZONTAL,
            label="Rozmiar okna", variable=self.window_size_value
        ).pack(fill=tk.X)
        self._menu_button(panel, "Zastosuj binaryzację", self.apply_binaryzation)

    def _build_segmentation_menu(self, panel):
        tk.Label(panel, text="Narzędzie").pack(anchor=tk.W)
        for label, value in (
            ("Ołówek", "pencil"),
            ("Wypełnienie", "fill"),
            ("Magic Wand", "magic_wand"),
        ):
            tk.Radiobutton(
                panel, text=label, variable=self.tool_var, value=value
            ).pack(anchor=tk.W)
        self.max_pixels_entry = tk.Entry(panel, textvariable=self.max_pixels_var)
        self._labeled_entry(panel, "Maksymalna liczba pikseli", self.max_pixels_entry)
        self.tolerance_min_entry = tk.Entry(panel, textvariable=self.tolerance_min_var)
        self._labeled_entry(panel, "Minimalna tolerancja", self.tolerance_min_entry)
        self.tolerance_max_entry = tk.Entry(panel, textvariable=self.tolerance_max_var)
        self._labeled_entry(panel, "Maksymalna tolerancja", self.tolerance_max_entry)
        tk.Checkbutton(
            panel, text="Tryb globalny (Flood)", variable=self.global_mode_var
        ).pack(anchor=tk.W, pady=4)
        tk.Label(panel, text="Zaznaczenie").pack(anchor=tk.W, pady=(8, 2))
        self._menu_button(
            panel, "Rozszerz (dylatacja)", lambda: self.resize_selected("dilate")
        )
        self._menu_button(
            panel, "Zmniejsz (erozja)", lambda: self.resize_selected("erode")
        )
        tk.Label(panel, text="Watershed").pack(anchor=tk.W, pady=(8, 2))
        self._menu_button(panel, "Segmentuj obraz", self.apply_watershed)

    @staticmethod
    def _labeled_entry(parent, label, entry):
        tk.Label(parent, text=label).pack(anchor=tk.W, pady=(5, 0))
        entry.pack(fill=tk.X)

    def _build_skeleton_menu(self, panel):
        self._menu_button(
            panel, "Zhang-Suen",
            lambda: self.apply_skeletonization(zhang_suen)
        )
        self._menu_button(
            panel, "CalThin",
            lambda: self.apply_skeletonization(calibrated_thinning)
        )

    def _build_minutiae_menu(self, panel):
        self._menu_button(panel, "Wykryj minucje na obrazie", self.show_minutiae)
        self._menu_button(panel, "Porównaj z drugim odciskiem", self.compare_minutiae)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Wybierz obraz",
            filetypes=(("Pliki obrazów", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"), ("Wszystkie pliki", "*.*"))
        )
        if not file_path:
            self.status_bar.config(text="Nie wybrano pliku.")
            return

        # Wczytaj obraz
        pil_image = Image.open(file_path).convert("RGB")
        self.img_array = np.array(pil_image, dtype=np.uint8)
        self.image = pil_image
        self.selection_mask = None
        self.selection_source = None
        self.update_canvas()
        self.status_bar.config(text=f"Wczytano obraz: {file_path}")

    def save_image(self):
        if self.image is None:
            self.status_bar.config(text="Brak obrazu do zapisania.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Zapisz obraz",
            defaultextension=".png",
            filetypes=(("Pliki PNG", "*.png"), ("Pliki JPEG", "*.jpg;*.jpeg"), ("Wszystkie pliki", "*.*"))
        )
        if not file_path:
            return

        self.image.save(file_path)
        self.status_bar.config(text=f"Zapisano obraz: {file_path}")

    def update_canvas(self):
        if self.img_array is None:
            return

        pil_image = Image.fromarray(self.img_array)
        self.image = pil_image
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def pencil_draw(self, event):
        if self.tool_var.get() != "pencil" or self.img_array is None:
            return

        x, y = event.x, event.y
        size = 5
        color = [0, 0, 0]  # Czarny
        self.apply_tool("pencil", x, y, size=size, color=color)
        self.update_canvas()

    def magic_wand_or_fill(self, event):
        if self.img_array is None:
            return

        x, y = event.x, event.y
        height, width = self.img_array.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return
        tool = self.tool_var.get()
        if tool not in ("magic_wand", "fill"):
            return
        try:
            max_pixels = int(self.max_pixels_entry.get())
            tolerance_min = int(self.tolerance_min_entry.get())
            tolerance_max = int(self.tolerance_max_entry.get())
            if max_pixels <= 0 or not 0 <= tolerance_min <= tolerance_max <= 255:
                raise ValueError("Ustaw limit > 0 i tolerancję w zakresie 0-255 (min <= max).")
        except ValueError as error:
            messagebox.showerror("Nieprawidłowe parametry", str(error))
            return
        global_mode = self.global_mode_var.get()
        if tool == "magic_wand":
            self.magic_wand(
                x, y, max_pixels, tolerance_min, tolerance_max, global_mode
            )
        else:
            self.fill(
                x, y, [255, 0, 0], global_mode,
                max_pixels, tolerance_min, tolerance_max
            )
        self.update_canvas()

    def apply_tool(self, tool, x, y, **kwargs):
        if tool == "pencil":
            self.pencil_tool(x, y, kwargs.get("color", [0, 0, 0]), kwargs.get("size", 1))
        elif tool == "magic_wand":
            self.magic_wand(x, y, kwargs.get("max_pixels", 1000), kwargs.get("tolerance_min", 0), kwargs.get("tolerance_max", 30), kwargs.get("global_mode", False))
        elif tool == "fill":
            self.fill(x, y, kwargs.get("color", [255, 0, 0]), kwargs.get("global_mode", False))

    def pencil_tool(self, x, y, color, size):
        for dx in range(-size, size + 1):
            for dy in range(-size, size + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.img_array.shape[1] and 0 <= ny < self.img_array.shape[0]:
                    self.img_array[ny, nx] = color

    def fill(
        self, x, y, color, global_mode=False, max_pixels=1000,
        tolerance_min=0, tolerance_max=30
    ):
        mask = color_selection_mask(
            self.img_array, x, y, max_pixels,
            tolerance_min, tolerance_max, global_mode
        )
        self.img_array[mask] = color
        self.selection_mask = None
        self.selection_source = None

    def magic_wand(self, x, y, max_pixels, tolerance_min, tolerance_max, global_mode=False):
        mask = color_selection_mask(
            self.img_array, x, y, max_pixels,
            tolerance_min, tolerance_max, global_mode
        )
        self.selection_source = self.img_array.copy()
        self.selection_mask = mask
        self.img_array[mask] = [0, 255, 0]

    def _get_flood_mask(self, x, y, tolerance_min=0, tolerance_max=30):
        return color_selection_mask(
            self.img_array, x, y, self.img_array.shape[0] * self.img_array.shape[1],
            tolerance_min, tolerance_max
        )

    def _get_flood_mask_global(self, target_color, tolerance_min=0, tolerance_max=30):
        difference = np.abs(
            self.img_array.astype(np.int16) - np.asarray(target_color, dtype=np.int16)
        )
        return np.all(
            (difference >= tolerance_min) & (difference <= tolerance_max), axis=2
        )
   # tu jest modul binaryzecji, moj metoda też jest
    def apply_binaryzation(self):

        if self.img_array is None:
            messagebox.showerror("Brak obrazu", "Najpierw wczytaj obraz.")
            return
        
        source = Image.fromarray(self.img_array)
        window_size = self.window_size_value.get()
        method = self.binary_method_var.get()

        methods = {
            "Progowa": lambda: Binaryzation.apply_threshold_binaryzation(
                source, self.threshold_value.get(), "RGB"
            ),
            "Otsu": lambda: Binaryzation.apply_otsu(source),
            "Niblack": lambda: Binaryzation.apply_niblack_binaryzation(
                source, window_size=window_size
            ),
            "Sauvola": lambda: Binaryzation.apply_sauvola_binaryzation(
                source, window_size=window_size
            ),
            "Phansalkar": lambda: Binaryzation.apply_phansalkar_binaryzation(
                source, window_size=window_size
            ),
            "Własna": lambda: Binaryzation.apply_custom_binaryzation(
                source, window_size=window_size
            ),
        }

        result = methods[method]()
        self.img_array = np.asarray(result.convert("RGB"))
        self.selection_mask = None
        self.selection_source = None
        self.update_canvas()

    def _apply_image_filter(self, filter_function):

        if self.img_array is None:
            messagebox.showerror("Brak obrazu", "Najpierw wczytaj obraz.")
            return
        
        result = filter_function(Image.fromarray(self.img_array))
        self.img_array = np.asarray(result.convert("RGB"))

        self.selection_mask = None
        self.selection_source = None
        self.update_canvas()

    def apply_selected_filter(self):
        filters = {
            "Rozmycie Gaussa": self.apply_gaussian_blur,
            "Filtr medianowy": self.apply_median_filter,
            "Pikselizacja": self.apply_pixelation,
            "Filtr Predatora": self.apply_predator_filter,
        }
        filters[self.filter_method_var.get()]()

    def apply_gaussian_blur(self):
        size = self.filter_size_value.get()
        kernel = ImageFilters.gaussian_blur(
            size=size, sigma=self.filter_sigma_value.get()
        )
        self._apply_image_filter(
            lambda image: ImageFilters.apply_convolution(image, kernel)
        )

    def apply_median_filter(self):
        self._apply_image_filter(
            lambda image: ImageFilters.apply_median_filter(
                image, kernel_size=self.filter_size_value.get()
            )
        )

    def apply_pixelation(self):
        self._apply_image_filter(
            lambda image: ImageFilters.pixelation(
                image, pixel_size=self.filter_size_value.get()
            )
        )

    def apply_predator_filter(self):
        self._apply_image_filter(
            lambda image: PredatorFilter.apply_predator_filter(
                image, pixel_size=self.filter_size_value.get()
            )
        )

    def resize_selected(self, operation):
        if self.selection_mask is None or self.selection_source is None:
            messagebox.showinfo(
                "Brak zaznaczenia", "Najpierw wybierz obszar narzędziem Magic Wand."
            )
            return
        self.selection_mask = resize_selection(self.selection_mask, operation)
        self.img_array = self.selection_source.copy()
        self.img_array[self.selection_mask] = [0, 255, 0]
        self.update_canvas()

    def apply_skeletonization(self, algorithm):
        if self.img_array is None:
            messagebox.showerror("Brak obrazu", "Najpierw wczytaj obraz.")
            return
        gray = Image.fromarray(self.img_array).convert("L")
        skeleton = algorithm(gray)
        self.img_array = np.repeat(np.asarray(skeleton)[:, :, None], 3, axis=2)
        self.selection_mask = None
        self.selection_source = None
        self.update_canvas()

    def apply_watershed(self):
        if self.img_array is None:
            messagebox.showerror("Brak obrazu", "Najpierw wczytaj obraz.")
            return
        gray = Image.fromarray(self.img_array).convert("L")
        labels = watershed_segmentation(gray)
        boundaries = np.zeros(labels.shape, dtype=bool)
        boundaries[1:, :] |= labels[1:, :] != labels[:-1, :]
        boundaries[:, 1:] |= labels[:, 1:] != labels[:, :-1]
        result = np.repeat(np.asarray(gray)[:, :, None], 3, axis=2)
        result[boundaries] = [255, 0, 0]
        self.img_array = result
        self.update_canvas()

    def show_minutiae(self):
        if self.img_array is None:
            messagebox.showerror("Brak obrazu", "Najpierw wczytaj obraz.")
            return
        gray = Image.fromarray(self.img_array).convert("L")
        skeleton = Image.fromarray(zhang_suen(gray))
        minutiae = detect_minutiae(skeleton)
        self.img_array = annotate_minutiae(skeleton, minutiae)
        endings = sum(kind == "ending" for _, _, kind in minutiae)
        bifurcations = len(minutiae) - endings
        self.update_canvas()
        self.status_bar.config(
            text=(
                f"Minucje: {len(minutiae)} (zakończenia: {endings}, "
                f"bifurkacje: {bifurcations})"
            )
        )

    def compare_minutiae(self):

        if self.img_array is None:
            messagebox.showerror("Brak obrazu", "Najpierw wczytaj obraz.")
            return
        
        path = filedialog.askopenfilename(
            title="Wybierz drugi odcisk",
            filetypes=(("Pliki obrazów", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"),)
        )

        if not path:
            return
        
        second = Image.open(path).convert("L")

        first_gray = Image.fromarray(self.img_array).convert("L")
        first_skeleton = Image.fromarray(zhang_suen(first_gray))
        second_skeleton = Image.fromarray(zhang_suen(second))

        first_points = detect_minutiae(first_skeleton)
        second_points = detect_minutiae(second_skeleton)

        first_endings = sum(kind == "ending" for _, _, kind in first_points)
        second_endings = sum(kind == "ending" for _, _, kind in second_points)

        self.img_array = annotate_minutiae(first_skeleton, first_points)
        self.update_canvas()

        self.status_bar.config(
            text=(
                f"Obraz 1: {len(first_points)} minucji ({first_endings} zakończeń, "
                f"{len(first_points) - first_endings} bifurkacji); obraz 2: "
                f"{len(second_points)} ({second_endings} zakończeń, "
                f"{len(second_points) - second_endings} bifurkacji)"
            )
        )

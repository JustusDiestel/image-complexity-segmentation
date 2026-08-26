import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from complexity_seg.batch import process_folder
from complexity_seg.metrics.delentropy import Delentropy
from complexity_seg.metrics.entropy import ShannonEntropy
from complexity_seg.metrics.mean_frequency import MeanFrequency
from complexity_seg.metrics.median_frequency import MedianFrequency
from complexity_seg.pipeline import ComplexityPipeline
from complexity_seg.segmenters.kmeans import KMeansSegmenter


class ComplexitySegmentationApp:

    def __init__(self, root: tk.Tk):
        self.root = root

        self.root.title("Image Complexity Segmentation")
        self.root.geometry("600x600")

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()

        self.window_size = tk.IntVar(value=15)
        self.n_clusters = tk.IntVar(value=3)

        self.use_entropy = tk.BooleanVar(value=True)
        self.use_delentropy = tk.BooleanVar(value=True)
        self.use_mnf = tk.BooleanVar(value=True)
        self.use_mdf = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self) -> None:

        main_frame = ttk.Frame(
            self.root,
            padding=20,
        )

        main_frame.pack(
            fill="both",
            expand=True,
        )

        # Input folder
        ttk.Label(
            main_frame,
            text="Input folder",
        ).pack(anchor="w")

        input_frame = ttk.Frame(main_frame)
        input_frame.pack(
            fill="x",
            pady=(5, 15),
        )

        ttk.Entry(
            input_frame,
            textvariable=self.input_dir,
        ).pack(
            side="left",
            fill="x",
            expand=True,
        )

        ttk.Button(
            input_frame,
            text="Browse",
            command=self._select_input_dir,
        ).pack(
            side="left",
            padx=(10, 0),
        )

        # Output folder
        ttk.Label(
            main_frame,
            text="Output folder",
        ).pack(anchor="w")

        output_frame = ttk.Frame(main_frame)
        output_frame.pack(
            fill="x",
            pady=(5, 20),
        )

        ttk.Entry(
            output_frame,
            textvariable=self.output_dir,
        ).pack(
            side="left",
            fill="x",
            expand=True,
        )

        ttk.Button(
            output_frame,
            text="Browse",
            command=self._select_output_dir,
        ).pack(
            side="left",
            padx=(10, 0),
        )

        # Metrics
        ttk.Label(
            main_frame,
            text="Complexity metrics",
        ).pack(anchor="w")

        ttk.Checkbutton(
            main_frame,
            text="Shannon Entropy",
            variable=self.use_entropy,
        ).pack(anchor="w")

        ttk.Checkbutton(
            main_frame,
            text="Delentropy",
            variable=self.use_delentropy,
        ).pack(anchor="w")

        ttk.Checkbutton(
            main_frame,
            text="Mean Frequency",
            variable=self.use_mnf,
        ).pack(anchor="w")

        ttk.Checkbutton(
            main_frame,
            text="Median Frequency",
            variable=self.use_mdf,
        ).pack(anchor="w")

        # Window size
        ttk.Label(
            main_frame,
            text="Window size",
        ).pack(
            anchor="w",
            pady=(20, 5),
        )

        ttk.Spinbox(
            main_frame,
            from_=3,
            to=101,
            increment=2,
            textvariable=self.window_size,
            width=10,
        ).pack(anchor="w")

        # Clusters
        ttk.Label(
            main_frame,
            text="Number of complexity classes",
        ).pack(
            anchor="w",
            pady=(20, 5),
        )

        ttk.Spinbox(
            main_frame,
            from_=2,
            to=10,
            textvariable=self.n_clusters,
            width=10,
        ).pack(anchor="w")

        # Start button
        ttk.Button(
            main_frame,
            text="Create Dataset",
            command=self._create_dataset,
        ).pack(
            pady=30,
            fill="x",
        )

    def _select_input_dir(self) -> None:
        directory = filedialog.askdirectory()

        if directory:
            self.input_dir.set(directory)

    def _select_output_dir(self) -> None:
        directory = filedialog.askdirectory()

        if directory:
            self.output_dir.set(directory)

    def _create_dataset(self) -> None:

        input_dir = Path(self.input_dir.get())
        output_dir = Path(self.output_dir.get())

        if not input_dir.is_dir():
            messagebox.showerror(
                "Error",
                "Please select a valid input folder.",
            )
            return

        if not self.output_dir.get():
            messagebox.showerror(
                "Error",
                "Please select an output folder.",
            )
            return

        window_size = self.window_size.get()

        if window_size < 3 or window_size % 2 == 0:
            messagebox.showerror(
                "Error",
                "Window size must be odd and at least 3.",
            )
            return

        metrics = self._build_metrics()

        if not metrics:
            messagebox.showerror(
                "Error",
                "Select at least one complexity metric.",
            )
            return

        pipeline = ComplexityPipeline(
            metrics=metrics,
            segmenter=KMeansSegmenter(
                n_clusters=self.n_clusters.get(),
                random_state=42,
            ),
        )

        try:
            process_folder(
                input_dir=input_dir,
                output_dir=output_dir,
                pipeline=pipeline,
            )

        except Exception as error:
            messagebox.showerror(
                "Processing failed",
                str(error),
            )
            return

        messagebox.showinfo(
            "Finished",
            "Dataset successfully created.",
        )

    def _build_metrics(self):
        metrics = []

        window_size = self.window_size.get()

        if self.use_entropy.get():
            metrics.append(
                ShannonEntropy(
                    window_size=window_size,
                    bins=32,
                )
            )

        if self.use_delentropy.get():
            metrics.append(
                Delentropy(
                    window_size=window_size,
                    bins=16,
                )
            )

        if self.use_mnf.get():
            metrics.append(
                MeanFrequency(
                    window_size=window_size,
                )
            )

        if self.use_mdf.get():
            metrics.append(
                MedianFrequency(
                    window_size=window_size,
                )
            )

        return metrics


def run_app() -> None:
    root = tk.Tk()

    ComplexitySegmentationApp(root)

    root.mainloop()
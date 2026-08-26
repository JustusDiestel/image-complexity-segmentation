import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt

from complexity_seg.batch import process_folder
from complexity_seg.dataset_export import export_sample
from complexity_seg.metrics.delentropy import Delentropy
from complexity_seg.metrics.entropy import ShannonEntropy
from complexity_seg.metrics.mean_frequency import MeanFrequency
from complexity_seg.metrics.median_frequency import MedianFrequency
from complexity_seg.pipeline import ComplexityPipeline
from complexity_seg.segmenters.kmeans import KMeansSegmenter
from complexity_seg.visualization import show_result


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


class ComplexitySegmentationApp:
    def __init__(self, root: tk.Tk):
        self.root = root

        self.root.title("Image Complexity Segmentation")
        self.root.geometry("640x700")

        self.input_mode = tk.StringVar(value="single")
        self.input_path = tk.StringVar()
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

        ttk.Label(
            main_frame,
            text="Input mode",
        ).pack(anchor="w")

        mode_frame = ttk.Frame(main_frame)
        mode_frame.pack(
            fill="x",
            pady=(5, 15),
        )

        ttk.Radiobutton(
            mode_frame,
            text="Single Image",
            variable=self.input_mode,
            value="single",
            command=self._on_input_mode_changed,
        ).pack(side="left")

        ttk.Radiobutton(
            mode_frame,
            text="Folder",
            variable=self.input_mode,
            value="folder",
            command=self._on_input_mode_changed,
        ).pack(
            side="left",
            padx=(20, 0),
        )

        self.input_label = ttk.Label(
            main_frame,
            text="Input image",
        )
        self.input_label.pack(anchor="w")

        input_frame = ttk.Frame(main_frame)
        input_frame.pack(
            fill="x",
            pady=(5, 15),
        )

        ttk.Entry(
            input_frame,
            textvariable=self.input_path,
        ).pack(
            side="left",
            fill="x",
            expand=True,
        )

        ttk.Button(
            input_frame,
            text="Browse",
            command=self._select_input,
        ).pack(
            side="left",
            padx=(10, 0),
        )

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

        ttk.Separator(
            main_frame,
            orient="horizontal",
        ).pack(
            fill="x",
            pady=(0, 20),
        )

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

        self.process_button = ttk.Button(
            main_frame,
            text="Process Image",
            command=self._process,
        )
        self.process_button.pack(
            pady=(30, 10),
            fill="x",
        )

        ttk.Label(
            main_frame,
            text=(
                "Single Image: exports the result and opens the metric maps and "
                "segmentation directly. Folder: creates a dataset for all supported images."
            ),
            wraplength=580,
        ).pack(anchor="w")

    def _on_input_mode_changed(self) -> None:
        self.input_path.set("")

        if self.input_mode.get() == "single":
            self.input_label.config(text="Input image")
            self.process_button.config(text="Process Image")
        else:
            self.input_label.config(text="Input folder")
            self.process_button.config(text="Create Dataset")

    def _select_input(self) -> None:
        if self.input_mode.get() == "single":
            path = filedialog.askopenfilename(
                title="Select image",
                filetypes=[
                    (
                        "Image files",
                        "*.png *.jpg *.jpeg *.bmp *.tif *.tiff",
                    ),
                    ("All files", "*.*"),
                ],
            )
        else:
            path = filedialog.askdirectory(
                title="Select input folder",
            )

        if path:
            self.input_path.set(path)

    def _select_output_dir(self) -> None:
        directory = filedialog.askdirectory(
            title="Select output folder",
        )

        if directory:
            self.output_dir.set(directory)

    def _process(self) -> None:
        if not self.input_path.get():
            messagebox.showerror(
                "Error",
                "Please select an input image or folder.",
            )
            return

        if not self.output_dir.get():
            messagebox.showerror(
                "Error",
                "Please select an output folder.",
            )
            return

        try:
            window_size = self.window_size.get()
            n_clusters = self.n_clusters.get()
        except tk.TclError:
            messagebox.showerror(
                "Error",
                "Window size and number of classes must be integers.",
            )
            return

        if window_size < 3 or window_size % 2 == 0:
            messagebox.showerror(
                "Error",
                "Window size must be odd and at least 3.",
            )
            return

        if n_clusters < 2:
            messagebox.showerror(
                "Error",
                "Use at least 2 complexity classes.",
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
                n_clusters=n_clusters,
                random_state=42,
            ),
        )

        input_path = Path(self.input_path.get())
        output_dir = Path(self.output_dir.get())

        try:
            if self.input_mode.get() == "single":
                self._process_single_image(
                    image_path=input_path,
                    output_dir=output_dir,
                    pipeline=pipeline,
                )
            else:
                self._process_folder(
                    input_dir=input_path,
                    output_dir=output_dir,
                    pipeline=pipeline,
                )

        except Exception as error:
            messagebox.showerror(
                "Processing failed",
                str(error),
            )

    def _process_single_image(
        self,
        image_path: Path,
        output_dir: Path,
        pipeline: ComplexityPipeline,
    ) -> None:
        if not image_path.is_file():
            raise ValueError("Please select a valid image file.")

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError("Unsupported image format.")

        image = self._load_image(image_path)

        features, mask = pipeline.run(image)

        metadata = self._build_metadata(
            source_file=image_path.name,
            pipeline=pipeline,
        )

        export_sample(
            image=image,
            mask=mask,
            output_dir=output_dir,
            sample_name=image_path.stem,
            metadata=metadata,
        )

        messagebox.showinfo(
            "Finished",
            "Image processed and exported successfully.",
        )

        metric_names = [
            metric.name
            for metric in pipeline.metrics
        ]

        show_result(
            image=image,
            features=features,
            mask=mask,
            metric_names=metric_names,
        )

    def _process_folder(
        self,
        input_dir: Path,
        output_dir: Path,
        pipeline: ComplexityPipeline,
    ) -> None:
        if not input_dir.is_dir():
            raise ValueError("Please select a valid input folder.")

        process_folder(
            input_dir=input_dir,
            output_dir=output_dir,
            pipeline=pipeline,
        )

        messagebox.showinfo(
            "Finished",
            "Dataset successfully created.",
        )

    def _load_image(self, image_path: Path):
        image = plt.imread(image_path)

        if image.ndim == 3:
            image = image[:, :, :3].mean(axis=2)

        return image

    def _build_metadata(
        self,
        source_file: str,
        pipeline: ComplexityPipeline,
    ) -> dict:
        return {
            "source_file": source_file,
            "metrics": [
                {
                    "name": metric.name,
                    "window_size": getattr(
                        metric,
                        "window_size",
                        None,
                    ),
                    "bins": getattr(
                        metric,
                        "bins",
                        None,
                    ),
                }
                for metric in pipeline.metrics
            ],
            "segmenter": pipeline.segmenter.name,
            "n_clusters": getattr(
                pipeline.segmenter,
                "n_clusters",
                None,
            ),
            "classes": {
                "0": "low",
                "1": "medium",
                "2": "high",
            },
        }

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


if __name__ == "__main__":
    run_app()

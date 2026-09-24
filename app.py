from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from config import (
    APP_NAME,
    APP_VERSION,
    BATCH_OUTPUT_ROOT,
    IMAGE_PREVIEW_FALLBACK_HEIGHT,
    IMAGE_PREVIEW_FALLBACK_WIDTH,
    IMAGE_PREVIEW_MARGIN,
    IMAGE_PREVIEW_MIN_HEIGHT,
    IMAGE_PREVIEW_MIN_WIDTH,
    LOG_FILE,
    OCR_DEVICE,
    SINGLE_OUTPUT_ROOT,
    SUPPORTED_EXTENSIONS,
    WINDOW_HEIGHT,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
    WINDOW_WIDTH,
    configure_logging,
    ensure_project_directories,
)

from scripts.document_processing_engine import (
    DocumentProcessingEngine,
)


logger = configure_logging()


# =============================================================================
# APPLICATION
# =============================================================================

class DocumentUnderstandingApp:
    def __init__(
        self,
        root: tk.Tk,
    ):
        self.root = root

        self.root.title(
            f"{APP_NAME} v{APP_VERSION}"
        )

        self.root.geometry(
            f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"
        )

        self.root.minsize(
            WINDOW_MIN_WIDTH,
            WINDOW_MIN_HEIGHT,
        )

        # =====================================================================
        # ENGINE
        # =====================================================================

        self.engine: DocumentProcessingEngine | None = None

        self.engine_lock = threading.Lock()

        self.engine_loading = False


        # =====================================================================
        # SINGLE DOCUMENT
        # =====================================================================

        self.input_path: Path | None = None

        self.current_output_dir: Path | None = None

        self.current_summary: dict | None = None

        # Generated locally after successful single-document OCR.
        self.single_qa_json_path: Path | None = None

        self.single_running = False

        self.preview_photo = None

        self.perspective_photo = None

        self.scan_photo = None

        self.ocr_photo = None


        # =====================================================================
        # BATCH
        # =====================================================================

        self.batch_files: list[Path] = []

        self.batch_rows: list[dict] = []

        self.batch_output_dir: Path | None = None

        self.batch_running = False


        # =====================================================================
        # DOCUMENT Q&A
        # =====================================================================

        self.qa_service = None
        self.qa_service_lock = threading.Lock()
        self.qa_running = False
        # None means the validated pic1 + pic13 demo corpus.
        # Otherwise Q&A loads just one explicitly selected unified JSON.
        self.qa_document_path: Path | None = None
        self.qa_request_counter = 0
        self.qa_active_request = None


        # =====================================================================
        # START
        # =====================================================================

        logger.info(
            "%s v%s starting",
            APP_NAME,
            APP_VERSION,
        )

        self.create_ui()

        self.start_engine_loading()


    # =========================================================================
    # OCR ENGINE
    # =========================================================================

    def start_engine_loading(
        self,
    ) -> None:
        if self.engine_loading:
            return

        self.engine_loading = True

        self.engine_status_var.set(
            "OCR engine: loading..."
        )

        worker = threading.Thread(
            target=self.load_engine,
            daemon=True,
        )

        worker.start()


    def load_engine(
        self,
    ) -> None:
        try:
            start_time = (
                time.perf_counter()
            )

            engine = (
                DocumentProcessingEngine(
                    device=OCR_DEVICE,
                    verbose=False,
                )
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            with self.engine_lock:
                self.engine = engine

            self.root.after(
                0,
                lambda seconds=elapsed:
                    self.engine_loaded(
                        seconds
                    ),
            )

        except Exception as error:
            error_message = str(
                error
            )

            logger.exception(
                "OCR engine initialization failed."
            )

            self.root.after(
                0,
                lambda message=error_message:
                    self.engine_failed(
                        message
                    ),
            )


    def engine_loaded(
        self,
        elapsed: float,
    ) -> None:
        self.engine_loading = False

        self.engine_status_var.set(
            (
                f"OCR engine: ready on "
                f"{OCR_DEVICE.upper()} "
                f"({elapsed:.1f}s load)"
            )
        )

        if (
            self.input_path is not None
            and not self.single_running
        ):
            self.process_button.configure(
                state="normal"
            )

        if (
            self.batch_files
            and not self.batch_running
        ):
            self.batch_process_button.configure(
                state="normal"
            )


    def engine_failed(
        self,
        error_message: str,
    ) -> None:
        self.engine_loading = False

        self.engine_status_var.set(
            "OCR engine: failed"
        )

        messagebox.showerror(
            "OCR Engine Error",
            (
                "PaddleOCR could not be loaded.\n\n"
                f"{error_message}\n\n"
                f"Log file:\n{LOG_FILE}"
            ),
        )


    def get_engine(
        self,
    ) -> DocumentProcessingEngine:
        with self.engine_lock:
            if self.engine is None:
                raise RuntimeError(
                    "OCR engine is not ready."
                )

            return self.engine


    # =========================================================================
    # MAIN UI
    # =========================================================================

    def create_ui(
        self,
    ) -> None:
        header = ttk.Frame(
            self.root,
            padding=10,
        )

        header.pack(
            fill="x"
        )

        ttk.Label(
            header,
            text=APP_NAME,
            font=(
                "Segoe UI",
                16,
                "bold",
            ),
        ).pack(
            side="left"
        )

        self.engine_status_var = (
            tk.StringVar(
                value=(
                    "OCR engine: starting..."
                )
            )
        )

        ttk.Label(
            header,
            textvariable=
                self.engine_status_var,
        ).pack(
            side="right"
        )

        ttk.Label(
            header,
            text=f"v{APP_VERSION}",
        ).pack(
            side="right",
            padx=(0, 15),
        )

        self.mode_tabs = ttk.Notebook(
            self.root
        )

        self.mode_tabs.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=(0, 10),
        )

        self.single_mode = ttk.Frame(
            self.mode_tabs
        )

        self.batch_mode = ttk.Frame(
            self.mode_tabs
        )

        self.qa_mode = ttk.Frame(
            self.mode_tabs
        )

        self.mode_tabs.add(
            self.single_mode,
            text="Single Document",
        )

        self.mode_tabs.add(
            self.batch_mode,
            text="Batch Processing",
        )

        self.mode_tabs.add(
            self.qa_mode,
            text="Document Q&A",
        )

        self.create_single_mode()

        self.create_batch_mode()

        self.create_document_qa_mode()


    # =========================================================================
    # SINGLE DOCUMENT UI
    # =========================================================================

    def create_single_mode(
        self,
    ) -> None:
        toolbar = ttk.Frame(
            self.single_mode,
            padding=10,
        )

        toolbar.pack(
            fill="x"
        )

        self.file_label = ttk.Label(
            toolbar,
            text="No document selected",
        )

        self.file_label.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 10),
        )

        self.select_button = ttk.Button(
            toolbar,
            text="Select Document",
            command=self.select_document,
        )

        self.select_button.pack(
            side="left",
            padx=4,
        )

        self.process_button = ttk.Button(
            toolbar,
            text="Process Document",
            command=
                self.start_single_processing,
            state="disabled",
        )

        self.process_button.pack(
            side="left",
            padx=4,
        )

        self.single_qa_button = ttk.Button(
            toolbar,
            text="Ask About This Scan",
            command=self.select_current_scan_for_qa,
            state="disabled",
        )
        self.single_qa_button.pack(side="left", padx=4)

        self.clear_button = ttk.Button(
            toolbar,
            text="Clear",
            command=self.clear_single,
        )

        self.clear_button.pack(
            side="left",
            padx=4,
        )


        # ---------------------------------------------------------------------
        # Status
        # ---------------------------------------------------------------------

        status_frame = ttk.LabelFrame(
            self.single_mode,
            text="Processing Status",
            padding=10,
        )

        status_frame.pack(
            fill="x",
            padx=10,
            pady=(0, 10),
        )

        self.status_var = tk.StringVar(
            value="Ready"
        )

        self.detector_var = tk.StringVar(
            value="Detector: -"
        )

        self.route_var = tk.StringVar(
            value="OCR route: -"
        )

        self.quality_var = tk.StringVar(
            value="Quality: -"
        )

        self.stats_var = tk.StringVar(
            value="OCR: -"
        )

        self.time_var = tk.StringVar(
            value="Time: -"
        )

        status_variables = [
            self.status_var,
            self.detector_var,
            self.route_var,
            self.quality_var,
            self.stats_var,
            self.time_var,
        ]

        for column, variable in enumerate(
            status_variables
        ):
            ttk.Label(
                status_frame,
                textvariable=variable,
            ).grid(
                row=0,
                column=column,
                sticky="w",
                padx=8,
            )

        self.single_progress = (
            ttk.Progressbar(
                status_frame,
                mode="indeterminate",
            )
        )

        self.single_progress.grid(
            row=1,
            column=0,
            columnspan=6,
            sticky="ew",
            padx=8,
            pady=(10, 0),
        )

        status_frame.columnconfigure(
            5,
            weight=1,
        )


        # ---------------------------------------------------------------------
        # Result tabs
        # ---------------------------------------------------------------------

        self.result_tabs = ttk.Notebook(
            self.single_mode
        )

        self.result_tabs.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=(0, 10),
        )

        self.original_tab = ttk.Frame(
            self.result_tabs
        )

        self.perspective_tab = ttk.Frame(
            self.result_tabs
        )

        self.scan_tab = ttk.Frame(
            self.result_tabs
        )

        self.ocr_tab = ttk.Frame(
            self.result_tabs
        )

        self.text_tab = ttk.Frame(
            self.result_tabs
        )

        self.summary_tab = ttk.Frame(
            self.result_tabs
        )

        self.result_tabs.add(
            self.original_tab,
            text="Original",
        )

        self.result_tabs.add(
            self.perspective_tab,
            text="Perspective",
        )

        self.result_tabs.add(
            self.scan_tab,
            text="Clean Scan",
        )

        self.result_tabs.add(
            self.ocr_tab,
            text="OCR Regions",
        )

        self.result_tabs.add(
            self.text_tab,
            text="Recognized Text",
        )

        self.result_tabs.add(
            self.summary_tab,
            text="Processing Summary",
        )


        # ---------------------------------------------------------------------
        # Images
        # ---------------------------------------------------------------------

        self.original_image_label = (
            ttk.Label(
                self.original_tab,
                anchor="center",
            )
        )

        self.original_image_label.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10,
        )

        self.perspective_image_label = (
            ttk.Label(
                self.perspective_tab,
                anchor="center",
            )
        )

        self.perspective_image_label.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10,
        )


        # ---------------------------------------------------------------------
        # Scan tab
        # ---------------------------------------------------------------------

        scan_toolbar = ttk.Frame(
            self.scan_tab,
            padding=5,
        )

        scan_toolbar.pack(
            fill="x"
        )

        self.export_scan_button = (
            ttk.Button(
                scan_toolbar,
                text="Export Clean Scan",
                command=self.export_scan,
                state="disabled",
            )
        )

        self.export_scan_button.pack(
            side="right",
            padx=5,
        )

        self.scan_image_label = ttk.Label(
            self.scan_tab,
            anchor="center",
        )

        self.scan_image_label.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10,
        )


        # ---------------------------------------------------------------------
        # OCR tab
        # ---------------------------------------------------------------------

        ocr_toolbar = ttk.Frame(
            self.ocr_tab,
            padding=5,
        )

        ocr_toolbar.pack(
            fill="x"
        )

        self.export_csv_button = ttk.Button(
            ocr_toolbar,
            text="Export OCR CSV",
            command=self.export_ocr_csv,
            state="disabled",
        )

        self.export_csv_button.pack(
            side="right",
            padx=5,
        )

        self.ocr_image_label = ttk.Label(
            self.ocr_tab,
            anchor="center",
        )

        self.ocr_image_label.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10,
        )


        # ---------------------------------------------------------------------
        # Text tab
        # ---------------------------------------------------------------------

        text_toolbar = ttk.Frame(
            self.text_tab,
            padding=5,
        )

        text_toolbar.pack(
            fill="x"
        )

        self.copy_text_button = ttk.Button(
            text_toolbar,
            text="Copy Text",
            command=self.copy_text,
            state="disabled",
        )

        self.copy_text_button.pack(
            side="right",
            padx=5,
        )

        self.export_text_button = ttk.Button(
            text_toolbar,
            text="Export Text",
            command=self.export_text,
            state="disabled",
        )

        self.export_text_button.pack(
            side="right",
            padx=5,
        )

        text_frame = ttk.Frame(
            self.text_tab,
            padding=10,
        )

        text_frame.pack(
            fill="both",
            expand=True,
        )

        self.recognized_text = tk.Text(
            text_frame,
            wrap="word",
            font=(
                "Consolas",
                11,
            ),
        )

        text_scroll = ttk.Scrollbar(
            text_frame,
            orient="vertical",
            command=
                self.recognized_text.yview,
        )

        self.recognized_text.configure(
            yscrollcommand=
                text_scroll.set
        )

        self.recognized_text.pack(
            side="left",
            fill="both",
            expand=True,
        )

        text_scroll.pack(
            side="right",
            fill="y",
        )


        # ---------------------------------------------------------------------
        # Summary tab
        # ---------------------------------------------------------------------

        summary_toolbar = ttk.Frame(
            self.summary_tab,
            padding=5,
        )

        summary_toolbar.pack(
            fill="x"
        )

        self.open_output_button = ttk.Button(
            summary_toolbar,
            text="Open Output Folder",
            command=self.open_single_output,
            state="disabled",
        )

        self.open_output_button.pack(
            side="right",
            padx=5,
        )

        self.open_log_button = ttk.Button(
            summary_toolbar,
            text="Open Log File",
            command=self.open_log_file,
        )

        self.open_log_button.pack(
            side="right",
            padx=5,
        )

        summary_frame = ttk.Frame(
            self.summary_tab,
            padding=10,
        )

        summary_frame.pack(
            fill="both",
            expand=True,
        )

        self.summary_text = tk.Text(
            summary_frame,
            wrap="none",
            font=(
                "Consolas",
                10,
            ),
        )

        summary_y = ttk.Scrollbar(
            summary_frame,
            orient="vertical",
            command=self.summary_text.yview,
        )

        summary_x = ttk.Scrollbar(
            summary_frame,
            orient="horizontal",
            command=self.summary_text.xview,
        )

        self.summary_text.configure(
            yscrollcommand=
                summary_y.set,
            xscrollcommand=
                summary_x.set,
        )

        self.summary_text.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        summary_y.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        summary_x.grid(
            row=1,
            column=0,
            sticky="ew",
        )

        summary_frame.rowconfigure(
            0,
            weight=1,
        )

        summary_frame.columnconfigure(
            0,
            weight=1,
        )


    # =========================================================================
    # BATCH UI
    # =========================================================================

    def create_batch_mode(
        self,
    ) -> None:
        toolbar = ttk.Frame(
            self.batch_mode,
            padding=10,
        )

        toolbar.pack(
            fill="x"
        )

        self.batch_file_count_var = (
            tk.StringVar(
                value=(
                    "No documents selected"
                )
            )
        )

        ttk.Label(
            toolbar,
            textvariable=
                self.batch_file_count_var,
        ).pack(
            side="left",
            fill="x",
            expand=True,
        )

        self.batch_select_button = (
            ttk.Button(
                toolbar,
                text=(
                    "Select Multiple Documents"
                ),
                command=
                    self.select_batch_files,
            )
        )

        self.batch_select_button.pack(
            side="left",
            padx=4,
        )

        self.batch_process_button = (
            ttk.Button(
                toolbar,
                text="Process Batch",
                command=
                    self.start_batch_processing,
                state="disabled",
            )
        )

        self.batch_process_button.pack(
            side="left",
            padx=4,
        )

        self.batch_clear_button = ttk.Button(
            toolbar,
            text="Clear Batch",
            command=self.clear_batch,
        )

        self.batch_clear_button.pack(
            side="left",
            padx=4,
        )


        status_frame = ttk.LabelFrame(
            self.batch_mode,
            text="Batch Status",
            padding=10,
        )

        status_frame.pack(
            fill="x",
            padx=10,
            pady=(0, 10),
        )

        self.batch_status_var = (
            tk.StringVar(
                value="Ready"
            )
        )

        self.batch_progress_var = (
            tk.StringVar(
                value="0 / 0"
            )
        )

        self.batch_time_var = (
            tk.StringVar(
                value="Elapsed: 0.0 s"
            )
        )

        ttk.Label(
            status_frame,
            textvariable=
                self.batch_status_var,
        ).pack(
            side="left",
            padx=10,
        )

        ttk.Label(
            status_frame,
            textvariable=
                self.batch_time_var,
        ).pack(
            side="right",
            padx=10,
        )

        ttk.Label(
            status_frame,
            textvariable=
                self.batch_progress_var,
        ).pack(
            side="right",
            padx=10,
        )

        self.batch_progress = (
            ttk.Progressbar(
                status_frame,
                mode="determinate",
                value=0,
            )
        )

        self.batch_progress.pack(
            fill="x",
            padx=10,
            pady=(30, 0),
        )


        controls = ttk.Frame(
            self.batch_mode,
            padding=(10, 0, 10, 5),
        )

        controls.pack(
            fill="x"
        )

        self.batch_export_button = (
            ttk.Button(
                controls,
                text=(
                    "Export Batch Summary CSV"
                ),
                command=
                    self.export_batch_csv,
                state="disabled",
            )
        )

        self.batch_export_button.pack(
            side="right",
            padx=5,
        )

        self.batch_open_button = ttk.Button(
            controls,
            text="Open Batch Folder",
            command=self.open_batch_folder,
            state="disabled",
        )

        self.batch_open_button.pack(
            side="right",
            padx=5,
        )


        # ---------------------------------------------------------------------
        # Table
        # ---------------------------------------------------------------------

        table_frame = ttk.Frame(
            self.batch_mode,
            padding=10,
        )

        table_frame.pack(
            fill="both",
            expand=True,
        )

        columns = (
            "filename",
            "status",
            "detector",
            "route",
            "quality",
            "regions",
            "words",
            "characters",
            "confidence",
            "seconds",
        )

        self.batch_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
        )

        headings = {
            "filename": "Filename",
            "status": "Status",
            "detector": "Detector",
            "route": "OCR Route",
            "quality": "Quality",
            "regions": "Regions",
            "words": "Words",
            "characters": "Characters",
            "confidence": "Confidence",
            "seconds": "Seconds",
        }

        widths = {
            "filename": 220,
            "status": 90,
            "detector": 130,
            "route": 140,
            "quality": 110,
            "regions": 80,
            "words": 80,
            "characters": 90,
            "confidence": 100,
            "seconds": 85,
        }

        for column in columns:
            self.batch_tree.heading(
                column,
                text=headings[
                    column
                ],
            )

            self.batch_tree.column(
                column,
                width=widths[
                    column
                ],
                anchor="center",
            )

        self.batch_tree.column(
            "filename",
            anchor="w",
        )

        table_y = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.batch_tree.yview,
        )

        table_x = ttk.Scrollbar(
            table_frame,
            orient="horizontal",
            command=self.batch_tree.xview,
        )

        self.batch_tree.configure(
            yscrollcommand=table_y.set,
            xscrollcommand=table_x.set,
        )

        self.batch_tree.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        table_y.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        table_x.grid(
            row=1,
            column=0,
            sticky="ew",
        )

        table_frame.rowconfigure(
            0,
            weight=1,
        )

        table_frame.columnconfigure(
            0,
            weight=1,
        )

        self.batch_tree.bind(
            "<Double-1>",
            self.open_selected_batch_result,
        )

        self.batch_summary_var = (
            tk.StringVar(
                value=(
                    "Successful: 0 | "
                    "Strong: 0 | "
                    "Limited: 0 | "
                    "Insufficient: 0 | "
                    "Failed: 0"
                )
            )
        )

        ttk.Label(
            self.batch_mode,
            textvariable=
                self.batch_summary_var,
            padding=10,
        ).pack(
            fill="x"
        )


    # =========================================================================
    # DOCUMENT Q&A
    # =========================================================================

    def create_document_qa_mode(
        self,
    ) -> None:
        container = ttk.Frame(
            self.qa_mode,
            padding=15,
        )
        container.pack(
            fill="both",
            expand=True,
        )

        intro = ttk.LabelFrame(
            container,
            text="Grounded Document Question Answering",
            padding=12,
        )
        intro.pack(
            fill="x",
            pady=(0, 10),
        )

        ttk.Label(
            intro,
            text=(
                "Ask questions about a selected unified document. "
                "Only evidence selected by retrieval is sent to Groq when you ask."
            ),
            wraplength=1000,
            justify="left",
        ).pack(anchor="w")

        ttk.Label(
            intro,
            text=(
                "Use the demo corpus, load a unified JSON, or choose "
                "Ask About This Scan after local processing. New scans use raw OCR "
                "regions, not verified form fields or table structures."
            ),
            wraplength=1000,
            justify="left",
        ).pack(
            anchor="w",
            pady=(8, 0),
        )

        scope_frame = ttk.LabelFrame(
            container,
            text="Document scope",
            padding=10,
        )
        scope_frame.pack(fill="x", pady=(0, 10))

        self.qa_scope_var = tk.StringVar(
            value="Validated demo corpus: pic1 + pic13"
        )
        ttk.Label(
            scope_frame,
            textvariable=self.qa_scope_var,
            wraplength=950,
            justify="left",
        ).pack(side="left", fill="x", expand=True, padx=(0, 8))

        self.qa_demo_button = ttk.Button(
            scope_frame,
            text="Use demo corpus",
            command=self.use_qa_demo_corpus,
        )
        self.qa_demo_button.pack(side="right", padx=4)

        self.qa_load_button = ttk.Button(
            scope_frame,
            text="Load unified JSON",
            command=self.choose_qa_document,
        )
        self.qa_load_button.pack(side="right", padx=4)

        question_frame = ttk.LabelFrame(
            container,
            text="Question",
            padding=12,
        )
        question_frame.pack(
            fill="x",
            pady=(0, 10),
        )

        self.qa_question_var = tk.StringVar()
        self.qa_question_entry = ttk.Entry(
            question_frame,
            textvariable=self.qa_question_var,
        )
        self.qa_question_entry.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 8),
        )
        self.qa_question_entry.bind(
            "<Return>",
            self.start_document_question,
        )
        self.qa_question_entry.bind("<KeyRelease>", self.qa_question_edited, add="+")

        self.qa_ask_button = ttk.Button(
            question_frame,
            text="Ask Document",
            command=self.start_document_question,
        )
        self.qa_ask_button.pack(
            side="left",
            padx=4,
        )

        self.qa_clear_button = ttk.Button(
            question_frame,
            text="Clear",
            command=self.clear_document_qa,
        )
        self.qa_clear_button.pack(
            side="left",
            padx=4,
        )

        status_frame = ttk.Frame(container)
        status_frame.pack(
            fill="x",
            pady=(0, 10),
        )

        self.qa_status_var = tk.StringVar(
            value=(
                "Ready. GROQ_API_KEY will be checked when the first "
                "question is asked."
            )
        )
        ttk.Label(
            status_frame,
            textvariable=self.qa_status_var,
        ).pack(
            side="left",
            fill="x",
            expand=True,
        )

        self.qa_progress = ttk.Progressbar(
            status_frame,
            mode="indeterminate",
            length=180,
        )
        self.qa_progress.pack(side="right")

        panes = ttk.Panedwindow(
            container,
            orient="vertical",
        )
        panes.pack(
            fill="both",
            expand=True,
        )

        answer_frame = ttk.LabelFrame(
            panes,
            text="Answer",
            padding=10,
        )
        evidence_frame = ttk.LabelFrame(
            panes,
            text="Grounding / Evidence",
            padding=10,
        )
        panes.add(answer_frame, weight=2)
        panes.add(evidence_frame, weight=1)

        self.qa_answer_text = tk.Text(
            answer_frame,
            wrap="word",
            font=("Segoe UI", 11),
            height=10,
        )
        answer_scroll = ttk.Scrollbar(
            answer_frame,
            orient="vertical",
            command=self.qa_answer_text.yview,
        )
        self.qa_answer_text.configure(
            yscrollcommand=answer_scroll.set
        )
        self.qa_answer_text.pack(
            side="left",
            fill="both",
            expand=True,
        )
        answer_scroll.pack(
            side="right",
            fill="y",
        )

        self.qa_evidence_text = tk.Text(
            evidence_frame,
            wrap="word",
            font=("Consolas", 10),
            height=8,
        )
        evidence_scroll = ttk.Scrollbar(
            evidence_frame,
            orient="vertical",
            command=self.qa_evidence_text.yview,
        )
        self.qa_evidence_text.configure(
            yscrollcommand=evidence_scroll.set
        )
        self.qa_evidence_text.pack(
            side="left",
            fill="both",
            expand=True,
        )
        evidence_scroll.pack(
            side="right",
            fill="y",
        )


    def get_qa_service(self, document_path=None):
        """Lazy service initialization for the selected Q&A document scope."""
        with self.qa_service_lock:
            if self.qa_service is None:
                from services.document_qa_service import DocumentQAService

                paths = None if document_path is None else [document_path]
                self.qa_service = DocumentQAService(document_paths=paths)
            return self.qa_service

    def reset_qa_service(self):
        """Invalidate a previously loaded corpus when the user changes scope."""
        with self.qa_service_lock:
            self.qa_service = None

    def choose_qa_document(self):
        if self.qa_running:
            return
        filename = filedialog.askopenfilename(
            title="Select a unified document JSON",
            filetypes=[("JSON files", "*.json")],
        )
        if not filename:
            return
        path = Path(filename).resolve()
        try:
            # Check schema and semantic content without requiring an API key.
            from services.document_qa_service import DocumentQAService
            # Validate with the existing corpus builder but do not instantiate
            # an API client merely to change document scope.
            from experiments.experiment_41_generic_cross_document_retrieval import (
                load_json, build_search_corpus,
            )
            document = load_json(path)
            required = {
                "document_id", "schema_version", "document_type",
                "records", "fields", "controls", "free_text", "evidence",
            }
            if not isinstance(document, dict) or required - document.keys():
                raise ValueError("Choose a unified-document JSON, not a raw OCR JSON.")
            if not all(isinstance(document[key], list) for key in (
                "records", "fields", "controls", "free_text", "evidence"
            )):
                raise ValueError("Unified document semantic fields must be lists.")
            if not any(document[key] for key in (
                "records", "fields", "controls", "free_text"
            )) or not build_search_corpus([document]):
                raise ValueError("This JSON has no searchable structured items.")
        except Exception as error:
            messagebox.showerror("Invalid Q&A document", str(error))
            return

        self.qa_document_path = path
        self.reset_qa_service()
        self.qa_scope_var.set(
            f"Selected unified document: {document['document_id']} ({path.name})"
        )
        self.clear_document_qa()
        self.qa_status_var.set("Document scope changed. Ready to ask.")

    def use_qa_demo_corpus(self):
        if self.qa_running:
            return
        self.qa_document_path = None
        self.reset_qa_service()
        self.qa_scope_var.set("Validated demo corpus: pic1 + pic13")
        self.clear_document_qa()
        self.qa_status_var.set("Demo corpus selected. Ready to ask.")

    def qa_question_edited(self, event=None) -> None:
        """A newly typed question must not display a previous question's answer."""
        if self.qa_running:
            return
        self.qa_answer_text.delete("1.0", tk.END)
        self.qa_evidence_text.delete("1.0", tk.END)
        self.qa_status_var.set("Question changed. Click Ask Document for a new answer.")

    def start_document_question(
        self,
        event=None,
    ) -> None:
        if self.qa_running:
            return

        question = self.qa_question_var.get().strip()
        if not question:
            messagebox.showinfo(
                "Document Q&A",
                "Please enter a question first.",
            )
            return

        self.qa_running = True
        self.qa_request_counter += 1
        request_id = self.qa_request_counter
        self.qa_active_request = (request_id, question, self.qa_document_path)
        self.qa_ask_button.configure(state="disabled")
        self.qa_clear_button.configure(state="disabled")
        self.qa_load_button.configure(state="disabled")
        self.qa_demo_button.configure(state="disabled")
        self.qa_question_entry.configure(state="disabled")
        self.qa_progress.start(10)
        self.qa_status_var.set(
            "Retrieving evidence and asking Groq..."
        )
        self.qa_answer_text.delete("1.0", tk.END)
        self.qa_evidence_text.delete("1.0", tk.END)
        self.qa_answer_text.insert(tk.END, "Working...")

        worker = threading.Thread(
            target=self.process_document_question,
            args=(question, self.qa_document_path, request_id),
            daemon=True,
        )
        worker.start()


    def process_document_question(
        self,
        question: str,
        document_path: Path | None,
        request_id: int,
    ) -> None:
        try:
            start_time = time.perf_counter()
            result = self.get_qa_service(document_path).answer(question)
            elapsed = time.perf_counter() - start_time
            self.root.after(
                0,
                lambda response=result, seconds=elapsed, rid=request_id:
                    self.document_question_complete(
                        response,
                        seconds,
                        rid,
                    ),
            )
        except Exception as error:
            error_message = str(error)
            logger.exception("Document Q&A failed.")
            self.root.after(
                0,
                lambda message=error_message, rid=request_id:
                    self.document_question_failed(message, rid),
            )


    def document_question_complete(
        self,
        result: dict,
        elapsed: float,
        request_id: int,
    ) -> None:
        expected = self.qa_active_request
        if expected is None or expected[0] != request_id:
            return  # Late response to an obsolete request.
        _, asked_question, asked_scope = expected
        if (result.get("question") != asked_question
                or self.qa_document_path != asked_scope):
            self.document_question_failed(
                "The completed response did not match the active question or scope.",
                request_id,
            )
            return
        # Restore the exact question associated with this completed response.
        self.qa_question_var.set(asked_question)
        self.qa_active_request = None
        self.qa_running = False
        self.qa_progress.stop()
        self.qa_ask_button.configure(state="normal")
        self.qa_clear_button.configure(state="normal")
        self.qa_load_button.configure(state="normal")
        self.qa_demo_button.configure(state="normal")
        self.qa_question_entry.configure(state="normal")

        self.qa_answer_text.delete("1.0", tk.END)
        self.qa_answer_text.insert(
            tk.END,
            result.get("answer", ""),
        )

        evidence = result.get("evidence", [])
        additional_context = result.get("additional_spatial_context", [])
        validation = result.get("evidence_validation", {})
        usage = result.get("usage", {})

        lines = [
            f"Answered question: {result.get('question', '-')}",
            f"Grounded: {result.get('grounded', False)}",
            f"Provider: {result.get('provider', '-')}",
            f"Model: {result.get('model', '-')}",
            f"Answer mode: {result.get('answer_mode', '-')}",
            f"Evidence validation: {validation.get('valid', False)}",
            f"Answer consistency: {result.get('answer_consistency', {}).get('reason', '-')}",
            f"Uncertainty: {result.get('uncertainty')}",
            "",
            "Model-cited evidence:",
        ]

        if evidence:
            for item in evidence:
                content = item.get("content") or {}
                detail = content.get("text") if item.get("item_type") == "ocr_region" else None
                lines.append(
                    "  - "
                    f"{item.get('evidence_item_id', '-')} | "
                    f"{item.get('document_id', '-')} | "
                    f"{item.get('item_type', '-')}"
                    + (f" | {detail}" if detail is not None else "")
                )
        else:
            lines.append("  None")

        lines.extend(["", "Additional spatial context (not model-cited):"])
        if additional_context:
            for entry in additional_context:
                item = entry["evidence"]
                detail = (item.get("content") or {}).get("text", "")
                lines.append(
                    f"  - {item.get('evidence_item_id', '-')} | {detail}"
                    f" | related to {entry.get('related_model_citation_id', '-')}"
                    " | spatial candidate, NOT a verified field"
                )
        else:
            lines.append("  None")

        lines.extend([
            "",
            f"Token usage: {usage}",
            f"Response time: {elapsed:.2f} s",
        ])

        self.qa_evidence_text.delete("1.0", tk.END)
        self.qa_evidence_text.insert(
            tk.END,
            "\n".join(lines),
        )

        if result.get("grounded", False):
            self.qa_status_var.set(
                f"Answer complete and grounded ({elapsed:.2f}s)"
            )
        else:
            self.qa_status_var.set(
                f"No sufficient document evidence ({elapsed:.2f}s)"
            )

        self.qa_question_entry.focus_set()


    def document_question_failed(
        self,
        error_message: str,
        request_id: int | None = None,
    ) -> None:
        if request_id is not None and (
            self.qa_active_request is None or self.qa_active_request[0] != request_id
        ):
            return
        self.qa_active_request = None
        self.qa_running = False
        self.qa_progress.stop()
        self.qa_ask_button.configure(state="normal")
        self.qa_clear_button.configure(state="normal")
        self.qa_load_button.configure(state="normal")
        self.qa_demo_button.configure(state="normal")
        self.qa_question_entry.configure(state="normal")
        self.qa_answer_text.delete("1.0", tk.END)
        self.qa_answer_text.insert(
            tk.END,
            "The document question could not be completed.",
        )
        self.qa_status_var.set("Document Q&A failed")

        messagebox.showerror(
            "Document Q&A Error",
            (
                f"{error_message}\n\n"
                "Check GROQ_API_KEY, your internet connection, "
                "and the application log."
            ),
        )


    def clear_document_qa(
        self,
    ) -> None:
        if self.qa_running:
            return

        self.qa_question_var.set("")
        self.qa_answer_text.delete("1.0", tk.END)
        self.qa_evidence_text.delete("1.0", tk.END)
        self.qa_status_var.set("Ready")
        self.qa_question_entry.focus_set()


    # =========================================================================
    # SINGLE DOCUMENT
    # =========================================================================

    def select_document(
        self,
    ) -> None:
        filename = (
            filedialog.askopenfilename(
                title=(
                    "Select Document Photograph"
                ),
                filetypes=[
                    (
                        "Image files",
                        "*.jpg *.jpeg *.png *.bmp *.webp",
                    ),
                    (
                        "All files",
                        "*.*",
                    ),
                ],
            )
        )

        if not filename:
            return

        path = Path(
            filename
        )

        if (
            path.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):
            messagebox.showerror(
                "Unsupported File",
                (
                    "Please select a "
                    "supported image file."
                ),
            )

            return

        self.input_path = path

        # Changing the image does not create a structured unified JSON.
        # Invalidate any earlier Q&A output to avoid implying it came from
        # the newly selected image.
        if not self.qa_running:
            self.qa_answer_text.delete("1.0", tk.END)
            self.qa_evidence_text.delete("1.0", tk.END)
            self.qa_status_var.set(
                "New image selected. Q&A still uses its explicitly displayed scope."
            )

        self.current_output_dir = None
        self.current_summary = None
        self.single_qa_json_path = None
        self.single_qa_button.configure(state="disabled")

        self.file_label.configure(
            text=str(
                path
            )
        )

        self.reset_single_status()

        self.status_var.set(
            "Document selected"
        )

        if (
            self.engine is not None
            and not self.single_running
        ):
            self.process_button.configure(
                state="normal"
            )

        self.disable_single_exports()

        self.clear_single_results()

        self.display_image(
            path,
            self.original_image_label,
            "preview_photo",
        )

        self.result_tabs.select(
            self.original_tab
        )


    def start_single_processing(
        self,
    ) -> None:
        if self.input_path is None:
            return

        if self.engine is None:
            messagebox.showinfo(
                "OCR Engine",
                (
                    "The OCR engine "
                    "is still loading."
                ),
            )

            return

        if self.single_running:
            return

        self.single_running = True
        self.single_qa_json_path = None
        self.single_qa_button.configure(state="disabled")

        self.process_button.configure(
            state="disabled"
        )

        self.select_button.configure(
            state="disabled"
        )

        self.clear_button.configure(
            state="disabled"
        )

        self.disable_single_exports()

        self.single_progress.start(
            10
        )

        self.status_var.set(
            "Processing..."
        )

        self.time_var.set(
            "Time: processing..."
        )

        logger.info(
            "GUI processing started: %s",
            self.input_path,
        )

        worker = threading.Thread(
            target=self.process_single,
            daemon=True,
        )

        worker.start()


    def process_single(
        self,
    ) -> None:
        try:
            input_path = self.input_path

            if input_path is None:
                raise RuntimeError(
                    "No input document selected."
                )

            output_dir = (
                SINGLE_OUTPUT_ROOT
                / input_path.stem
            )

            start_time = (
                time.perf_counter()
            )

            summary = (
                self.get_engine().process(
                    input_path,
                    output_dir,
                )
            )

            # Local-only conversion. Failure must not discard successful OCR.
            # Do not reuse an old Q&A file from a previous processing run.
            qa_path = output_dir / "unified_raw_ocr.json"
            try:
                qa_path.unlink(missing_ok=True)
                from experiments.experiment_49_ocr_unified_bridge import (
                    make_unified_ocr_document,
                )
                import hashlib
                import re

                source_id = hashlib.sha256(
                    str(input_path.resolve()).encode("utf-8")
                ).hexdigest()[:12]
                safe_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", input_path.stem)
                safe_stem = safe_stem[:80].lstrip("._-") or "scan"
                document = make_unified_ocr_document(
                    output_dir / "ocr" / "results.csv",
                    document_id=f"scan_{safe_stem}_{source_id}",
                    source_image=str(output_dir / "scan" / "scan.png"),
                )
                temp_path = qa_path.with_suffix(".json.tmp")
                try:
                    temp_path.write_text(
                        json.dumps(document, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    temp_path.replace(qa_path)
                finally:
                    temp_path.unlink(missing_ok=True)
                self.single_qa_json_path = qa_path.resolve()
            except Exception:
                self.single_qa_json_path = None
                logger.exception(
                    "OCR completed, but raw-OCR unified JSON generation failed."
                )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            self.current_output_dir = (
                output_dir
            )

            self.current_summary = (
                summary
            )

            self.root.after(
                0,
                lambda seconds=elapsed:
                    self.single_complete(
                        seconds
                    ),
            )

        except Exception as error:
            error_message = str(
                error
            )

            logger.exception(
                "Single-document GUI processing failed."
            )

            self.root.after(
                0,
                lambda message=error_message:
                    self.single_failed(
                        message
                    ),
            )


    def single_complete(
        self,
        elapsed: float,
    ) -> None:
        self.single_running = False

        self.single_progress.stop()

        self.select_button.configure(
            state="normal"
        )

        self.clear_button.configure(
            state="normal"
        )

        self.process_button.configure(
            state="normal"
        )

        summary = self.current_summary

        if summary is None:
            self.single_failed(
                (
                    "Processing completed "
                    "without a summary."
                )
            )

            return

        self.status_var.set(
            "Complete" if self.single_qa_json_path else "Complete (Q&A JSON unavailable; check log)"
        )
        self.single_qa_button.configure(
            state="normal" if self.single_qa_json_path else "disabled"
        )

        self.detector_var.set(
            (
                "Detector: "
                f"{summary.get('detector_route', '-')}"
            )
        )

        self.route_var.set(
            (
                "OCR route: "
                f"{summary.get('ocr_route', '-')}"
            )
        )

        quality = summary.get(
            "ocr_quality",
            "-",
        )

        self.quality_var.set(
            f"Quality: {quality}"
        )

        regions = int(
            summary.get(
                "meaningful_regions",
                0,
            )
        )

        words = int(
            summary.get(
                "words",
                0,
            )
        )

        confidence = float(
            summary.get(
                "mean_confidence",
                0.0,
            )
        )

        self.stats_var.set(
            (
                f"{regions} regions | "
                f"{words} words | "
                f"{confidence:.3f}"
            )
        )

        self.time_var.set(
            f"Time: {elapsed:.2f}s"
        )

        self.load_single_results()

        self.enable_single_exports()

        self.result_tabs.select(
            self.scan_tab
        )

        self.root.after(
            100,
            self.refresh_current_result_images,
        )

        logger.info(
            (
                "GUI processing completed "
                "in %.2f seconds."
            ),
            elapsed,
        )

        if quality == "INSUFFICIENT":
            messagebox.showwarning(
                "OCR Quality Warning",
                (
                    "The document was processed, "
                    "but the text could not be "
                    "reliably recovered."
                ),
            )

        elif quality == "LIMITED":
            messagebox.showwarning(
                "Limited OCR",
                (
                    "Only part of the document "
                    "text was reliably recovered."
                ),
            )


    def select_current_scan_for_qa(self) -> None:
        """Explicit local handoff; do not send OCR text to Groq here."""
        if self.single_running:
            return
        if self.qa_running:
            messagebox.showinfo(
                "Document Q&A", "Wait for the current question to finish."
            )
            return
        path = self.single_qa_json_path
        if path is None or not path.is_file():
            messagebox.showerror(
                "Document Q&A", "No current scan Q&A JSON. Process the image again."
            )
            self.single_qa_button.configure(state="disabled")
            return
        try:
            from experiments.experiment_41_generic_cross_document_retrieval import (
                load_json, build_search_corpus,
            )
            document = load_json(path)
            if (
                not isinstance(document, dict)
                or document.get("document_type") != "unclassified"
                or not document.get("free_text")
                or not build_search_corpus([document])
            ):
                raise ValueError("The current scan has no searchable raw OCR regions.")
        except Exception as error:
            messagebox.showerror("Document Q&A", str(error))
            return

        self.qa_document_path = path
        self.reset_qa_service()
        self.qa_scope_var.set(
            f"Selected raw-OCR scan: {document['document_id']} ({path.name})"
        )
        self.clear_document_qa()
        self.qa_status_var.set(
            "Current scan selected. Asking a question will send selected evidence to Groq."
        )
        self.mode_tabs.select(self.qa_mode)

    def single_failed(
        self,
        error_message: str,
    ) -> None:
        self.single_running = False

        self.single_progress.stop()

        self.select_button.configure(
            state="normal"
        )

        self.clear_button.configure(
            state="normal"
        )

        if (
            self.input_path is not None
            and self.engine is not None
        ):
            self.process_button.configure(
                state="normal"
            )

        self.status_var.set(
            "Failed"
        )

        self.time_var.set(
            "Time: -"
        )

        messagebox.showerror(
            "Processing Error",
            (
                f"{error_message}\n\n"
                f"Additional details were "
                f"written to:\n{LOG_FILE}"
            ),
        )


    # =========================================================================
    # SINGLE RESULTS
    # =========================================================================

    def load_single_results(
        self,
    ) -> None:
        if self.current_output_dir is None:
            return

        output_dir = (
            self.current_output_dir
        )

        image_outputs = [
            (
                output_dir
                / "perspective.png",
                self.perspective_image_label,
                "perspective_photo",
            ),
            (
                output_dir
                / "scan"
                / "scan.png",
                self.scan_image_label,
                "scan_photo",
            ),
            (
                output_dir
                / "ocr"
                / "regions.png",
                self.ocr_image_label,
                "ocr_photo",
            ),
        ]

        for (
            path,
            label,
            attribute_name,
        ) in image_outputs:
            if path.exists():
                self.display_image(
                    path,
                    label,
                    attribute_name,
                )

        text_path = (
            output_dir
            / "ocr"
            / "text.txt"
        )

        self.recognized_text.delete(
            "1.0",
            tk.END,
        )

        if text_path.exists():
            text = text_path.read_text(
                encoding="utf-8"
            )

            self.recognized_text.insert(
                tk.END,
                (
                    text
                    if text.strip()
                    else (
                        "No reliable text "
                        "was recovered."
                    )
                ),
            )

        self.summary_text.delete(
            "1.0",
            tk.END,
        )

        self.summary_text.insert(
            tk.END,
            json.dumps(
                self.current_summary,
                indent=4,
                ensure_ascii=False,
            ),
        )


    def refresh_current_result_images(
        self,
    ) -> None:
        if self.current_output_dir is None:
            return

        output_dir = (
            self.current_output_dir
        )

        for (
            path,
            label,
            attribute_name,
        ) in (
            (
                output_dir
                / "perspective.png",
                self.perspective_image_label,
                "perspective_photo",
            ),
            (
                output_dir
                / "scan"
                / "scan.png",
                self.scan_image_label,
                "scan_photo",
            ),
            (
                output_dir
                / "ocr"
                / "regions.png",
                self.ocr_image_label,
                "ocr_photo",
            ),
        ):
            if path.exists():
                self.display_image(
                    path,
                    label,
                    attribute_name,
                )


    # =========================================================================
    # BATCH
    # =========================================================================

    def select_batch_files(
        self,
    ) -> None:
        filenames = (
            filedialog.askopenfilenames(
                title=(
                    "Select Document Images"
                ),
                filetypes=[
                    (
                        "Image files",
                        "*.jpg *.jpeg *.png *.bmp *.webp",
                    ),
                    (
                        "All files",
                        "*.*",
                    ),
                ],
            )
        )

        if not filenames:
            return

        self.batch_files = [
            Path(
                filename
            )
            for filename in filenames
            if (
                Path(
                    filename
                ).suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]

        self.batch_file_count_var.set(
            (
                f"{len(self.batch_files)} "
                f"documents selected"
            )
        )

        self.batch_status_var.set(
            "Ready to process batch"
        )

        if (
            self.batch_files
            and self.engine is not None
            and not self.batch_running
        ):
            self.batch_process_button.configure(
                state="normal"
            )


    def start_batch_processing(
        self,
    ) -> None:
        if not self.batch_files:
            return

        if self.engine is None:
            messagebox.showinfo(
                "OCR Engine",
                (
                    "The OCR engine "
                    "is still loading."
                ),
            )

            return

        if self.batch_running:
            return

        self.batch_running = True

        self.batch_select_button.configure(
            state="disabled"
        )

        self.batch_process_button.configure(
            state="disabled"
        )

        self.batch_clear_button.configure(
            state="disabled"
        )

        self.batch_export_button.configure(
            state="disabled"
        )

        self.batch_open_button.configure(
            state="disabled"
        )

        self.clear_batch_table()

        total = len(
            self.batch_files
        )

        self.batch_progress.configure(
            maximum=total,
            value=0,
        )

        self.batch_progress_var.set(
            f"0 / {total}"
        )

        self.batch_time_var.set(
            "Elapsed: 0.0 s"
        )

        self.batch_status_var.set(
            "Processing batch..."
        )

        worker = threading.Thread(
            target=self.process_batch,
            daemon=True,
        )

        worker.start()


    def process_batch(
        self,
    ) -> None:
        try:
            run_name = time.strftime(
                "batch_%Y%m%d_%H%M%S"
            )

            output_dir = (
                BATCH_OUTPUT_ROOT
                / run_name
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.batch_output_dir = (
                output_dir
            )

            rows = []

            engine = self.get_engine()

            batch_start = (
                time.perf_counter()
            )

            total = len(
                self.batch_files
            )

            for index, image_path in enumerate(
                self.batch_files,
                start=1,
            ):
                document_output = (
                    output_dir
                    / (
                        f"{index:03d}_"
                        f"{image_path.stem}"
                    )
                )

                start_time = (
                    time.perf_counter()
                )

                try:
                    summary = engine.process(
                        image_path,
                        document_output,
                    )

                    elapsed = (
                        time.perf_counter()
                        - start_time
                    )

                    row = {
                        "filename":
                            image_path.name,

                        "status":
                            "SUCCESS",

                        "detector_route":
                            summary.get(
                                "detector_route",
                                "",
                            ),

                        "ocr_route":
                            summary.get(
                                "ocr_route",
                                "",
                            ),

                        "ocr_quality":
                            summary.get(
                                "ocr_quality",
                                "",
                            ),

                        "meaningful_regions":
                            int(
                                summary.get(
                                    "meaningful_regions",
                                    0,
                                )
                            ),

                        "words":
                            int(
                                summary.get(
                                    "words",
                                    0,
                                )
                            ),

                        "characters":
                            int(
                                summary.get(
                                    "characters",
                                    0,
                                )
                            ),

                        "mean_confidence":
                            float(
                                summary.get(
                                    "mean_confidence",
                                    0.0,
                                )
                            ),

                        "processing_seconds":
                            elapsed,

                        "output_directory":
                            str(
                                document_output
                            ),

                        "error":
                            "",
                    }

                except Exception as error:
                    elapsed = (
                        time.perf_counter()
                        - start_time
                    )

                    logger.exception(
                        (
                            "Batch document "
                            "processing failed: %s"
                        ),
                        image_path,
                    )

                    row = {
                        "filename":
                            image_path.name,

                        "status":
                            "FAILED",

                        "detector_route":
                            "",

                        "ocr_route":
                            "",

                        "ocr_quality":
                            "",

                        "meaningful_regions":
                            0,

                        "words":
                            0,

                        "characters":
                            0,

                        "mean_confidence":
                            0.0,

                        "processing_seconds":
                            elapsed,

                        "output_directory":
                            str(
                                document_output
                            ),

                        "error":
                            str(
                                error
                            ),
                    }

                rows.append(
                    row
                )

                total_elapsed = (
                    time.perf_counter()
                    - batch_start
                )

                self.root.after(
                    0,
                    lambda batch_row=row,
                    current=index,
                    count=total,
                    elapsed_time=total_elapsed:
                        self.update_batch_ui(
                            batch_row,
                            current,
                            count,
                            elapsed_time,
                        ),
                )

            self.batch_rows = rows

            total_elapsed = (
                time.perf_counter()
                - batch_start
            )

            self.save_batch_summary(
                output_dir,
                rows,
                total_elapsed,
            )

            self.root.after(
                0,
                lambda seconds=total_elapsed:
                    self.batch_complete(
                        seconds
                    ),
            )

        except Exception as error:
            error_message = str(
                error
            )

            logger.exception(
                "Batch processing failed."
            )

            self.root.after(
                0,
                lambda message=error_message:
                    self.batch_failed(
                        message
                    ),
            )


    def update_batch_ui(
        self,
        row: dict,
        current: int,
        total: int,
        total_elapsed: float,
    ) -> None:
        self.batch_tree.insert(
            "",
            tk.END,
            values=(
                row[
                    "filename"
                ],
                row[
                    "status"
                ],
                row[
                    "detector_route"
                ],
                row[
                    "ocr_route"
                ],
                row[
                    "ocr_quality"
                ],
                row[
                    "meaningful_regions"
                ],
                row[
                    "words"
                ],
                row[
                    "characters"
                ],
                (
                    f"{float(row['mean_confidence']):.4f}"
                ),
                (
                    f"{float(row['processing_seconds']):.2f}"
                ),
            ),
        )

        self.batch_progress.configure(
            value=current
        )

        self.batch_progress_var.set(
            f"{current} / {total}"
        )

        self.batch_time_var.set(
            (
                f"Elapsed: "
                f"{total_elapsed:.1f} s"
            )
        )

        self.batch_status_var.set(
            (
                "Processed: "
                f"{row['filename']}"
            )
        )


    def batch_complete(
        self,
        elapsed: float,
    ) -> None:
        self.batch_running = False

        self.batch_select_button.configure(
            state="normal"
        )

        self.batch_clear_button.configure(
            state="normal"
        )

        self.batch_process_button.configure(
            state="normal"
        )

        self.batch_export_button.configure(
            state="normal"
        )

        self.batch_open_button.configure(
            state="normal"
        )

        successful = sum(
            row[
                "status"
            ] == "SUCCESS"
            for row in self.batch_rows
        )

        strong = sum(
            row[
                "ocr_quality"
            ] == "STRONG"
            for row in self.batch_rows
        )

        limited = sum(
            row[
                "ocr_quality"
            ] == "LIMITED"
            for row in self.batch_rows
        )

        insufficient = sum(
            row[
                "ocr_quality"
            ] == "INSUFFICIENT"
            for row in self.batch_rows
        )

        failed = (
            len(
                self.batch_rows
            )
            - successful
        )

        self.batch_summary_var.set(
            (
                f"Successful: {successful} | "
                f"Strong: {strong} | "
                f"Limited: {limited} | "
                f"Insufficient: {insufficient} | "
                f"Failed: {failed}"
            )
        )

        self.batch_status_var.set(
            "Batch processing complete"
        )

        self.batch_time_var.set(
            f"Elapsed: {elapsed:.2f} s"
        )


    def batch_failed(
        self,
        error_message: str,
    ) -> None:
        self.batch_running = False

        self.batch_select_button.configure(
            state="normal"
        )

        self.batch_clear_button.configure(
            state="normal"
        )

        if self.batch_files:
            self.batch_process_button.configure(
                state="normal"
            )

        self.batch_status_var.set(
            "Batch processing failed"
        )

        messagebox.showerror(
            "Batch Processing Error",
            (
                f"{error_message}\n\n"
                f"Log:\n{LOG_FILE}"
            ),
        )


    # =========================================================================
    # BATCH SUMMARY
    # =========================================================================

    def save_batch_summary(
        self,
        output_dir: Path,
        rows: list[dict],
        total_seconds: float,
    ) -> None:
        csv_path = (
            output_dir
            / "batch_summary.csv"
        )

        fieldnames = [
            "filename",
            "status",
            "detector_route",
            "ocr_route",
            "ocr_quality",
            "meaningful_regions",
            "words",
            "characters",
            "mean_confidence",
            "processing_seconds",
            "output_directory",
            "error",
        ]

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                rows
            )

        successful = [
            row
            for row in rows
            if row[
                "status"
            ] == "SUCCESS"
        ]

        summary = {
            "app_version":
                APP_VERSION,

            "device":
                OCR_DEVICE,

            "shared_engine":
                True,

            "total_images":
                len(
                    rows
                ),

            "successful":
                len(
                    successful
                ),

            "failed":
                (
                    len(rows)
                    - len(successful)
                ),

            "strong":
                sum(
                    row[
                        "ocr_quality"
                    ] == "STRONG"
                    for row in successful
                ),

            "limited":
                sum(
                    row[
                        "ocr_quality"
                    ] == "LIMITED"
                    for row in successful
                ),

            "insufficient":
                sum(
                    row[
                        "ocr_quality"
                    ] == "INSUFFICIENT"
                    for row in successful
                ),

            "total_processing_seconds":
                total_seconds,

            "average_seconds_per_image":
                (
                    total_seconds
                    / len(rows)
                    if rows
                    else 0.0
                ),
        }

        with (
            output_dir
            / "batch_summary.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=4,
                ensure_ascii=False,
            )


    # =========================================================================
    # EXPORTS
    # =========================================================================

    def copy_text(
        self,
    ) -> None:
        text = self.recognized_text.get(
            "1.0",
            tk.END,
        ).strip()

        if not text:
            return

        self.root.clipboard_clear()

        self.root.clipboard_append(
            text
        )

        self.root.update()


    def export_scan(
        self,
    ) -> None:
        self.export_single_file(
            Path(
                "scan/scan.png"
            ),
            ".png",
            [
                (
                    "PNG image",
                    "*.png",
                ),
            ],
        )


    def export_text(
        self,
    ) -> None:
        self.export_single_file(
            Path(
                "ocr/text.txt"
            ),
            ".txt",
            [
                (
                    "Text file",
                    "*.txt",
                ),
            ],
        )


    def export_ocr_csv(
        self,
    ) -> None:
        self.export_single_file(
            Path(
                "ocr/results.csv"
            ),
            ".csv",
            [
                (
                    "CSV file",
                    "*.csv",
                ),
            ],
        )


    def export_single_file(
        self,
        relative_path: Path,
        extension: str,
        filetypes,
    ) -> None:
        if self.current_output_dir is None:
            return

        source = (
            self.current_output_dir
            / relative_path
        )

        if not source.exists():
            messagebox.showerror(
                "Export Error",
                (
                    "The requested output "
                    "file does not exist."
                ),
            )

            return

        destination = (
            filedialog.asksaveasfilename(
                defaultextension=extension,
                filetypes=filetypes,
            )
        )

        if not destination:
            return

        shutil.copy2(
            source,
            destination,
        )


    def export_batch_csv(
        self,
    ) -> None:
        if self.batch_output_dir is None:
            return

        source = (
            self.batch_output_dir
            / "batch_summary.csv"
        )

        if not source.exists():
            return

        destination = (
            filedialog.asksaveasfilename(
                title=(
                    "Export Batch Summary"
                ),
                defaultextension=".csv",
                filetypes=[
                    (
                        "CSV file",
                        "*.csv",
                    ),
                ],
            )
        )

        if not destination:
            return

        shutil.copy2(
            source,
            destination,
        )


    # =========================================================================
    # OPEN FILES / FOLDERS
    # =========================================================================

    def open_single_output(
        self,
    ) -> None:
        if (
            self.current_output_dir
            and self.current_output_dir.exists()
        ):
            os.startfile(
                self.current_output_dir
            )


    def open_batch_folder(
        self,
    ) -> None:
        if (
            self.batch_output_dir
            and self.batch_output_dir.exists()
        ):
            os.startfile(
                self.batch_output_dir
            )


    def open_log_file(
        self,
    ) -> None:
        if LOG_FILE.exists():
            os.startfile(
                LOG_FILE
            )


    def open_selected_batch_result(
        self,
        event=None,
    ) -> None:
        selection = (
            self.batch_tree.selection()
        )

        if not selection:
            return

        values = (
            self.batch_tree.item(
                selection[0]
            ).get(
                "values",
                [],
            )
        )

        if not values:
            return

        filename = str(
            values[0]
        )

        for row in self.batch_rows:
            if (
                row[
                    "filename"
                ] == filename
            ):
                output = Path(
                    row[
                        "output_directory"
                    ]
                )

                if output.exists():
                    os.startfile(
                        output
                    )

                return


    # =========================================================================
    # CLEAR
    # =========================================================================

    def reset_single_status(
        self,
    ) -> None:
        self.detector_var.set(
            "Detector: -"
        )

        self.route_var.set(
            "OCR route: -"
        )

        self.quality_var.set(
            "Quality: -"
        )

        self.stats_var.set(
            "OCR: -"
        )

        self.time_var.set(
            "Time: -"
        )


    def clear_single(
        self,
    ) -> None:
        if self.single_running:
            return

        self.input_path = None
        self.single_qa_json_path = None
        self.single_qa_button.configure(state="disabled")

        self.current_output_dir = None

        self.current_summary = None

        self.preview_photo = None

        self.perspective_photo = None

        self.scan_photo = None

        self.ocr_photo = None

        self.file_label.configure(
            text="No document selected"
        )

        self.status_var.set(
            "Ready"
        )

        self.reset_single_status()

        self.process_button.configure(
            state="disabled"
        )

        self.disable_single_exports()

        self.clear_single_results()


    def clear_single_results(
        self,
    ) -> None:
        for label in (
            self.original_image_label,
            self.perspective_image_label,
            self.scan_image_label,
            self.ocr_image_label,
        ):
            label.configure(
                image="",
                text="",
            )

        self.recognized_text.delete(
            "1.0",
            tk.END,
        )

        self.summary_text.delete(
            "1.0",
            tk.END,
        )


    def clear_batch(
        self,
    ) -> None:
        if self.batch_running:
            return

        self.batch_files = []

        self.batch_rows = []

        self.batch_output_dir = None

        self.batch_file_count_var.set(
            "No documents selected"
        )

        self.batch_status_var.set(
            "Ready"
        )

        self.batch_progress_var.set(
            "0 / 0"
        )

        self.batch_time_var.set(
            "Elapsed: 0.0 s"
        )

        self.batch_progress.configure(
            value=0
        )

        self.batch_summary_var.set(
            (
                "Successful: 0 | "
                "Strong: 0 | "
                "Limited: 0 | "
                "Insufficient: 0 | "
                "Failed: 0"
            )
        )

        self.batch_process_button.configure(
            state="disabled"
        )

        self.batch_export_button.configure(
            state="disabled"
        )

        self.batch_open_button.configure(
            state="disabled"
        )

        self.clear_batch_table()


    def clear_batch_table(
        self,
    ) -> None:
        for item in (
            self.batch_tree.get_children()
        ):
            self.batch_tree.delete(
                item
            )


    # =========================================================================
    # EXPORT BUTTON STATES
    # =========================================================================

    def enable_single_exports(
        self,
    ) -> None:
        for button in (
            self.export_scan_button,
            self.export_text_button,
            self.export_csv_button,
            self.copy_text_button,
            self.open_output_button,
        ):
            button.configure(
                state="normal"
            )


    def disable_single_exports(
        self,
    ) -> None:
        for button in (
            self.export_scan_button,
            self.export_text_button,
            self.export_csv_button,
            self.copy_text_button,
            self.open_output_button,
        ):
            button.configure(
                state="disabled"
            )


    # =========================================================================
    # IMAGE DISPLAY
    # =========================================================================

    def display_image(
        self,
        path: Path,
        label: ttk.Label,
        attribute_name: str,
    ) -> None:
        try:
            image = Image.open(
                path
            )

            self.root.update_idletasks()

            available_width = (
                label.winfo_width()
            )

            available_height = (
                label.winfo_height()
            )

            if available_width <= 1:
                available_width = (
                    IMAGE_PREVIEW_FALLBACK_WIDTH
                )

            if available_height <= 1:
                available_height = (
                    IMAGE_PREVIEW_FALLBACK_HEIGHT
                )

            max_width = max(
                IMAGE_PREVIEW_MIN_WIDTH,
                (
                    available_width
                    - IMAGE_PREVIEW_MARGIN
                ),
            )

            max_height = max(
                IMAGE_PREVIEW_MIN_HEIGHT,
                (
                    available_height
                    - IMAGE_PREVIEW_MARGIN
                ),
            )

            image.thumbnail(
                (
                    max_width,
                    max_height,
                ),
                Image.Resampling.LANCZOS,
            )

            photo = ImageTk.PhotoImage(
                image
            )

            setattr(
                self,
                attribute_name,
                photo,
            )

            label.configure(
                image=photo,
                text="",
            )

        except Exception as error:
            logger.exception(
                "Unable to display image: %s",
                path,
            )

            label.configure(
                image="",
                text=(
                    "Unable to display image:\n"
                    f"{error}"
                ),
            )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    ensure_project_directories()

    logger.info(
        "Launching GUI."
    )

    root = tk.Tk()

    DocumentUnderstandingApp(
        root
    )

    root.mainloop()

    logger.info(
        "GUI closed."
    )


if __name__ == "__main__":
    main()
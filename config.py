from __future__ import annotations

import logging
from pathlib import Path


# =============================================================================
# APPLICATION
# =============================================================================

APP_NAME = "Document Image Understanding and Enhancement System"
APP_VERSION = "1.0.0"

WINDOW_WIDTH = 1550
WINDOW_HEIGHT = 950
WINDOW_MIN_WIDTH = 1150
WINDOW_MIN_HEIGHT = 720


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_ROOT = PROJECT_ROOT / "results"
GUI_OUTPUT_ROOT = RESULTS_ROOT / "gui_v4"

SINGLE_OUTPUT_ROOT = GUI_OUTPUT_ROOT / "single"
BATCH_OUTPUT_ROOT = GUI_OUTPUT_ROOT / "batch"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"


# =============================================================================
# SUPPORTED INPUTS
# =============================================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# =============================================================================
# PRODUCTION OCR CONFIGURATION
# =============================================================================

# CPU is intentionally the production device.
#
# GPU PaddleOCR was experimentally tested on this machine but produced
# corrupted OCR output despite significantly faster inference.
OCR_DEVICE = "cpu"

# Must remain False on the validated Windows/PaddlePaddle 3.3.1 environment.
#
# Enabling oneDNN / MKL-DNN produced:
#
# ConvertPirAttribute2RuntimeAttribute not support
# pir::ArrayAttribute<pir::DoubleAttribute>
OCR_ENABLE_MKLDNN = False

OCR_LANGUAGE = "en"


# =============================================================================
# OCR FILTERING
# =============================================================================

MIN_CONFIDENCE = 0.25
MIN_ALPHANUMERIC_CHARACTERS = 2

# Allows legitimate single-character tokens such as table values 1-9.
SINGLE_CHARACTER_MIN_CONFIDENCE = 0.85


# =============================================================================
# OCR QUALITY GATE
# =============================================================================

STRONG_MIN_REGIONS = 5
STRONG_MIN_CHARACTERS = 20
STRONG_MIN_MEAN_CONFIDENCE = 0.50

LIMITED_MIN_REGIONS = 2
LIMITED_MIN_CHARACTERS = 10


# =============================================================================
# OCR RECOVERY
# =============================================================================

RECOVERY_SCALE = 4
RECOVERY_UNSHARP_AMOUNT = 1.2
RECOVERY_UNSHARP_SIGMA = 1.0


# =============================================================================
# GUI IMAGE PREVIEW
# =============================================================================

IMAGE_PREVIEW_MARGIN = 30

IMAGE_PREVIEW_FALLBACK_WIDTH = 1100
IMAGE_PREVIEW_FALLBACK_HEIGHT = 600

IMAGE_PREVIEW_MIN_WIDTH = 200
IMAGE_PREVIEW_MIN_HEIGHT = 200


# =============================================================================
# LOGGING
# =============================================================================

def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(
        "document_image_understanding"
    )

    logger.setLevel(
        logging.INFO
    )

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        (
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        )
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )

    file_handler.setLevel(
        logging.INFO
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        file_handler
    )

    return logger


# =============================================================================
# DIRECTORY INITIALIZATION
# =============================================================================

def ensure_project_directories() -> None:
    for path in (
        RESULTS_ROOT,
        GUI_OUTPUT_ROOT,
        SINGLE_OUTPUT_ROOT,
        BATCH_OUTPUT_ROOT,
        LOG_DIR,
    ):
        path.mkdir(
            parents=True,
            exist_ok=True,
        )
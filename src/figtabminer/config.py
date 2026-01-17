import os
import json
from typing import Optional
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
CONFIG_PATH = BASE_DIR / "config" / "figtabminer.json"


def _load_user_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_USER_CONFIG = _load_user_config()


def _cfg_value(key: str, default=None, env: Optional[str] = None):
    if key in _USER_CONFIG:
        return _USER_CONFIG[key]
    if env:
        env_val = os.getenv(env)
        if env_val is not None:
            return env_val
    return default


def _cfg_str(key: str, default: str, env: Optional[str] = None) -> str:
    val = _cfg_value(key, default=default, env=env)
    if isinstance(val, str):
        return val.strip()
    return str(val)


def _cfg_bool(key: str, default: bool, env: Optional[str] = None) -> bool:
    val = _cfg_value(key, default=default, env=env)
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
    return default


def _cfg_int(key: str, default: int, env: Optional[str] = None) -> int:
    val = _cfg_value(key, default=default, env=env)
    try:
        return int(val)
    except Exception:
        return default


def _cfg_float(key: str, default: float, env: Optional[str] = None) -> float:
    val = _cfg_value(key, default=default, env=env)
    try:
        return float(val)
    except Exception:
        return default

# Rendering
RENDER_ZOOM = 2.0  # Zoom factor for PDF rendering (2.0 = 144 DPI approx)
PREVIEW_MAX_SIZE = (800, 800)

# OCR
# FIGTABMINER_OCR_GPU: "auto" | "true"/"false"
OCR_GPU = _cfg_str("ocr_gpu", default="auto", env="FIGTABMINER_OCR_GPU").lower()

# Layout detection (optional)
# FIGTABMINER_LAYOUT: "auto" | "true"/"false"
LAYOUT_ENABLE = _cfg_str("layout_enable", default="auto", env="FIGTABMINER_LAYOUT").lower()
LAYOUT_MODEL_CONFIG = _cfg_str(
    "layout_model",
    default="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    env="FIGTABMINER_LAYOUT_MODEL",
)
LAYOUT_SCORE_THRESH = _cfg_float("layout_score", default=0.5, env="FIGTABMINER_LAYOUT_SCORE")
LAYOUT_WEIGHTS_PATH = _cfg_str("layout_weights_path", default="", env="FIGTABMINER_LAYOUT_WEIGHTS")
LAYOUT_LABEL_MAP = {
    0: "Text",
    1: "Title",
    2: "List",
    3: "Table",
    4: "Figure",
}

# Figure extraction
MIN_FIGURE_AREA = _cfg_int("figure_min_area", default=2000)
MIN_FIGURE_DIM = _cfg_int("figure_min_dim", default=40)
MAX_FIGURE_PAGE_RATIO = _cfg_float("figure_max_page_ratio", default=0.85)
FIGURE_MERGE_IOU = _cfg_float("figure_merge_iou", default=0.2)
FIGURE_CROP_PAD = _cfg_int("figure_crop_pad", default=4)
MIN_FIGURE_INK_RATIO = _cfg_float("figure_min_ink_ratio", default=0.01)
MIN_TEXT_LINES_FOR_PAGE_SKIP = _cfg_int("figure_min_text_lines_skip", default=6)

# Table extraction
MIN_TABLE_AREA = _cfg_int("table_min_area", default=2000)
MIN_TABLE_DIM = _cfg_int("table_min_dim", default=40)
TABLE_MERGE_IOU = _cfg_float("table_merge_iou", default=0.2)
TABLE_CROP_PAD = _cfg_int("table_crop_pad", default=2)

# Table extraction settings for pdfplumber
TABLE_SETTINGS_LINE = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "intersection_tolerance": 5,
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 6,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
    "text_tolerance": 3,
    "text_x_tolerance": 2,
}
TABLE_SETTINGS_TEXT = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "intersection_tolerance": 5,
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "edge_min_length": 6,
    "min_words_vertical": 3,
    "min_words_horizontal": 1,
    "text_tolerance": 3,
    "text_x_tolerance": 2,
}

# Keywords for heuristic detection
CAPTION_KEYWORDS = [
    "Figure", "Fig.", "Table", "Tab.", "Scheme", "Chart", 
    "图", "表"
]

# AI / Science Keywords
SUBTYPE_KEYWORDS = {
    "microscopy": ["SEM", "TEM", "AFM", "Microscope", "Micrograph", "Scale bar"],
    "spectrum": ["XRD", "FTIR", "Raman", "UV-Vis", "Spectrum", "Spectra", "Absorbance", "Transmittance", "Intensity"],
    "line_plot": ["vs", "dependence", "function of", "plot", "curve"],
}

# Regex Patterns for Condition Extraction
CONDITION_PATTERNS = {
    "temperature": r"(\d+(\.\d+)?\s*(K|°C|deg\s*C))",
    "wavelength": r"(\d+(\.\d+)?\s*(nm|μm))",
    "wavenumber": r"(\d+(\.\d+)?\s*cm\^-1)",
    "pressure": r"(\d+(\.\d+)?\s*(Pa|kPa|MPa|bar|Torr))",
    "time": r"(\d+(\.\d+)?\s*(s|min|h|hours|seconds))",
    "ph": r"(pH\s*=?\s*\d+(\.\d+)?)",
}

MATERIAL_PATTERNS = [
    r"\b[A-Z][a-z]?\d*([A-Z][a-z]?\d*)*\b",  # Simple chemical formula heuristic
]

# Fallback / Thresholds
CAPTION_SEARCH_WINDOW = _cfg_int("caption_search_window", default=300)  # pixels vertical distance to search for caption
CAPTION_DIRECTION_PENALTY = _cfg_int("caption_direction_penalty", default=120)
CAPTION_CONTINUATION_GAP = _cfg_int("caption_continuation_gap", default=12)

# BBox Merger Configuration
BBOX_MERGER_CONFIG = _cfg_value("bbox_merger", default={})
if not isinstance(BBOX_MERGER_CONFIG, dict):
    BBOX_MERGER_CONFIG = {}

# Merger defaults
BBOX_MERGER_ENABLE_SEMANTIC = BBOX_MERGER_CONFIG.get("enable_semantic_merge", True)
BBOX_MERGER_ENABLE_VISUAL = BBOX_MERGER_CONFIG.get("enable_visual_merge", True)
BBOX_MERGER_ENABLE_NOISE_FILTER = BBOX_MERGER_CONFIG.get("enable_noise_filter", True)
BBOX_MERGER_OVERLAP_THRESHOLD = BBOX_MERGER_CONFIG.get("overlap_threshold", 0.7)
BBOX_MERGER_DISTANCE_THRESHOLD = BBOX_MERGER_CONFIG.get("distance_threshold", 50)
BBOX_MERGER_ARROW_ASPECT_MIN = BBOX_MERGER_CONFIG.get("arrow_aspect_ratio_min", 5.0)
BBOX_MERGER_ARROW_ASPECT_MAX = BBOX_MERGER_CONFIG.get("arrow_aspect_ratio_max", 0.2)
BBOX_MERGER_ARROW_INK_MAX = BBOX_MERGER_CONFIG.get("arrow_ink_ratio_max", 0.05)
BBOX_MERGER_CONNECTION_THRESHOLD = BBOX_MERGER_CONFIG.get("connection_detection_threshold", 0.3)

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
FIGURE_TEXT_FILTER_ENABLE = _cfg_bool("figure_text_filter_enable", default=True)
FIGURE_TEXT_COVERAGE_THRESHOLD = _cfg_float("figure_text_coverage_threshold", default=0.35)
FIGURE_TEXT_LINE_MIN_WIDTH_RATIO = _cfg_float("figure_text_line_min_width_ratio", default=0.6)
FIGURE_TEXT_BAND_MIN_LINES = _cfg_int("figure_text_band_min_lines", default=2)
FIGURE_TEXT_BAND_MAX_HEIGHT_RATIO = _cfg_float("figure_text_band_max_height_ratio", default=0.25)
FIGURE_TEXT_BAND_CENTER_MIN = _cfg_float("figure_text_band_center_min", default=0.3)
FIGURE_TEXT_BAND_CENTER_MAX = _cfg_float("figure_text_band_center_max", default=0.7)
FIGURE_TEXT_BARRIER_MIN_OVERLAP = _cfg_float("figure_text_barrier_min_overlap", default=0.6)
FIGURE_PROXIMITY_MERGE_GAP = _cfg_int("figure_proximity_merge_gap", default=16)
FIGURE_PROXIMITY_ALIGNMENT = _cfg_float("figure_proximity_alignment", default=0.65)
FIGURE_REFINE_BBOX_ENABLE = _cfg_bool("figure_refine_bbox_enable", default=True)
FIGURE_SPLIT_TEXT_BAND_ENABLE = _cfg_bool("figure_split_text_band_enable", default=True)

# Table extraction
MIN_TABLE_AREA = _cfg_int("table_min_area", default=2000)
MIN_TABLE_DIM = _cfg_int("table_min_dim", default=40)
TABLE_MERGE_IOU = _cfg_float("table_merge_iou", default=0.2)
TABLE_CROP_PAD = _cfg_int("table_crop_pad", default=2)
TABLE_TEXT_REFINE_ENABLE = _cfg_bool("table_text_refine_enable", default=True)
TABLE_TEXT_REFINE_MIN_LINES = _cfg_int("table_text_refine_min_lines", default=2)
TABLE_TEXT_REFINE_MIN_WIDTH_RATIO = _cfg_float("table_text_refine_min_width_ratio", default=0.5)
TABLE_TEXT_REFINE_PADDING = _cfg_int("table_text_refine_padding", default=8)
TABLE_TEXT_REFINE_MIN_AREA_RATIO = _cfg_float("table_text_refine_min_area_ratio", default=0.25)
TABLE_THREE_LINE_DETECT_ENABLE = _cfg_bool("table_three_line_detect_enable", default=True)
TABLE_THREE_LINE_MIN_LINE_LENGTH_RATIO = _cfg_float("table_three_line_min_line_length_ratio", default=0.3)
TABLE_THREE_LINE_MIN_LINES = _cfg_int("table_three_line_min_lines", default=2)
TABLE_THREE_LINE_MAX_LINES = _cfg_int("table_three_line_max_lines", default=8)

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

# v1.3: Chart Classification Configuration
CHART_CLASSIFICATION_CONFIG = _cfg_value("chart_classification", default={})
if not isinstance(CHART_CLASSIFICATION_CONFIG, dict):
    CHART_CLASSIFICATION_CONFIG = {}

CHART_CLASSIFICATION_USE_ENHANCED = CHART_CLASSIFICATION_CONFIG.get("use_enhanced_classifier", True)
CHART_CLASSIFICATION_VISUAL = CHART_CLASSIFICATION_CONFIG.get("enable_visual_analysis", True)
CHART_CLASSIFICATION_OCR = CHART_CLASSIFICATION_CONFIG.get("enable_ocr_assist", False)
CHART_CLASSIFICATION_VISUAL_WEIGHT = CHART_CLASSIFICATION_CONFIG.get("visual_weight", 0.6)
CHART_CLASSIFICATION_KEYWORD_WEIGHT = CHART_CLASSIFICATION_CONFIG.get("keyword_weight", 0.4)

# v1.3: Bar Chart Extraction Configuration
BAR_CHART_CONFIG = _cfg_value("bar_chart_extraction", default={})
if not isinstance(BAR_CHART_CONFIG, dict):
    BAR_CHART_CONFIG = {}

BAR_CHART_AUTO_DIGITIZE = BAR_CHART_CONFIG.get("enable_auto_digitize", True)
BAR_CHART_MIN_WIDTH = BAR_CHART_CONFIG.get("min_bar_width", 5)
BAR_CHART_MIN_HEIGHT = BAR_CHART_CONFIG.get("min_bar_height", 10)
BAR_CHART_AXIS_THRESHOLD = BAR_CHART_CONFIG.get("axis_detection_threshold", 0.5)
BAR_CHART_OCR_LABELS = BAR_CHART_CONFIG.get("enable_ocr_labels", False)

# Text False Positive Filter Configuration (Phase 2: Critical Accuracy Fixes)
# 更严格的默认值以减少误报
TEXT_FILTER_CONFIDENCE_THRESHOLD = _cfg_float("text_filter_confidence_threshold", default=0.75)  # 提高到 0.75
TEXT_FILTER_ENABLE_TRANSFORMER = _cfg_bool("text_filter_enable_transformer", default=False)
TEXT_FILTER_TEXT_DENSITY_THRESHOLD = _cfg_float("text_filter_text_density_threshold", default=0.05)  # 降低到 5%
TEXT_FILTER_MIN_STRUCTURE_SCORE = _cfg_float("text_filter_min_structure_score", default=300)  # 提高到 300
TEXT_FILTER_ENABLE_POSITION_HEURISTICS = _cfg_bool("text_filter_enable_position_heuristics", default=True)
TEXT_FILTER_ENABLE_OCR_PATTERN = _cfg_bool("text_filter_enable_ocr_pattern", default=True)
TEXT_FILTER_ENABLE_TEXT_LINE_DETECTION = _cfg_bool("text_filter_enable_text_line_detection", default=True)

# arXiv Filter Configuration (Phase 1: Critical Accuracy Fixes)
ARXIV_FILTER_ENABLE_OCR = _cfg_bool("arxiv_filter_enable_ocr", default=True)
ARXIV_FILTER_POSITION_THRESHOLD = _cfg_float("arxiv_filter_position_threshold", default=0.1)
ARXIV_FILTER_AREA_THRESHOLD = _cfg_float("arxiv_filter_area_threshold", default=0.05)
ARXIV_FILTER_ASPECT_RATIO_MIN = _cfg_float("arxiv_filter_aspect_ratio_min", default=1.5)
ARXIV_FILTER_ASPECT_RATIO_MAX = _cfg_float("arxiv_filter_aspect_ratio_max", default=8.0)
ARXIV_FILTER_CHECK_LEFT_MARGIN = _cfg_bool("arxiv_filter_check_left_margin", default=True)
ARXIV_FILTER_LEFT_MARGIN_THRESHOLD = _cfg_float("arxiv_filter_left_margin_threshold", default=0.15)
ARXIV_FILTER_CHECK_ROTATION = _cfg_bool("arxiv_filter_check_rotation", default=True)

# Table Enhancer Configuration (Phase 3: Critical Accuracy Fixes)
TABLE_ENHANCER_ENABLE_IMG2TABLE = _cfg_bool("table_enhancer_enable_img2table", default=True)
TABLE_ENHANCER_IOU_THRESHOLD = _cfg_float("table_enhancer_iou_threshold", default=0.3)
TABLE_ENHANCER_MIN_CONFIDENCE = _cfg_float("table_enhancer_min_confidence", default=0.5)
TABLE_ENHANCER_SHRINK_BBOX = _cfg_bool("table_enhancer_shrink_bbox", default=True)
TABLE_ENHANCER_SHRINK_RATIO = _cfg_float("table_enhancer_shrink_ratio", default=0.05)

# Table Extraction Configuration
TABLE_EXTRACTION_CONFIG = _cfg_value("table_extraction", default={})
if not isinstance(TABLE_EXTRACTION_CONFIG, dict):
    TABLE_EXTRACTION_CONFIG = {}

TABLE_EXTRACTION_USE_ENHANCED = TABLE_EXTRACTION_CONFIG.get("use_enhanced_extractor", True)
TABLE_EXTRACTION_ENABLE_TABLE_TRANSFORMER = TABLE_EXTRACTION_CONFIG.get("enable_table_transformer", False)
TABLE_EXTRACTION_TABLE_TRANSFORMER_CONFIDENCE = TABLE_EXTRACTION_CONFIG.get("table_transformer_confidence", 0.85)
TABLE_EXTRACTION_ENABLE_VISUAL_DETECTION = TABLE_EXTRACTION_CONFIG.get("enable_visual_detection", True)

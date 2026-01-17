import os
import json
import hashlib
import logging
import importlib
from pathlib import Path
from typing import Any, List, Optional, Union

# Logging Setup
def setup_logging(name: str = "FigTabMiner"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = setup_logging()

def get_doc_id(pdf_bytes: bytes) -> str:
    """Generate first 12 chars of SHA256 hash of PDF content."""
    return hashlib.sha256(pdf_bytes).hexdigest()[:12]

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_json(data: Any, path: Union[str, Path], indent: int = 2):
    """Write dictionary to JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def read_json(path: Union[str, Path]) -> Any:
    """Read JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def safe_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False
    except Exception as e:
        logger.warning(f"Error importing {module_name}: {e}")
        return False

def clamp_bbox(bbox: list, max_w: float, max_h: float) -> list:
    """Clamp bbox coordinates to image boundaries."""
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(x0, max_w)),
        max(0, min(y0, max_h)),
        max(0, min(x1, max_w)),
        max(0, min(y1, max_h))
    ]

def bbox_distance(bbox1: list, bbox2: list) -> float:
    """
    Calculate vertical distance between two bboxes.
    Positive if bbox2 is below bbox1.
    bbox format: [x0, y0, x1, y1]
    """
    # bbox1 y_bottom vs bbox2 y_top
    return bbox2[1] - bbox1[3]

def rect_to_bbox(rect) -> list:
    """Convert PyMuPDF rect to [x0, y0, x1, y1] list."""
    return [rect.x0, rect.y0, rect.x1, rect.y1]

def bbox_area(bbox: list) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)

def bbox_iou(b1: list, b2: list) -> float:
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    area1 = bbox_area(b1)
    area2 = bbox_area(b2)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def bbox_overlap_ratio(b1: list, b2: list) -> float:
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    return inter / min(bbox_area(b1), bbox_area(b2))

def merge_bboxes(b1: list, b2: list) -> list:
    return [
        min(b1[0], b2[0]),
        min(b1[1], b2[1]),
        max(b1[2], b2[2]),
        max(b1[3], b2[3]),
    ]

def expand_bbox(bbox: list, pad: int, max_w: Optional[int] = None, max_h: Optional[int] = None) -> list:
    x0, y0, x1, y1 = bbox
    x0 = x0 - pad
    y0 = y0 - pad
    x1 = x1 + pad
    y1 = y1 + pad
    if max_w is not None:
        x0 = max(0, x0)
        x1 = min(max_w, x1)
    if max_h is not None:
        y0 = max(0, y0)
        y1 = min(max_h, y1)
    return [x0, y0, x1, y1]

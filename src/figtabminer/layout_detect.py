from typing import Dict, List

from . import config
from . import utils

logger = utils.setup_logging(__name__)

_MODEL = None
_CACHE: Dict[str, List[dict]] = {}


def layout_available() -> bool:
    if config.LAYOUT_ENABLE in ("0", "false", "no", "n", "off"):
        return False
    if not utils.safe_import("layoutparser"):
        return False
    if not utils.safe_import("detectron2"):
        return False
    return True


def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        import layoutparser as lp
        _MODEL = lp.Detectron2LayoutModel(
            config.LAYOUT_MODEL_CONFIG,
            extra_config=[
                "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                config.LAYOUT_SCORE_THRESH,
            ],
            label_map=config.LAYOUT_LABEL_MAP,
        )
    except Exception as e:
        logger.error(f"Layout model init failed: {e}")
        _MODEL = None
    return _MODEL


def _block_to_bbox(block) -> List[float]:
    coords = None
    if hasattr(block, "coordinates"):
        coords = block.coordinates
    elif hasattr(block, "block") and hasattr(block.block, "coordinates"):
        coords = block.block.coordinates
    if coords is None:
        coords = (block.x_1, block.y_1, block.x_2, block.y_2)
    return [float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])]


def detect_layout(page_img_path: str) -> List[dict]:
    if page_img_path in _CACHE:
        return _CACHE[page_img_path]
    results: List[dict] = []
    if not layout_available():
        _CACHE[page_img_path] = results
        return results
    model = _get_model()
    if model is None:
        _CACHE[page_img_path] = results
        return results
    try:
        import layoutparser as lp
        image = lp.io.read_image(page_img_path)
        layout = model.detect(image)
        for block in layout:
            block_type = str(block.type).lower()
            score = float(getattr(block, "score", 1.0))
            if score < config.LAYOUT_SCORE_THRESH:
                continue
            if block_type not in ("figure", "table"):
                continue
            results.append({
                "type": block_type,
                "bbox": _block_to_bbox(block),
                "score": score,
            })
    except Exception as e:
        logger.warning(f"Layout detection failed on {page_img_path}: {e}")
    _CACHE[page_img_path] = results
    return results

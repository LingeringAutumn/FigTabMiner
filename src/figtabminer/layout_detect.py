import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from . import config
from . import utils
from .text_false_positive_filter import TextFalsePositiveFilter
from .arxiv_filter import ArxivFilter
from .table_enhancer import TableEnhancer
from .models import Detection

logger = utils.setup_logging(__name__)

# Try to import DocLayout-YOLO detector
try:
    from .detectors import doclayout_detector
    DOCLAYOUT_AVAILABLE = doclayout_detector.is_available()
except ImportError:
    DOCLAYOUT_AVAILABLE = False
    logger.debug("DocLayout-YOLO detector not available")

_MODEL = None
_MODEL_FAILED = False
_DOCLAYOUT_DETECTOR = None
_DOCLAYOUT_FAILED = False
_CACHE: Dict[str, List[dict]] = {}


def get_layout_status() -> dict:
    """
    Get the current status of layout detection capability.
    
    Returns:
        Dictionary with status information including detector types
    """
    status = {
        "available": layout_available(),
        "doclayout_available": DOCLAYOUT_AVAILABLE,
        "doclayout_loaded": _DOCLAYOUT_DETECTOR is not None,
        "doclayout_failed": _DOCLAYOUT_FAILED,
        "publaynet_available": _publaynet_available(),
        "publaynet_loaded": _MODEL is not None,
        "publaynet_failed": _MODEL_FAILED,
        "cache_size": len(_CACHE),
    }
    
    # Determine primary detector
    if _DOCLAYOUT_DETECTOR is not None:
        status["primary_detector"] = "doclayout_yolo"
        status["status"] = "ready"
    elif _MODEL is not None:
        status["primary_detector"] = "publaynet"
        status["status"] = "ready"
    elif status["available"]:
        status["primary_detector"] = "none"
        status["status"] = "not_initialized"
    else:
        status["primary_detector"] = "none"
        status["status"] = "unavailable"
    
    return status


def layout_available() -> bool:
    """Check if any layout detection method is available"""
    if config.LAYOUT_ENABLE in ("0", "false", "no", "n", "off"):
        return False
    
    # DocLayout-YOLO is available
    if DOCLAYOUT_AVAILABLE:
        return True
    
    # PubLayNet is available
    if _publaynet_available():
        return True
    
    return False


def _publaynet_available() -> bool:
    """Check if PubLayNet (layoutparser + detectron2) is available"""
    if not utils.safe_import("layoutparser"):
        return False
    if not utils.safe_import("detectron2"):
        return False
    return True


def _get_doclayout_detector():
    """
    Initialize and return the DocLayout-YOLO detector.
    
    Returns:
        Initialized detector or None if initialization fails
    """
    global _DOCLAYOUT_DETECTOR
    global _DOCLAYOUT_FAILED
    
    if _DOCLAYOUT_DETECTOR is not None:
        return _DOCLAYOUT_DETECTOR
    
    if _DOCLAYOUT_FAILED:
        logger.debug("DocLayout-YOLO initialization previously failed, skipping")
        return None
    
    if not DOCLAYOUT_AVAILABLE:
        logger.debug("DocLayout-YOLO not available")
        _DOCLAYOUT_FAILED = True
        return None
    
    try:
        logger.info("Initializing DocLayout-YOLO detector...")
        from .detectors.doclayout_detector import DocLayoutYOLODetector
        _DOCLAYOUT_DETECTOR = DocLayoutYOLODetector()
        logger.info("DocLayout-YOLO detector initialized successfully")
        return _DOCLAYOUT_DETECTOR
    except Exception as e:
        logger.warning(f"DocLayout-YOLO initialization failed: {e}")
        logger.debug("Traceback:", exc_info=True)
        _DOCLAYOUT_FAILED = True
        return None


def _patch_layoutparser_config():
    """
    Patch layoutparser's model config to remove ?dl=1 from URLs.
    This is a workaround for the Dropbox URL issue.
    """
    try:
        import layoutparser as lp
        from layoutparser.models.detectron2 import catalog
        
        # Check if catalog has the model config
        if hasattr(catalog, 'PathManager'):
            # Monkey patch the PathManager to handle ?dl=1
            original_get_local_path = catalog.PathManager.get_local_path
            
            def patched_get_local_path(path, **kwargs):
                """Patched version that removes ?dl=1 from paths"""
                result = original_get_local_path(path, **kwargs)
                
                # If result has ?dl=1, try to find or create a clean version
                if result and '?dl=1' in result:
                    logger.debug(f"Detected ?dl=1 in path: {result}")
                    clean_path = result.replace('?dl=1', '')
                    
                    # If clean path doesn't exist, copy from the ?dl=1 version
                    if not os.path.exists(clean_path) and os.path.exists(result):
                        try:
                            logger.info(f"Creating clean copy: {clean_path}")
                            shutil.copyfile(result, clean_path)
                            return clean_path
                        except Exception as e:
                            logger.warning(f"Could not create clean copy: {e}")
                    elif os.path.exists(clean_path):
                        logger.debug(f"Using existing clean path: {clean_path}")
                        return clean_path
                
                return result
            
            catalog.PathManager.get_local_path = patched_get_local_path
            logger.info("Successfully patched layoutparser PathManager")
            return True
            
    except Exception as e:
        logger.debug(f"Could not patch layoutparser: {e}")
    
    return False


def _get_model():
    """
    Initialize and return the layout detection model.
    Implements robust error handling and fallback mechanisms.
    
    Returns:
        Initialized model or None if all attempts fail
    """
    global _MODEL
    global _MODEL_FAILED
    
    if _MODEL is not None:
        return _MODEL
    
    if _MODEL_FAILED:
        logger.debug("Model initialization previously failed, skipping")
        return None
    
    # Apply patch to fix ?dl=1 issue
    _patch_layoutparser_config()
    
    logger.info("Initializing layout detection model...")
    weights_path = _resolve_cached_weights()
    
    if weights_path:
        logger.info(f"Using weights from: {weights_path}")
    else:
        logger.info("No cached weights found, will download from model config")
    
    try:
        import layoutparser as lp
        extra_config = [
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
            config.LAYOUT_SCORE_THRESH,
        ]
        if weights_path:
            extra_config += ["MODEL.WEIGHTS", weights_path]
        
        logger.debug(f"Model config: {config.LAYOUT_MODEL_CONFIG}")
        logger.debug(f"Extra config: {extra_config}")
        
        _MODEL = lp.Detectron2LayoutModel(
            config.LAYOUT_MODEL_CONFIG,
            extra_config=extra_config,
            label_map=config.LAYOUT_LABEL_MAP,
        )
        logger.info("Layout model initialized successfully")
        return _MODEL
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Layout model initialization failed: {error_msg}")
        
        # Handle specific error: Unsupported query remaining (the ?dl=1 issue)
        if "Unsupported query remaining" in error_msg:
            logger.info("Detected ?dl=1 suffix issue, attempting to fix...")
            
            # Try to normalize weights again
            weights_path = _normalize_cached_weights()
            
            # If still not found, try to extract path from error message
            if not weights_path:
                weights_path = _weights_from_error(error_msg)
                if weights_path:
                    logger.info(f"Extracted weights path from error: {weights_path}")
                    weights_path = _copy_weights_to_clean(weights_path)
            
            if weights_path:
                logger.info(f"Retrying with normalized weights: {weights_path}")
                try:
                    import layoutparser as lp
                    extra_config = [
                        "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                        config.LAYOUT_SCORE_THRESH,
                        "MODEL.WEIGHTS",
                        weights_path,
                    ]
                    _MODEL = lp.Detectron2LayoutModel(
                        config.LAYOUT_MODEL_CONFIG,
                        extra_config=extra_config,
                        label_map=config.LAYOUT_LABEL_MAP,
                    )
                    logger.info("Layout model initialized successfully after fix")
                    return _MODEL
                except Exception as e2:
                    logger.error(f"Retry also failed: {e2}")
                    _MODEL = None
                    _MODEL_FAILED = True
                    return None
            else:
                logger.error("Could not find or fix weights file")
        
        # Handle specific error: Checkpoint not found
        elif "Checkpoint" in error_msg and "not found" in error_msg:
            logger.info("Detected missing checkpoint, attempting recovery...")
            weights_path = _recover_missing_checkpoint(error_msg)
            
            if weights_path:
                logger.info(f"Retrying with recovered checkpoint: {weights_path}")
                try:
                    import layoutparser as lp
                    extra_config = [
                        "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                        config.LAYOUT_SCORE_THRESH,
                        "MODEL.WEIGHTS",
                        weights_path,
                    ]
                    _MODEL = lp.Detectron2LayoutModel(
                        config.LAYOUT_MODEL_CONFIG,
                        extra_config=extra_config,
                        label_map=config.LAYOUT_LABEL_MAP,
                    )
                    logger.info("Layout model initialized successfully after recovery")
                    return _MODEL
                except Exception as e2:
                    logger.error(f"Recovery attempt failed: {e2}")
                    _MODEL = None
                    _MODEL_FAILED = True
                    return None
            else:
                logger.error("Could not recover checkpoint")
        
        # For any other error, mark as failed
        logger.error("Layout model initialization failed permanently")
        logger.info("System will fall back to basic image block detection")
        _MODEL = None
        _MODEL_FAILED = True
        
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


def _resolve_cached_weights() -> Optional[str]:
    """
    Resolve the path to cached model weights.
    
    Returns:
        Path to weights file, or None to let model download
    """
    # If user specified a path, use it
    if config.LAYOUT_WEIGHTS_PATH:
        logger.info(f"Using user-specified weights: {config.LAYOUT_WEIGHTS_PATH}")
        return config.LAYOUT_WEIGHTS_PATH
    
    # Try to find and normalize cached weights
    normalized = _normalize_cached_weights()
    
    # If we found a clean file, use it
    if normalized and not normalized.endswith("?dl=1"):
        return normalized
    
    # If we only found the problematic file, try to fix it
    if normalized and normalized.endswith("?dl=1"):
        logger.warning(f"Found problematic weights path: {normalized}")
        # Try to use the file without the suffix
        clean_path = normalized.replace("?dl=1", "")
        if os.path.exists(clean_path):
            logger.info(f"Using clean path: {clean_path}")
            return clean_path
        else:
            # Copy to clean location
            logger.info("Copying to clean location...")
            return _copy_weights_to_clean(normalized)
    
    return None


def _normalize_cached_weights() -> Optional[str]:
    """
    Find and normalize cached model weights, handling the ?dl=1 suffix issue.
    
    Returns:
        Path to a valid model_final.pth file, or None if not found
    """
    cache_root = Path(os.path.expanduser("~")) / ".torch" / "iopath_cache"
    if not cache_root.exists():
        logger.debug(f"Cache root does not exist: {cache_root}")
        return None
    
    # First, try to find a clean model_final.pth file
    candidates = list(cache_root.rglob("model_final.pth"))
    if candidates:
        logger.info(f"Found clean model weights at: {candidates[0]}")
        return str(candidates[0])
    
    # If not found, look for files with ?dl=1 suffix and normalize them
    logger.info("Looking for model weights with ?dl=1 suffix...")
    for cand in cache_root.rglob("*"):
        # Check for various problematic suffixes
        if cand.name not in ("model_final.pth?dl=1", "model_final.pth?dl=1.lock"):
            continue
        
        if cand.name == "model_final.pth?dl=1.lock":
            # Skip lock files, but look for the actual file
            actual_file = cand.with_name("model_final.pth?dl=1")
            if actual_file.exists():
                cand = actual_file
            else:
                continue
        
        logger.info(f"Found problematic weight file: {cand}")
        target = cand.with_name("model_final.pth")
        
        if not target.exists():
            try:
                # Try to rename in place
                logger.info(f"Attempting to rename {cand.name} to model_final.pth")
                cand.rename(target)
                
                # Clean up lock file if exists
                lock_path = cand.with_name("model_final.pth?dl=1.lock")
                if lock_path.exists():
                    try:
                        lock_path.unlink()
                        logger.debug("Removed lock file")
                    except Exception as e:
                        logger.debug(f"Could not remove lock file: {e}")
                
                logger.info(f"Successfully renamed to: {target}")
                return str(target)
                
            except Exception as e:
                # If rename fails (e.g., lock/file in use), copy to a clean path
                logger.warning(f"Rename failed: {e}. Copying to clean location...")
                clean_dir = cache_root / "figtabminer"
                clean_dir.mkdir(parents=True, exist_ok=True)
                clean_target = clean_dir / "model_final.pth"
                
                try:
                    shutil.copyfile(cand, clean_target)
                    logger.info(f"Successfully copied to: {clean_target}")
                    return str(clean_target)
                except Exception as e2:
                    logger.error(f"Copy also failed: {e2}")
                    continue
        else:
            logger.info(f"Target already exists: {target}")
            return str(target)
    
    logger.warning("No model weights found in cache")
    return None


def _weights_from_error(err: str) -> Optional[str]:
    """
    Extract the problematic weights path from error message.
    
    Args:
        err: Error message string
        
    Returns:
        Path to the weights file (with ?dl=1 suffix) if found
    """
    # Example: "... orginal filename: /path/to/model_final.pth?dl=1"
    match = re.search(r"filename:\s*([^\s]+)", err)
    if not match:
        return None
    path = match.group(1).strip()
    logger.debug(f"Extracted path from error: {path}")
    return path if os.path.exists(path) else None


def _copy_weights_to_clean(src_path: str) -> Optional[str]:
    """
    Copy weights file to a clean path without query parameters.
    
    Args:
        src_path: Source path (may contain ?dl=1)
        
    Returns:
        Clean path to the copied weights file
    """
    cache_root = Path(os.path.expanduser("~")) / ".torch" / "iopath_cache"
    clean_dir = cache_root / "figtabminer_clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_target = clean_dir / "model_final.pth"
    
    try:
        logger.info(f"Copying weights from {src_path} to {clean_target}")
        shutil.copyfile(src_path, clean_target)
        logger.info(f"Successfully copied to clean path")
        return str(clean_target)
    except Exception as e:
        logger.error(f"Failed to copy weights: {e}")
        return None


def _recover_missing_checkpoint(err: str) -> Optional[str]:
    # Example: "Checkpoint /path/to/model_final.pth not found!"
    match = re.search(r"Checkpoint\s+([^\s]+)\s+not found", err)
    if not match:
        return None
    expected_path = match.group(1).strip()
    if os.path.exists(expected_path):
        return expected_path
    alt_path = expected_path + "?dl=1"
    if os.path.exists(alt_path):
        try:
            shutil.copyfile(alt_path, expected_path)
            return expected_path
        except Exception:
            return None
    # Fallback: search any cached weight and copy
    cache_root = Path(os.path.expanduser("~")) / ".torch" / "iopath_cache"
    for cand in cache_root.rglob("model_final.pth?dl=1"):
        try:
            shutil.copyfile(str(cand), expected_path)
            return expected_path
        except Exception:
            continue
    return None


def detect_layout(page_img_path: str, page_text: Optional[str] = None) -> List[dict]:
    """
    Detect layout elements (figures and tables) in a page image.
    
    Uses a fallback strategy:
    1. Try DocLayout-YOLO (best accuracy, document-specific)
    2. Fall back to PubLayNet (good accuracy, general purpose)
    3. Return empty list if both fail
    
    Args:
        page_img_path: Path to the page image
        page_text: Optional page text content for enhanced filtering
        
    Returns:
        List of detected layout blocks with type, bbox, and score
    """
    if page_img_path in _CACHE:
        logger.debug(f"Using cached layout for: {page_img_path}")
        return _CACHE[page_img_path]
    
    results: List[dict] = []
    
    if not layout_available():
        logger.debug("Layout detection not available (dependencies missing or disabled)")
        _CACHE[page_img_path] = results
        return results
    
    # Strategy 1: Try DocLayout-YOLO first (best accuracy)
    if DOCLAYOUT_AVAILABLE and not _DOCLAYOUT_FAILED:
        detector = _get_doclayout_detector()
        if detector is not None:
            try:
                logger.debug("Using DocLayout-YOLO for layout detection")
                detections = detector.detect(page_img_path, conf_threshold=config.LAYOUT_SCORE_THRESH)
                
                # Convert to standard format
                for det in detections:
                    # Only keep figures and tables
                    if det["label"] not in ("figure", "table"):
                        continue
                    
                    results.append({
                        "type": det["label"],
                        "bbox": det["bbox"],
                        "score": det["score"],
                        "detector": "doclayout_yolo"
                    })
                
                if results:
                    logger.info(f"DocLayout-YOLO found {len(results)} items ({sum(1 for r in results if r['type']=='figure')} figures, {sum(1 for r in results if r['type']=='table')} tables)")
                    _CACHE[page_img_path] = results
                    return results
                else:
                    logger.debug("DocLayout-YOLO found no figures/tables, trying fallback")
                    
            except Exception as e:
                logger.warning(f"DocLayout-YOLO detection failed: {e}, falling back to PubLayNet")
                logger.debug("Traceback:", exc_info=True)
    
    # Strategy 2: Fall back to PubLayNet
    if _publaynet_available():
        model = _get_model()
        if model is not None:
            try:
                import layoutparser as lp
                import cv2
                
                logger.debug("Using PubLayNet for layout detection (fallback)")
                image = cv2.imread(page_img_path)
                if image is None:
                    logger.warning(f"Failed to read image: {page_img_path}")
                    _CACHE[page_img_path] = results
                    return results
                
                layout = model.detect(image)
                
                logger.debug(f"PubLayNet detected {len(layout)} layout blocks")
                
                for block in layout:
                    block_type = str(block.type).lower()
                    score = float(getattr(block, "score", 1.0))
                    
                    # Filter by score threshold
                    if score < config.LAYOUT_SCORE_THRESH:
                        logger.debug(f"Filtered block {block_type} with score {score:.3f} < {config.LAYOUT_SCORE_THRESH}")
                        continue
                    
                    # Only keep figures and tables
                    if block_type not in ("figure", "table"):
                        logger.debug(f"Skipping block type: {block_type}")
                        continue
                    
                    bbox = _block_to_bbox(block)
                    results.append({
                        "type": block_type,
                        "bbox": bbox,
                        "score": score,
                        "detector": "publaynet"
                    })
                    logger.debug(f"Added {block_type} with score {score:.3f}, bbox: {bbox}")
                
                logger.info(f"PubLayNet found {len(results)} items ({sum(1 for r in results if r['type']=='figure')} figures, {sum(1 for r in results if r['type']=='table')} tables)")
                
            except Exception as e:
                logger.warning(f"PubLayNet detection failed on {page_img_path}: {e}")
                logger.debug("Traceback:", exc_info=True)
    else:
        logger.debug("PubLayNet not available, no fallback")
    
    # Apply filters in order:
    # 1. Table enhancer (only if no table detections were found)
    # 2. Text false positive filter (remove text false positives)
    # 3. Reclassify based on captions (fix figure/table misclassification)
    if config.TABLE_ENHANCER_ENABLE_IMG2TABLE:
        has_table = any(r["type"] == "table" for r in results)
        if not has_table:
            results = _apply_table_enhancer(results, page_img_path)
    if results:
        results = _apply_text_false_positive_filter(results, page_img_path, page_text)
        if config.LAYOUT_CAPTION_RECLASSIFY_ENABLE:
            results = _reclassify_by_caption(results, page_text)
    
    _CACHE[page_img_path] = results
    return results


def _reclassify_by_caption(detections: List[dict], page_text: Optional[str] = None) -> List[dict]:
    """
    Reclassify detections based on nearby captions with spatial awareness.
    
    If a detection labeled as "table" has a nearby "Figure X" caption,
    reclassify it as "figure" and vice versa.
    
    Args:
        detections: List of detection dictionaries
        page_text: Optional page text content
        
    Returns:
        Reclassified list of detections
    """
    if not page_text:
        return detections
    
    import re
    
    # Find all Figure and Table captions in the text
    # More comprehensive patterns including Chinese
    figure_pattern = r'(?:Figure|Fig\.|FIG\.|图|圖)\s*[0-9]+[a-z]?'
    table_pattern = r'(?:Table|Tab\.|TABLE|表|表格)\s*[0-9]+[a-z]?'
    
    figure_matches = list(re.finditer(figure_pattern, page_text, re.IGNORECASE))
    table_matches = list(re.finditer(table_pattern, page_text, re.IGNORECASE))
    
    if not figure_matches and not table_matches:
        return detections
    
    logger.info(f"Found {len(figure_matches)} figure captions and {len(table_matches)} table captions")
    
    reclassified = []
    reclassified_count = 0
    
    for det in detections:
        original_type = det['type']
        
        # Strategy 1: Count captions globally
        # If there are significantly more figure captions than table captions,
        # and a detection is labeled as "table", it's likely a figure
        figure_count = len(figure_matches)
        table_count = len(table_matches)
        
        # Only reclassify if there's a clear majority (at least 2:1 ratio)
        if det['type'] == 'table' and figure_count > table_count * 2 and figure_count >= 2:
            det['type'] = 'figure'
            det['reclassified'] = True
            det['reclassified_reason'] = f'Found {figure_count} figure captions vs {table_count} table captions (ratio {figure_count/max(table_count,1):.1f}:1)'
            reclassified_count += 1
            logger.info(f"Reclassified table -> figure based on caption ratio (bbox={det['bbox']})")
        
        elif det['type'] == 'figure' and table_count > figure_count * 2 and table_count >= 2:
            det['type'] = 'table'
            det['reclassified'] = True
            det['reclassified_reason'] = f'Found {table_count} table captions vs {figure_count} figure captions (ratio {table_count/max(figure_count,1):.1f}:1)'
            reclassified_count += 1
            logger.info(f"Reclassified figure -> table based on caption ratio (bbox={det['bbox']})")
        
        reclassified.append(det)
    
    if reclassified_count > 0:
        logger.info(f"Reclassified {reclassified_count} detections based on captions")
    
    return reclassified


def _apply_table_enhancer(
    detections: List[dict],
    image_path: str
) -> List[dict]:
    """
    Apply table enhancer to add missed tables using img2table.
    
    Args:
        detections: List of detection dictionaries
        image_path: Path to the page image
        
    Returns:
        Enhanced list of detections
    """
    # Get enhancer configuration from config
    enable_img2table = getattr(config, 'TABLE_ENHANCER_ENABLE_IMG2TABLE', True)
    iou_threshold = getattr(config, 'TABLE_ENHANCER_IOU_THRESHOLD', 0.3)
    min_confidence = getattr(config, 'TABLE_ENHANCER_MIN_CONFIDENCE', 0.5)
    shrink_bbox = getattr(config, 'TABLE_ENHANCER_SHRINK_BBOX', True)
    shrink_ratio = getattr(config, 'TABLE_ENHANCER_SHRINK_RATIO', 0.05)
    
    # Convert dict detections to Detection objects
    detection_objects = []
    for det in detections:
        detection_objects.append(Detection(
            type=det['type'],
            bbox=det['bbox'],
            score=det.get('score', 1.0),
            detector=det.get('detector', 'unknown')
        ))
    
    # Create and apply enhancer
    enhancer = TableEnhancer(
        iou_threshold=iou_threshold,
        min_confidence=min_confidence,
        enable_img2table=enable_img2table,
        shrink_bbox=shrink_bbox,
        shrink_ratio=shrink_ratio
    )
    
    enhanced_detections, added_detections = enhancer.enhance(detection_objects, image_path)
    
    # Log enhancement results
    if added_detections:
        logger.info(f"Table enhancer added {len(added_detections)} detections")
        for det in added_detections:
            logger.debug(f"  Added {det.type} at {det.bbox} (score={det.score:.3f})")
    
    # Convert back to dict format
    enhanced_results = []
    for det in enhanced_detections:
        # Check if this is a new detection (from enhancer)
        is_new = det in added_detections
        detector_name = 'img2table' if is_new else next((d['detector'] for d in detections if d['bbox'] == det.bbox), 'unknown')
        
        enhanced_results.append({
            'type': det.type,
            'bbox': det.bbox,
            'score': det.score,
            'detector': detector_name
        })
    
    return enhanced_results


def _apply_arxiv_filter(
    detections: List[dict],
    image_path: str
) -> List[dict]:
    """
    Apply arXiv filter to remove arXiv identifiers misdetected as figures.
    
    Args:
        detections: List of detection dictionaries
        image_path: Path to the page image
        
    Returns:
        Filtered list of detections
    """
    # Get filter configuration from config
    enable_ocr = getattr(config, 'ARXIV_FILTER_ENABLE_OCR', True)
    position_threshold = getattr(config, 'ARXIV_FILTER_POSITION_THRESHOLD', 0.1)
    area_threshold = getattr(config, 'ARXIV_FILTER_AREA_THRESHOLD', 0.05)
    aspect_ratio_min = getattr(config, 'ARXIV_FILTER_ASPECT_RATIO_MIN', 1.5)
    aspect_ratio_max = getattr(config, 'ARXIV_FILTER_ASPECT_RATIO_MAX', 8.0)
    check_left_margin = getattr(config, 'ARXIV_FILTER_CHECK_LEFT_MARGIN', True)
    left_margin_threshold = getattr(config, 'ARXIV_FILTER_LEFT_MARGIN_THRESHOLD', 0.15)
    check_rotation = getattr(config, 'ARXIV_FILTER_CHECK_ROTATION', True)
    
    # Convert dict detections to Detection objects
    detection_objects = []
    for det in detections:
        detection_objects.append(Detection(
            type=det['type'],
            bbox=det['bbox'],
            score=det.get('score', 1.0),
            detector=det.get('detector', 'unknown')
        ))
    
    # Create and apply filter
    arxiv_filter = ArxivFilter(
        enable_ocr=enable_ocr,
        position_threshold=position_threshold,
        area_threshold=area_threshold,
        aspect_ratio_range=(aspect_ratio_min, aspect_ratio_max),
        check_left_margin=check_left_margin,
        left_margin_threshold=left_margin_threshold,
        check_rotation=check_rotation
    )
    
    filtered_detections, removed_detections = arxiv_filter.filter(detection_objects, image_path)
    
    # Log filtering results
    if removed_detections:
        logger.info(f"arXiv filter removed {len(removed_detections)} detections")
        for det in removed_detections:
            logger.debug(f"  Removed {det.type} at {det.bbox} (score={det.score:.3f})")
    
    # Convert back to dict format
    filtered_results = []
    for det in filtered_detections:
        filtered_results.append({
            'type': det.type,
            'bbox': det.bbox,
            'score': det.score,
            'detector': next((d['detector'] for d in detections if d['bbox'] == det.bbox), 'unknown')
        })
    
    return filtered_results


def _apply_text_false_positive_filter(
    detections: List[dict],
    image_path: str,
    page_text: Optional[str] = None
) -> List[dict]:
    """
    Apply text false positive filter to remove text paragraphs misdetected as tables.
    
    Args:
        detections: List of detection dictionaries
        image_path: Path to the page image
        page_text: Optional page text content for pattern matching
        
    Returns:
        Filtered list of detections
    """
    # Get filter configuration from config
    table_confidence_threshold = getattr(config, 'TEXT_FILTER_CONFIDENCE_THRESHOLD', 0.7)
    enable_transformer = getattr(config, 'TEXT_FILTER_ENABLE_TRANSFORMER', False)
    text_density_threshold = getattr(config, 'TEXT_FILTER_TEXT_DENSITY_THRESHOLD', 0.08)
    min_table_structure_score = getattr(config, 'TEXT_FILTER_MIN_STRUCTURE_SCORE', 200)
    enable_position_heuristics = getattr(config, 'TEXT_FILTER_ENABLE_POSITION_HEURISTICS', True)
    enable_ocr_pattern_matching = getattr(config, 'TEXT_FILTER_ENABLE_OCR_PATTERN', True)
    enable_text_line_detection = getattr(config, 'TEXT_FILTER_ENABLE_TEXT_LINE_DETECTION', True)
    
    # Convert dict detections to Detection objects
    detection_objects = []
    for det in detections:
        detection_objects.append(Detection(
            type=det['type'],
            bbox=det['bbox'],
            score=det.get('score', 1.0),
            detector=det.get('detector', 'unknown')
        ))
    
    # Create and apply filter
    text_filter = TextFalsePositiveFilter(
        table_confidence_threshold=table_confidence_threshold,
        enable_transformer_verification=enable_transformer,
        text_density_threshold=text_density_threshold,
        min_table_structure_score=min_table_structure_score,
        enable_position_heuristics=enable_position_heuristics,
        enable_ocr_pattern_matching=enable_ocr_pattern_matching,
        enable_text_line_detection=enable_text_line_detection
    )
    
    filtered_detections, removed_detections = text_filter.filter(detection_objects, image_path, page_text)
    
    # Log filtering results
    if removed_detections:
        logger.info(f"Text false positive filter removed {len(removed_detections)} detections")
        for det in removed_detections:
            logger.debug(f"  Removed {det.type} at {det.bbox} (score={det.score:.3f})")
    
    # Convert back to dict format
    filtered_results = []
    for det in filtered_detections:
        filtered_results.append({
            'type': det.type,
            'bbox': det.bbox,
            'score': det.score,
            'detector': det.detector
        })
    
    return filtered_results

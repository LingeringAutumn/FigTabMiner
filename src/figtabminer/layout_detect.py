import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from . import config
from . import utils

logger = utils.setup_logging(__name__)

_MODEL = None
_MODEL_FAILED = False
_CACHE: Dict[str, List[dict]] = {}


def get_layout_status() -> dict:
    """
    Get the current status of layout detection capability.
    
    Returns:
        Dictionary with status information
    """
    status = {
        "available": layout_available(),
        "model_loaded": _MODEL is not None,
        "model_failed": _MODEL_FAILED,
        "cache_size": len(_CACHE),
    }
    
    if status["available"] and not status["model_loaded"] and not status["model_failed"]:
        status["status"] = "not_initialized"
    elif status["model_loaded"]:
        status["status"] = "ready"
    elif status["model_failed"]:
        status["status"] = "failed"
    else:
        status["status"] = "unavailable"
    
    return status


def layout_available() -> bool:
    if config.LAYOUT_ENABLE in ("0", "false", "no", "n", "off"):
        return False
    if not utils.safe_import("layoutparser"):
        return False
    if not utils.safe_import("detectron2"):
        return False
    return True


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


def detect_layout(page_img_path: str) -> List[dict]:
    """
    Detect layout elements (figures and tables) in a page image.
    
    Args:
        page_img_path: Path to the page image
        
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
    
    model = _get_model()
    if model is None:
        logger.debug("Layout model not initialized, skipping detection")
        _CACHE[page_img_path] = results
        return results
    
    try:
        import layoutparser as lp
        import cv2
        
        logger.debug(f"Running layout detection on: {page_img_path}")
        # Use cv2 to read image instead of lp.io.read_image (which may not exist in all versions)
        image = cv2.imread(page_img_path)
        if image is None:
            logger.warning(f"Failed to read image: {page_img_path}")
            _CACHE[page_img_path] = results
            return results
        
        layout = model.detect(image)
        
        logger.debug(f"Detected {len(layout)} layout blocks")
        
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
            })
            logger.debug(f"Added {block_type} with score {score:.3f}, bbox: {bbox}")
        
        logger.info(f"Layout detection found {len(results)} items ({sum(1 for r in results if r['type']=='figure')} figures, {sum(1 for r in results if r['type']=='table')} tables)")
        
    except Exception as e:
        logger.warning(f"Layout detection failed on {page_img_path}: {e}")
        logger.debug("Traceback:", exc_info=True)
    
    _CACHE[page_img_path] = results
    return results

#!/usr/bin/env python3
"""
DocLayout-YOLO detector for document layout analysis.

This detector provides state-of-the-art layout detection specifically
designed for document analysis tasks.
"""

import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from .. import utils

logger = utils.setup_logging(__name__)

try:
    from doclayout_yolo import YOLOv10
    DOCLAYOUT_AVAILABLE = True
    logger.info("DocLayout-YOLO is available")
except ImportError:
    DOCLAYOUT_AVAILABLE = False
    logger.debug("DocLayout-YOLO not installed")


class DocLayoutYOLODetector:
    """DocLayout-YOLO detector wrapper for document layout analysis"""
    
    # Label mapping from DocLayout-YOLO class IDs to semantic labels
    LABEL_MAP = {
        0: "text",
        1: "title",
        2: "figure",
        3: "table",
        4: "caption",
        5: "header",
        6: "footer",
        7: "reference",
        8: "equation"
    }
    
    def __init__(self, model_name: str = None):
        """
        Initialize DocLayout-YOLO detector.
        
        Args:
            model_name: Name of the model to use. If None, will try multiple options.
        
        Raises:
            ImportError: If doclayout-yolo is not installed
            RuntimeError: If model initialization fails
        """
        if not DOCLAYOUT_AVAILABLE:
            raise ImportError(
                "doclayout-yolo not installed. "
                "Install with: pip install doclayout-yolo"
            )
        
        # Try multiple model names in order of preference
        if model_name is None:
            import os
            from pathlib import Path
            
            # Check for downloaded model in Hugging Face cache
            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            model_names = []
            
            # Look for downloaded model in cache
            if hf_cache.exists():
                for model_dir in hf_cache.glob("models--juliozhao--DocLayout-YOLO-DocStructBench"):
                    for snapshot_dir in model_dir.glob("snapshots/*"):
                        model_file = snapshot_dir / "doclayout_yolo_docstructbench_imgsz1024.pt"
                        if model_file.exists():
                            model_names.append(str(model_file))
                            logger.debug(f"Found cached model: {model_file}")
            
            # Add other options
            model_names.extend([
                # From Hugging Face (will download if not cached)
                "juliozhao/DocLayout-YOLO-DocStructBench",
                # Local cached versions
                "doclayout_yolo_docstructbench_imgsz1024.pt",
                # Fallback to basic YOLO
                "yolov10n.pt",
            ])
        else:
            model_names = [model_name]
        
        last_error = None
        for name in model_names:
            try:
                logger.info(f"Trying to load DocLayout-YOLO model: {name}")
                self.model = YOLOv10(name)
                self.model_name = name
                logger.info(f"DocLayout-YOLO model initialized successfully: {name}")
                return
            except Exception as e:
                logger.debug(f"Failed to load {name}: {e}")
                last_error = e
                continue
        
        # If all attempts failed
        error_msg = f"Failed to initialize DocLayout-YOLO with any model name. Last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def detect(self, image_path: str, conf_threshold: float = 0.25) -> List[Dict]:
        """
        Detect layout elements in image.
        
        Args:
            image_path: Path to the image file
            conf_threshold: Confidence threshold for detections (0.0-1.0)
        
        Returns:
            List of detections, each containing:
                - bbox: [x0, y0, x1, y1] bounding box coordinates
                - label: Semantic label (e.g., "figure", "table")
                - score: Confidence score (0.0-1.0)
                - class_id: Original class ID from model
        """
        try:
            logger.debug(f"Running DocLayout-YOLO detection on: {image_path}")
            
            # Run prediction
            results = self.model.predict(
                image_path,
                imgsz=1024,
                conf=conf_threshold,
                device="cuda" if self._has_cuda() else "cpu",
                verbose=False  # Suppress YOLO output
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Extract bbox coordinates [x0, y0, x1, y1]
                    bbox = boxes.xyxy[i].cpu().numpy()
                    
                    # Extract confidence score
                    conf = float(boxes.conf[i].cpu().numpy())
                    
                    # Extract class ID
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    # Map class ID to semantic label
                    label = self.LABEL_MAP.get(cls, "unknown")
                    
                    detections.append({
                        "bbox": bbox.tolist(),
                        "label": label,
                        "score": conf,
                        "class_id": cls
                    })
            
            logger.debug(f"DocLayout-YOLO detected {len(detections)} elements")
            
            # Log detection summary
            if detections:
                label_counts = {}
                for det in detections:
                    label = det["label"]
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                summary = ", ".join([f"{count} {label}s" for label, count in sorted(label_counts.items())])
                logger.info(f"DocLayout-YOLO found: {summary}")
            
            return detections
            
        except Exception as e:
            logger.error(f"DocLayout-YOLO detection failed: {e}")
            logger.debug("Traceback:", exc_info=True)
            return []
    
    def _has_cuda(self) -> bool:
        """Check if CUDA is available for GPU acceleration"""
        try:
            import torch
            available = torch.cuda.is_available()
            if available:
                logger.debug("CUDA is available, using GPU")
            else:
                logger.debug("CUDA not available, using CPU")
            return available
        except ImportError:
            logger.debug("PyTorch not available, using CPU")
            return False


def detect_layout_doclayout(
    image_path: str, 
    conf_threshold: float = 0.25
) -> List[Dict]:
    """
    Convenience function for DocLayout-YOLO layout detection.
    
    Args:
        image_path: Path to the image file
        conf_threshold: Confidence threshold for detections (0.0-1.0)
    
    Returns:
        List of detections with bbox, label, score, and class_id
    """
    if not DOCLAYOUT_AVAILABLE:
        logger.warning("DocLayout-YOLO not available")
        return []
    
    try:
        detector = DocLayoutYOLODetector()
        return detector.detect(image_path, conf_threshold)
    except Exception as e:
        logger.error(f"DocLayout-YOLO detection failed: {e}")
        return []


def is_available() -> bool:
    """Check if DocLayout-YOLO is available"""
    return DOCLAYOUT_AVAILABLE

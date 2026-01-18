#!/usr/bin/env python3
"""
Table Transformer detector for table detection and structure recognition.

This detector uses Microsoft's Table Transformer models for:
1. Table detection in document images
2. Table structure recognition (rows, columns, cells)
"""

import torch
from PIL import Image
from typing import List, Dict, Optional
import numpy as np

from .. import utils

logger = utils.setup_logging(__name__)

try:
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    TABLE_TRANSFORMER_AVAILABLE = True
    logger.info("Table Transformer is available")
except ImportError:
    TABLE_TRANSFORMER_AVAILABLE = False
    logger.debug("Table Transformer not installed")


class TableTransformerDetector:
    """Table Transformer detector wrapper for table detection and structure recognition"""
    
    def __init__(self):
        """
        Initialize Table Transformer detector.
        
        Loads both detection and structure recognition models.
        
        Raises:
            ImportError: If transformers is not installed
            RuntimeError: If model initialization fails
        """
        if not TABLE_TRANSFORMER_AVAILABLE:
            raise ImportError(
                "transformers not installed. "
                "Install with: pip install transformers torch torchvision"
            )
        
        try:
            logger.info("Initializing Table Transformer models...")
            
            # Detection model
            logger.debug("Loading table detection model...")
            self.detection_processor = AutoImageProcessor.from_pretrained(
                "microsoft/table-transformer-detection"
            )
            self.detection_model = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-detection"
            )
            
            # Structure recognition model
            logger.debug("Loading table structure recognition model...")
            self.structure_processor = AutoImageProcessor.from_pretrained(
                "microsoft/table-transformer-structure-recognition"
            )
            self.structure_model = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-structure-recognition"
            )
            
            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.debug(f"Using device: {self.device}")
            
            self.detection_model.to(self.device)
            self.structure_model.to(self.device)
            
            logger.info("Table Transformer models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Table Transformer: {e}")
            raise RuntimeError(f"Table Transformer initialization failed: {e}")
    
    def detect_tables(self, image_path: str, conf_threshold: float = 0.7) -> List[Dict]:
        """
        Detect tables in image.
        
        Args:
            image_path: Path to the image file
            conf_threshold: Confidence threshold for detections (0.0-1.0)
        
        Returns:
            List of table detections, each containing:
                - bbox: [x0, y0, x1, y1] bounding box coordinates
                - score: Confidence score (0.0-1.0)
                - label: Always "table"
        """
        try:
            logger.debug(f"Running Table Transformer detection on: {image_path}")
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Prepare inputs
            inputs = self.detection_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run detection
            with torch.no_grad():
                outputs = self.detection_model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.detection_processor.post_process_object_detection(
                outputs, 
                threshold=conf_threshold, 
                target_sizes=target_sizes
            )[0]
            
            tables = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                # Table Transformer detection model: class 0 = table
                if int(label.cpu().numpy()) == 0:
                    tables.append({
                        "bbox": box.cpu().numpy().tolist(),
                        "score": float(score.cpu().numpy()),
                        "label": "table"
                    })
            
            logger.debug(f"Table Transformer detected {len(tables)} tables")
            
            if tables:
                logger.info(f"Table Transformer found {len(tables)} tables")
            
            return tables
            
        except Exception as e:
            logger.error(f"Table Transformer detection failed: {e}")
            logger.debug("Traceback:", exc_info=True)
            return []
    
    def recognize_structure(self, image_path: str, table_bbox: List[float]) -> Dict:
        """
        Recognize table structure (rows, columns, cells).
        
        Args:
            image_path: Path to the image file
            table_bbox: Table bounding box [x0, y0, x1, y1]
        
        Returns:
            Dictionary containing:
                - rows: List of row bounding boxes
                - columns: List of column bounding boxes
                - cells: List of cell bounding boxes
        """
        try:
            logger.debug(f"Recognizing table structure for bbox: {table_bbox}")
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Crop to table region
            table_crop = image.crop(table_bbox)
            
            # Prepare inputs
            inputs = self.structure_processor(images=table_crop, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run structure recognition
            with torch.no_grad():
                outputs = self.structure_model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([table_crop.size[::-1]]).to(self.device)
            results = self.structure_processor.post_process_object_detection(
                outputs,
                threshold=0.6,
                target_sizes=target_sizes
            )[0]
            
            # Extract structure elements
            rows = []
            columns = []
            cells = []
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                element = {
                    "bbox": box.cpu().numpy().tolist(),
                    "score": float(score.cpu().numpy())
                }
                
                label_id = int(label.cpu().numpy())
                # Structure model: 0=table row, 1=table column, 2=table cell
                if label_id == 0:
                    rows.append(element)
                elif label_id == 1:
                    columns.append(element)
                elif label_id == 2:
                    cells.append(element)
            
            logger.debug(f"Structure recognition found: {len(rows)} rows, {len(columns)} columns, {len(cells)} cells")
            
            return {
                "rows": rows,
                "columns": columns,
                "cells": cells
            }
            
        except Exception as e:
            logger.error(f"Table structure recognition failed: {e}")
            logger.debug("Traceback:", exc_info=True)
            return {
                "rows": [],
                "columns": [],
                "cells": []
            }


def detect_tables_transformer(image_path: str, conf_threshold: float = 0.7) -> List[Dict]:
    """
    Convenience function for table detection using Table Transformer.
    
    Args:
        image_path: Path to the image file
        conf_threshold: Confidence threshold for detections (0.0-1.0)
    
    Returns:
        List of table detections with bbox, score, and label
    """
    if not TABLE_TRANSFORMER_AVAILABLE:
        logger.warning("Table Transformer not available")
        return []
    
    try:
        detector = TableTransformerDetector()
        return detector.detect_tables(image_path, conf_threshold)
    except Exception as e:
        logger.error(f"Table Transformer detection failed: {e}")
        return []


def is_available() -> bool:
    """Check if Table Transformer is available"""
    return TABLE_TRANSFORMER_AVAILABLE

#!/usr/bin/env python3
"""
Intelligent Detection Fusion Module for FigTabMiner v1.7.

This module implements smart fusion strategies for combining results from
multiple detection models (DocLayout-YOLO, Table Transformer, PubLayNet).
It handles detector weighting, conflict resolution, and adaptive thresholding.

Key Features:
- Weighted NMS fusion for multi-detector results
- Conflict resolution based on detector reliability
- Adaptive confidence thresholding
- Configurable fusion strategies

Requirements: 6.1, 6.5
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from .models import Detection, DetectionConfig
from . import bbox_utils


# Configure logging
logger = logging.getLogger(__name__)


class IntelligentDetectionFusion:
    """
    Intelligent fusion of multiple detector results.
    
    This class implements sophisticated strategies for combining detection
    results from multiple models, taking into account each detector's
    strengths, weaknesses, and reliability for different types of content.
    
    Attributes:
        config: DetectionConfig object containing fusion parameters
        detector_weights: Dictionary mapping detector names to weights
        fusion_strategy: Strategy to use ('weighted_nms', 'voting', 'cascade')
        confidence_thresholds: Per-detector confidence thresholds
        enable_adaptive_threshold: Whether to use adaptive thresholding
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Initialize the intelligent detection fusion module.
        
        Args:
            config: DetectionConfig object. If None, uses default configuration.
        
        Requirements: 6.1, 6.5
        """
        # Load configuration
        if config is None:
            from .models import get_default_config
            config = get_default_config()
        
        self.config = config
        
        # Extract fusion-specific parameters
        self.detector_weights = config.detector_weights.copy()
        self.fusion_strategy = config.fusion_strategy
        self.nms_iou_threshold = config.nms_iou_threshold
        
        # Build confidence thresholds dictionary
        self.confidence_thresholds = {
            'doclayout': config.doclayout_confidence,
            'table_transformer': config.table_transformer_confidence,
            'legacy': config.legacy_confidence,
            'publaynet': config.legacy_confidence,  # Alias for legacy
        }
        
        # Adaptive thresholding (enabled by default for v1.7)
        self.enable_adaptive_threshold = True
        
        # Detector reliability scores (can be updated based on performance metrics)
        # Higher scores indicate more reliable detectors
        self.detector_reliability = {
            'doclayout': 0.85,  # Good for general figures
            'table_transformer': 0.90,  # Excellent for tables
            'legacy': 0.75,  # Baseline detector
            'publaynet': 0.75,
        }
        
        logger.info(
            f"Initialized IntelligentDetectionFusion with strategy='{self.fusion_strategy}', "
            f"weights={self.detector_weights}"
        )
    
    def get_detector_weight(self, detector_name: str) -> float:
        """
        Get the weight for a specific detector.
        
        Args:
            detector_name: Name of the detector
            
        Returns:
            Weight value (0-1), defaults to 0.5 if detector not found
        """
        # Normalize detector name (handle variations)
        detector_name = detector_name.lower()
        
        # Try exact match first
        if detector_name in self.detector_weights:
            return self.detector_weights[detector_name]
        
        # Try common aliases
        aliases = {
            'publaynet': 'legacy',
            'detectron': 'legacy',
            'detectron2': 'legacy',
            'doclayout-yolo': 'doclayout',
            'doclayout_yolo': 'doclayout',
            'table-transformer': 'table_transformer',
        }
        
        if detector_name in aliases:
            canonical_name = aliases[detector_name]
            if canonical_name in self.detector_weights:
                return self.detector_weights[canonical_name]
        
        # Default weight
        logger.warning(f"Unknown detector '{detector_name}', using default weight 0.5")
        return 0.5
    
    def get_detector_reliability(self, detector_name: str) -> float:
        """
        Get the reliability score for a specific detector.
        
        Args:
            detector_name: Name of the detector
            
        Returns:
            Reliability score (0-1), defaults to 0.7 if detector not found
        """
        detector_name = detector_name.lower()
        
        if detector_name in self.detector_reliability:
            return self.detector_reliability[detector_name]
        
        # Check aliases
        aliases = {
            'publaynet': 'legacy',
            'detectron': 'legacy',
            'detectron2': 'legacy',
        }
        
        if detector_name in aliases:
            canonical_name = aliases[detector_name]
            if canonical_name in self.detector_reliability:
                return self.detector_reliability[canonical_name]
        
        # Default reliability
        return 0.7
    
    def update_detector_weights(self, weights: Dict[str, float]):
        """
        Update detector weights dynamically.
        
        This allows for runtime adjustment of detector weights based on
        performance metrics or user preferences.
        
        Args:
            weights: Dictionary mapping detector names to new weights
        """
        for detector, weight in weights.items():
            if not 0.0 <= weight <= 1.0:
                logger.warning(
                    f"Invalid weight {weight} for detector '{detector}', "
                    f"must be in range [0, 1]"
                )
                continue
            
            self.detector_weights[detector.lower()] = weight
        
        logger.info(f"Updated detector weights: {self.detector_weights}")
    
    def update_detector_reliability(self, reliability: Dict[str, float]):
        """
        Update detector reliability scores.
        
        This should be called periodically based on evaluation metrics
        to reflect the actual performance of each detector.
        
        Args:
            reliability: Dictionary mapping detector names to reliability scores
        """
        for detector, score in reliability.items():
            if not 0.0 <= score <= 1.0:
                logger.warning(
                    f"Invalid reliability score {score} for detector '{detector}', "
                    f"must be in range [0, 1]"
                )
                continue
            
            self.detector_reliability[detector.lower()] = score
        
        logger.info(f"Updated detector reliability: {self.detector_reliability}")
    
    def filter_by_confidence(
        self,
        detections: List[Detection],
        detector_name: Optional[str] = None
    ) -> List[Detection]:
        """
        Filter detections by confidence threshold.
        
        Args:
            detections: List of Detection objects
            detector_name: Optional detector name to use specific threshold
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        filtered = []
        for det in detections:
            # Get threshold for this detector
            threshold = self.confidence_thresholds.get(
                det.detector.lower(),
                0.5  # Default threshold
            )
            
            if det.score >= threshold:
                filtered.append(det)
            else:
                logger.debug(
                    f"Filtered out detection from {det.detector} "
                    f"with score {det.score:.3f} < {threshold:.3f}"
                )
        
        logger.info(
            f"Filtered {len(detections)} detections to {len(filtered)} "
            f"using confidence thresholds"
        )
        
        return filtered
    
    def calculate_weighted_score(self, detection: Detection) -> float:
        """
        Calculate weighted confidence score for a detection.
        
        The weighted score combines the detector's confidence with the
        detector's weight and reliability.
        
        Args:
            detection: Detection object
            
        Returns:
            Weighted score (0-1)
        """
        detector_weight = self.get_detector_weight(detection.detector)
        detector_reliability = self.get_detector_reliability(detection.detector)
        
        # Weighted score = detection_confidence * detector_weight * detector_reliability
        weighted_score = detection.score * detector_weight * detector_reliability
        
        return weighted_score
    
    def get_detector_priority(self, detector_name: str, detection_type: str) -> int:
        """
        Get priority for a detector based on detection type.
        
        Different detectors have different strengths:
        - Table Transformer: Best for tables
        - DocLayout-YOLO: Good for general figures
        - Legacy: Baseline
        
        Args:
            detector_name: Name of the detector
            detection_type: Type of detection ('figure' or 'table')
            
        Returns:
            Priority value (higher = more priority)
        """
        detector_name = detector_name.lower()
        
        # Priority matrix: detector -> type -> priority
        priority_matrix = {
            'table_transformer': {'table': 3, 'figure': 1},
            'doclayout': {'figure': 2, 'table': 2},
            'legacy': {'figure': 1, 'table': 1},
            'publaynet': {'figure': 1, 'table': 1},
        }
        
        if detector_name in priority_matrix:
            return priority_matrix[detector_name].get(detection_type, 1)
        
        # Default priority
        return 1
    
    def __repr__(self) -> str:
        """String representation of the fusion module."""
        return (
            f"IntelligentDetectionFusion("
            f"strategy='{self.fusion_strategy}', "
            f"weights={self.detector_weights}, "
            f"adaptive_threshold={self.enable_adaptive_threshold})"
        )

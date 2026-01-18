#!/usr/bin/env python3
"""
Core data models for FigTabMiner v1.7.

This module defines the data structures used throughout the detection,
classification, quality assessment, and analysis pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import json


@dataclass
class Detection:
    """
    Detection result data model.
    
    Represents a detected figure or table with its bounding box,
    type, confidence score, and optional metadata.
    """
    bbox: List[float]  # [x0, y0, x1, y1]
    type: str  # 'figure' or 'table'
    score: float  # Confidence score (0-1)
    detector: str  # Detector name (e.g., 'doclayout', 'table_transformer', 'publaynet')
    class_id: int = 0  # Original class ID from detector
    label: str = ""  # Semantic label (e.g., 'figure', 'table')
    
    # Optional fields for enhanced processing
    merged_from: Optional[int] = None  # Number of detections merged to create this
    quality_score: Optional[float] = None  # Quality score (0-1)
    quality_details: Optional[Dict[str, Any]] = None  # Detailed quality metrics
    chart_type: Optional[str] = None  # Chart type (e.g., 'bar_chart', 'line_plot')
    chart_confidence: Optional[float] = None  # Chart type classification confidence
    
    def __post_init__(self):
        """Validate and normalize fields after initialization."""
        # Ensure bbox has exactly 4 coordinates
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must have exactly 4 coordinates, got {len(self.bbox)}")
        
        # Ensure coordinates are floats
        self.bbox = [float(x) for x in self.bbox]
        
        # Validate type
        if self.type not in ('figure', 'table'):
            raise ValueError(f"type must be 'figure' or 'table', got '{self.type}'")
        
        # Validate score range
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in range [0, 1], got {self.score}")
        
        # Set label from type if not provided
        if not self.label:
            self.label = self.type
    
    def area(self) -> float:
        """Calculate the area of the bounding box."""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    
    def width(self) -> float:
        """Calculate the width of the bounding box."""
        return self.bbox[2] - self.bbox[0]
    
    def height(self) -> float:
        """Calculate the height of the bounding box."""
        return self.bbox[3] - self.bbox[1]
    
    def aspect_ratio(self) -> float:
        """Calculate the aspect ratio (width/height) of the bounding box."""
        h = self.height()
        if h == 0:
            return float('inf')
        return self.width() / h
    
    def iou(self, other: 'Detection') -> float:
        """
        Calculate Intersection over Union (IoU) with another detection.
        
        Args:
            other: Another Detection object
            
        Returns:
            IoU value in range [0, 1]
        """
        # Calculate intersection
        x1 = max(self.bbox[0], other.bbox[0])
        y1 = max(self.bbox[1], other.bbox[1])
        x2 = min(self.bbox[2], other.bbox[2])
        y2 = min(self.bbox[3], other.bbox[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area() + other.area() - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'bbox': self.bbox,
            'type': self.type,
            'score': self.score,
            'detector': self.detector,
            'class_id': self.class_id,
            'label': self.label,
            'merged_from': self.merged_from,
            'quality_score': self.quality_score,
            'quality_details': self.quality_details,
            'chart_type': self.chart_type,
            'chart_confidence': self.chart_confidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Detection':
        """Create Detection from dictionary."""
        return cls(
            bbox=data['bbox'],
            type=data['type'],
            score=data['score'],
            detector=data['detector'],
            class_id=data.get('class_id', 0),
            label=data.get('label', ''),
            merged_from=data.get('merged_from'),
            quality_score=data.get('quality_score'),
            quality_details=data.get('quality_details'),
            chart_type=data.get('chart_type'),
            chart_confidence=data.get('chart_confidence'),
        )


@dataclass
class ClassificationResult:
    """
    Chart classification result.
    
    Contains the classified chart type, confidence score, and optional
    metadata about the classification process.
    """
    chart_type: str  # Chart type (e.g., 'bar_chart', 'line_plot', 'unknown')
    confidence: float  # Classification confidence (0-1)
    sub_type: Optional[str] = None  # Sub-type (e.g., 'grouped_bar', 'stacked_bar')
    matched_keywords: List[str] = field(default_factory=list)  # Keywords that matched
    visual_features: Optional[Dict[str, Any]] = None  # Extracted visual features
    debug_info: Optional[Dict[str, Any]] = None  # Debug information
    
    def __post_init__(self):
        """Validate fields after initialization."""
        # Validate confidence range
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in range [0, 1], got {self.confidence}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'chart_type': self.chart_type,
            'confidence': self.confidence,
            'sub_type': self.sub_type,
            'matched_keywords': self.matched_keywords,
            'visual_features': self.visual_features,
            'debug_info': self.debug_info,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClassificationResult':
        """Create ClassificationResult from dictionary."""
        return cls(
            chart_type=data['chart_type'],
            confidence=data['confidence'],
            sub_type=data.get('sub_type'),
            matched_keywords=data.get('matched_keywords', []),
            visual_features=data.get('visual_features'),
            debug_info=data.get('debug_info'),
        )


@dataclass
class QualityScore:
    """
    Quality assessment score.
    
    Multi-dimensional quality score for a detection result,
    including overall score and individual dimension scores.
    """
    overall_score: float  # Overall quality score (0-1)
    dimension_scores: Dict[str, float] = field(default_factory=dict)  # Individual dimension scores
    issues: List[str] = field(default_factory=list)  # Identified issues
    recommendations: List[str] = field(default_factory=list)  # Improvement recommendations
    
    def __post_init__(self):
        """Validate fields after initialization."""
        # Validate overall score range
        if not 0.0 <= self.overall_score <= 1.0:
            raise ValueError(f"overall_score must be in range [0, 1], got {self.overall_score}")
        
        # Validate dimension scores
        for dim, score in self.dimension_scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"dimension score '{dim}' must be in range [0, 1], got {score}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'overall_score': self.overall_score,
            'dimension_scores': self.dimension_scores,
            'issues': self.issues,
            'recommendations': self.recommendations,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QualityScore':
        """Create QualityScore from dictionary."""
        return cls(
            overall_score=data['overall_score'],
            dimension_scores=data.get('dimension_scores', {}),
            issues=data.get('issues', []),
            recommendations=data.get('recommendations', []),
        )


@dataclass
class AnalysisResult:
    """
    AI analysis result.
    
    Contains detailed analysis of a chart including subtype,
    scientific metadata, and extracted information.
    """
    subtype: str  # Chart subtype
    subtype_confidence: float  # Subtype classification confidence (0-1)
    conditions: List[str] = field(default_factory=list)  # Experimental conditions
    materials: List[str] = field(default_factory=list)  # Material candidates
    keywords: List[str] = field(default_factory=list)  # Extracted keywords
    method: str = "enhanced_analyzer_v1.7"  # Analysis method identifier
    debug: Optional[Dict[str, Any]] = None  # Debug information
    
    def __post_init__(self):
        """Validate fields after initialization."""
        # Validate confidence range
        if not 0.0 <= self.subtype_confidence <= 1.0:
            raise ValueError(f"subtype_confidence must be in range [0, 1], got {self.subtype_confidence}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'subtype': self.subtype,
            'subtype_confidence': self.subtype_confidence,
            'conditions': self.conditions,
            'materials': self.materials,
            'keywords': self.keywords,
            'method': self.method,
            'debug': self.debug,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Create AnalysisResult from dictionary."""
        return cls(
            subtype=data['subtype'],
            subtype_confidence=data['subtype_confidence'],
            conditions=data.get('conditions', []),
            materials=data.get('materials', []),
            keywords=data.get('keywords', []),
            method=data.get('method', 'enhanced_analyzer_v1.7'),
            debug=data.get('debug'),
        )


@dataclass
class EvaluationMetrics:
    """
    Accuracy evaluation metrics.
    
    Contains precision, recall, F1 score, and detailed information
    about true positives, false positives, and false negatives.
    """
    precision: float  # Precision (TP / (TP + FP))
    recall: float  # Recall (TP / (TP + FN))
    f1_score: float  # F1 score (harmonic mean of precision and recall)
    mean_iou: float  # Mean IoU for true positives
    
    true_positives: List[Tuple[Detection, Detection]] = field(default_factory=list)  # (pred, gt) pairs
    false_positives: List[Detection] = field(default_factory=list)  # Unmatched predictions
    false_negatives: List[Detection] = field(default_factory=list)  # Unmatched ground truth
    
    # Statistics
    total_predictions: int = 0
    total_ground_truth: int = 0
    
    def __post_init__(self):
        """Validate fields after initialization."""
        # Validate metric ranges
        for metric_name, metric_value in [
            ('precision', self.precision),
            ('recall', self.recall),
            ('f1_score', self.f1_score),
            ('mean_iou', self.mean_iou)
        ]:
            if not 0.0 <= metric_value <= 1.0:
                raise ValueError(f"{metric_name} must be in range [0, 1], got {metric_value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation (without detection objects)."""
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'mean_iou': self.mean_iou,
            'tp_count': len(self.true_positives),
            'fp_count': len(self.false_positives),
            'fn_count': len(self.false_negatives),
            'total_predictions': self.total_predictions,
            'total_ground_truth': self.total_ground_truth,
        }


@dataclass
class DetectionConfig:
    """
    Detection system configuration.
    
    Contains all configurable parameters for the detection pipeline,
    including detector settings, fusion strategy, merging parameters,
    quality thresholds, and performance options.
    """
    # Detector enable/disable flags
    enable_doclayout: bool = True
    enable_table_transformer: bool = True
    enable_legacy_detectron: bool = False
    
    # Confidence thresholds for each detector
    doclayout_confidence: float = 0.35  # Increased from 0.25 to reduce false positives
    table_transformer_confidence: float = 0.75
    legacy_confidence: float = 0.5
    
    # Fusion configuration
    fusion_strategy: str = 'weighted_nms'  # Options: 'weighted_nms', 'voting', 'cascade'
    detector_weights: Dict[str, float] = field(default_factory=lambda: {
        'doclayout': 0.4,
        'table_transformer': 0.4,
        'legacy': 0.2
    })
    nms_iou_threshold: float = 0.5  # IoU threshold for NMS
    
    # Merging configuration
    merge_iou_threshold: float = 0.3
    merge_distance_threshold: float = 50
    enable_context_aware_merge: bool = True
    enable_boundary_refinement: bool = True
    
    # Quality assessment configuration
    min_quality_score: float = 0.4
    enable_anomaly_detection: bool = True
    quality_weights: Dict[str, float] = field(default_factory=lambda: {
        'detection_confidence': 0.30,
        'content_completeness': 0.25,
        'boundary_precision': 0.20,
        'caption_match': 0.15,
        'position_reasonableness': 0.10
    })
    
    # Classification configuration
    enable_hierarchical_classification: bool = True
    min_classification_confidence: float = 0.5
    
    # Performance configuration
    enable_parallel_detection: bool = True
    enable_model_caching: bool = True
    max_workers: int = 4  # For parallel processing
    
    # Backward compatibility
    use_legacy_pipeline: bool = False  # If True, use v1.6 pipeline
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        errors = []
        
        # Validate confidence thresholds
        for name, value in [
            ('doclayout_confidence', self.doclayout_confidence),
            ('table_transformer_confidence', self.table_transformer_confidence),
            ('legacy_confidence', self.legacy_confidence),
            ('min_classification_confidence', self.min_classification_confidence),
        ]:
            if not 0.0 <= value <= 1.0:
                errors.append(f"{name} must be in range [0, 1], got {value}")
        
        # Validate IoU thresholds
        for name, value in [
            ('nms_iou_threshold', self.nms_iou_threshold),
            ('merge_iou_threshold', self.merge_iou_threshold),
        ]:
            if not 0.0 <= value <= 1.0:
                errors.append(f"{name} must be in range [0, 1], got {value}")
        
        # Validate quality score
        if not 0.0 <= self.min_quality_score <= 1.0:
            errors.append(f"min_quality_score must be in range [0, 1], got {self.min_quality_score}")
        
        # Validate fusion strategy
        valid_strategies = ['weighted_nms', 'voting', 'cascade']
        if self.fusion_strategy not in valid_strategies:
            errors.append(f"fusion_strategy must be one of {valid_strategies}, got '{self.fusion_strategy}'")
        
        # Validate detector weights
        if self.detector_weights:
            for detector, weight in self.detector_weights.items():
                if weight < 0.0 or weight > 1.0:
                    errors.append(f"detector weight for '{detector}' must be in range [0, 1], got {weight}")
        
        # Validate quality weights
        if self.quality_weights:
            total_weight = sum(self.quality_weights.values())
            if not 0.99 <= total_weight <= 1.01:  # Allow small floating point error
                errors.append(f"quality_weights must sum to 1.0, got {total_weight}")
            
            for dimension, weight in self.quality_weights.items():
                if weight < 0.0 or weight > 1.0:
                    errors.append(f"quality weight for '{dimension}' must be in range [0, 1], got {weight}")
        
        # Validate distance threshold
        if self.merge_distance_threshold < 0:
            errors.append(f"merge_distance_threshold must be non-negative, got {self.merge_distance_threshold}")
        
        # Validate max_workers
        if self.max_workers < 1:
            errors.append(f"max_workers must be at least 1, got {self.max_workers}")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'enable_doclayout': self.enable_doclayout,
            'enable_table_transformer': self.enable_table_transformer,
            'enable_legacy_detectron': self.enable_legacy_detectron,
            'doclayout_confidence': self.doclayout_confidence,
            'table_transformer_confidence': self.table_transformer_confidence,
            'legacy_confidence': self.legacy_confidence,
            'fusion_strategy': self.fusion_strategy,
            'detector_weights': self.detector_weights,
            'nms_iou_threshold': self.nms_iou_threshold,
            'merge_iou_threshold': self.merge_iou_threshold,
            'merge_distance_threshold': self.merge_distance_threshold,
            'enable_context_aware_merge': self.enable_context_aware_merge,
            'enable_boundary_refinement': self.enable_boundary_refinement,
            'min_quality_score': self.min_quality_score,
            'enable_anomaly_detection': self.enable_anomaly_detection,
            'quality_weights': self.quality_weights,
            'enable_hierarchical_classification': self.enable_hierarchical_classification,
            'min_classification_confidence': self.min_classification_confidence,
            'enable_parallel_detection': self.enable_parallel_detection,
            'enable_model_caching': self.enable_model_caching,
            'max_workers': self.max_workers,
            'use_legacy_pipeline': self.use_legacy_pipeline,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionConfig':
        """
        Create DetectionConfig from dictionary.
        
        Provides default values for missing keys to support backward compatibility.
        """
        return cls(
            enable_doclayout=data.get('enable_doclayout', True),
            enable_table_transformer=data.get('enable_table_transformer', True),
            enable_legacy_detectron=data.get('enable_legacy_detectron', False),
            doclayout_confidence=data.get('doclayout_confidence', 0.35),
            table_transformer_confidence=data.get('table_transformer_confidence', 0.75),
            legacy_confidence=data.get('legacy_confidence', 0.5),
            fusion_strategy=data.get('fusion_strategy', 'weighted_nms'),
            detector_weights=data.get('detector_weights', {
                'doclayout': 0.4,
                'table_transformer': 0.4,
                'legacy': 0.2
            }),
            nms_iou_threshold=data.get('nms_iou_threshold', 0.5),
            merge_iou_threshold=data.get('merge_iou_threshold', 0.3),
            merge_distance_threshold=data.get('merge_distance_threshold', 50),
            enable_context_aware_merge=data.get('enable_context_aware_merge', True),
            enable_boundary_refinement=data.get('enable_boundary_refinement', True),
            min_quality_score=data.get('min_quality_score', 0.4),
            enable_anomaly_detection=data.get('enable_anomaly_detection', True),
            quality_weights=data.get('quality_weights', {
                'detection_confidence': 0.30,
                'content_completeness': 0.25,
                'boundary_precision': 0.20,
                'caption_match': 0.15,
                'position_reasonableness': 0.10
            }),
            enable_hierarchical_classification=data.get('enable_hierarchical_classification', True),
            min_classification_confidence=data.get('min_classification_confidence', 0.5),
            enable_parallel_detection=data.get('enable_parallel_detection', True),
            enable_model_caching=data.get('enable_model_caching', True),
            max_workers=data.get('max_workers', 4),
            use_legacy_pipeline=data.get('use_legacy_pipeline', False),
        )
    
    @classmethod
    def from_json_file(cls, filepath: str) -> 'DetectionConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to JSON configuration file
            
        Returns:
            DetectionConfig instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid or configuration is invalid
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Extract v1.7 specific config if it exists, otherwise use root
            if 'v17_detection' in data:
                config_data = data['v17_detection']
            else:
                config_data = data
            
            return cls.from_dict(config_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")
    
    def save_to_json_file(self, filepath: str):
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to output JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def get_default_config() -> DetectionConfig:
    """
    Get default detection configuration.
    
    Returns:
        DetectionConfig with default values
    """
    return DetectionConfig()


def validate_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize a configuration dictionary.
    
    This function creates a DetectionConfig from the dictionary,
    which triggers validation, then returns the normalized dictionary.
    
    Args:
        config_dict: Configuration dictionary to validate
        
    Returns:
        Validated and normalized configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    config = DetectionConfig.from_dict(config_dict)
    return config.to_dict()

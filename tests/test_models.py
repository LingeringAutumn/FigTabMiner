#!/usr/bin/env python3
"""
Unit tests for core data models.

Tests the basic functionality of Detection, ClassificationResult,
QualityScore, AnalysisResult, EvaluationMetrics, and DetectionConfig.
"""

import pytest
import json
import tempfile
from pathlib import Path

from src.figtabminer.models import (
    Detection,
    ClassificationResult,
    QualityScore,
    AnalysisResult,
    EvaluationMetrics,
    DetectionConfig,
    get_default_config,
    validate_config,
)


class TestDetection:
    """Test Detection data model."""
    
    def test_basic_creation(self):
        """Test creating a basic Detection."""
        det = Detection(
            bbox=[10.0, 20.0, 100.0, 200.0],
            type='figure',
            score=0.95,
            detector='doclayout'
        )
        
        assert det.bbox == [10.0, 20.0, 100.0, 200.0]
        assert det.type == 'figure'
        assert det.score == 0.95
        assert det.detector == 'doclayout'
        assert det.label == 'figure'  # Auto-set from type
    
    def test_bbox_validation(self):
        """Test that bbox must have exactly 4 coordinates."""
        with pytest.raises(ValueError, match="must have exactly 4 coordinates"):
            Detection(
                bbox=[10.0, 20.0, 100.0],  # Only 3 coordinates
                type='figure',
                score=0.95,
                detector='doclayout'
            )
    
    def test_type_validation(self):
        """Test that type must be 'figure' or 'table'."""
        with pytest.raises(ValueError, match="must be 'figure' or 'table'"):
            Detection(
                bbox=[10.0, 20.0, 100.0, 200.0],
                type='invalid',
                score=0.95,
                detector='doclayout'
            )
    
    def test_score_validation(self):
        """Test that score must be in range [0, 1]."""
        with pytest.raises(ValueError, match="must be in range"):
            Detection(
                bbox=[10.0, 20.0, 100.0, 200.0],
                type='figure',
                score=1.5,  # Invalid score
                detector='doclayout'
            )
    
    def test_area_calculation(self):
        """Test area calculation."""
        det = Detection(
            bbox=[10.0, 20.0, 100.0, 200.0],
            type='figure',
            score=0.95,
            detector='doclayout'
        )
        
        expected_area = (100.0 - 10.0) * (200.0 - 20.0)
        assert det.area() == expected_area
    
    def test_iou_calculation(self):
        """Test IoU calculation between two detections."""
        det1 = Detection(
            bbox=[0.0, 0.0, 100.0, 100.0],
            type='figure',
            score=0.95,
            detector='doclayout'
        )
        
        det2 = Detection(
            bbox=[50.0, 50.0, 150.0, 150.0],
            type='figure',
            score=0.90,
            detector='table_transformer'
        )
        
        iou = det1.iou(det2)
        
        # Calculate expected IoU
        # Intersection: [50, 50, 100, 100] = 50 * 50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        # IoU: 2500 / 17500 = 0.142857...
        assert abs(iou - 0.142857) < 0.001
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        det = Detection(
            bbox=[10.0, 20.0, 100.0, 200.0],
            type='figure',
            score=0.95,
            detector='doclayout',
            chart_type='bar_chart',
            chart_confidence=0.85
        )
        
        # Convert to dict
        det_dict = det.to_dict()
        
        # Convert back to Detection
        det2 = Detection.from_dict(det_dict)
        
        assert det2.bbox == det.bbox
        assert det2.type == det.type
        assert det2.score == det.score
        assert det2.detector == det.detector
        assert det2.chart_type == det.chart_type
        assert det2.chart_confidence == det.chart_confidence


class TestClassificationResult:
    """Test ClassificationResult data model."""
    
    def test_basic_creation(self):
        """Test creating a basic ClassificationResult."""
        result = ClassificationResult(
            chart_type='bar_chart',
            confidence=0.85
        )
        
        assert result.chart_type == 'bar_chart'
        assert result.confidence == 0.85
    
    def test_confidence_validation(self):
        """Test that confidence must be in range [0, 1]."""
        with pytest.raises(ValueError, match="must be in range"):
            ClassificationResult(
                chart_type='bar_chart',
                confidence=1.5
            )
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        result = ClassificationResult(
            chart_type='bar_chart',
            confidence=0.85,
            sub_type='grouped_bar',
            matched_keywords=['bar', 'chart']
        )
        
        result_dict = result.to_dict()
        result2 = ClassificationResult.from_dict(result_dict)
        
        assert result2.chart_type == result.chart_type
        assert result2.confidence == result.confidence
        assert result2.sub_type == result.sub_type
        assert result2.matched_keywords == result.matched_keywords


class TestQualityScore:
    """Test QualityScore data model."""
    
    def test_basic_creation(self):
        """Test creating a basic QualityScore."""
        score = QualityScore(
            overall_score=0.75,
            dimension_scores={
                'detection_confidence': 0.8,
                'content_completeness': 0.7
            }
        )
        
        assert score.overall_score == 0.75
        assert score.dimension_scores['detection_confidence'] == 0.8
    
    def test_overall_score_validation(self):
        """Test that overall_score must be in range [0, 1]."""
        with pytest.raises(ValueError, match="must be in range"):
            QualityScore(overall_score=1.5)
    
    def test_dimension_score_validation(self):
        """Test that dimension scores must be in range [0, 1]."""
        with pytest.raises(ValueError, match="must be in range"):
            QualityScore(
                overall_score=0.75,
                dimension_scores={'bad_dimension': 2.0}
            )


class TestAnalysisResult:
    """Test AnalysisResult data model."""
    
    def test_basic_creation(self):
        """Test creating a basic AnalysisResult."""
        result = AnalysisResult(
            subtype='grouped_bar',
            subtype_confidence=0.85,
            conditions=['temperature: 25C'],
            materials=['silicon', 'gold']
        )
        
        assert result.subtype == 'grouped_bar'
        assert result.subtype_confidence == 0.85
        assert len(result.conditions) == 1
        assert len(result.materials) == 2
    
    def test_confidence_validation(self):
        """Test that subtype_confidence must be in range [0, 1]."""
        with pytest.raises(ValueError, match="must be in range"):
            AnalysisResult(
                subtype='grouped_bar',
                subtype_confidence=1.5
            )


class TestEvaluationMetrics:
    """Test EvaluationMetrics data model."""
    
    def test_basic_creation(self):
        """Test creating basic EvaluationMetrics."""
        metrics = EvaluationMetrics(
            precision=0.95,
            recall=0.90,
            f1_score=0.925,
            mean_iou=0.87,
            total_predictions=20,
            total_ground_truth=18
        )
        
        assert metrics.precision == 0.95
        assert metrics.recall == 0.90
        assert metrics.f1_score == 0.925
        assert metrics.mean_iou == 0.87
    
    def test_metric_validation(self):
        """Test that metrics must be in range [0, 1]."""
        with pytest.raises(ValueError, match="must be in range"):
            EvaluationMetrics(
                precision=1.5,  # Invalid
                recall=0.90,
                f1_score=0.925,
                mean_iou=0.87
            )
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        det1 = Detection(
            bbox=[0.0, 0.0, 100.0, 100.0],
            type='figure',
            score=0.95,
            detector='doclayout'
        )
        
        det2 = Detection(
            bbox=[10.0, 10.0, 110.0, 110.0],
            type='figure',
            score=0.90,
            detector='ground_truth'
        )
        
        metrics = EvaluationMetrics(
            precision=0.95,
            recall=0.90,
            f1_score=0.925,
            mean_iou=0.87,
            true_positives=[(det1, det2)],
            false_positives=[],
            false_negatives=[],
            total_predictions=1,
            total_ground_truth=1
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict['precision'] == 0.95
        assert metrics_dict['tp_count'] == 1
        assert metrics_dict['fp_count'] == 0
        assert metrics_dict['fn_count'] == 0


class TestDetectionConfig:
    """Test DetectionConfig data model."""
    
    def test_default_creation(self):
        """Test creating a DetectionConfig with default values."""
        config = DetectionConfig()
        
        assert config.enable_doclayout is True
        assert config.doclayout_confidence == 0.35
        assert config.fusion_strategy == 'weighted_nms'
        assert config.min_quality_score == 0.4
    
    def test_confidence_validation(self):
        """Test that confidence thresholds must be in range [0, 1]."""
        with pytest.raises(ValueError, match="must be in range"):
            DetectionConfig(doclayout_confidence=1.5)
    
    def test_fusion_strategy_validation(self):
        """Test that fusion_strategy must be valid."""
        with pytest.raises(ValueError, match="must be one of"):
            DetectionConfig(fusion_strategy='invalid_strategy')
    
    def test_detector_weights_validation(self):
        """Test that detector weights must be in range [0, 1]."""
        with pytest.raises(ValueError, match="must be in range"):
            DetectionConfig(detector_weights={'doclayout': 1.5})
    
    def test_quality_weights_validation(self):
        """Test that quality weights must sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            DetectionConfig(quality_weights={
                'detection_confidence': 0.5,
                'content_completeness': 0.3
                # Sum is 0.8, not 1.0
            })
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        config = DetectionConfig(
            enable_doclayout=True,
            doclayout_confidence=0.4,
            fusion_strategy='voting'
        )
        
        config_dict = config.to_dict()
        config2 = DetectionConfig.from_dict(config_dict)
        
        assert config2.enable_doclayout == config.enable_doclayout
        assert config2.doclayout_confidence == config.doclayout_confidence
        assert config2.fusion_strategy == config.fusion_strategy
    
    def test_from_dict_with_missing_keys(self):
        """Test that from_dict provides defaults for missing keys."""
        # Minimal config dict
        config_dict = {
            'doclayout_confidence': 0.4
        }
        
        config = DetectionConfig.from_dict(config_dict)
        
        # Should have default values for missing keys
        assert config.doclayout_confidence == 0.4
        assert config.enable_doclayout is True  # Default
        assert config.fusion_strategy == 'weighted_nms'  # Default
    
    def test_json_file_operations(self):
        """Test loading and saving configuration from/to JSON file."""
        config = DetectionConfig(
            enable_doclayout=True,
            doclayout_confidence=0.4,
            fusion_strategy='voting'
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save_to_json_file(temp_path)
            
            # Load from file
            config2 = DetectionConfig.from_json_file(temp_path)
            
            assert config2.enable_doclayout == config.enable_doclayout
            assert config2.doclayout_confidence == config.doclayout_confidence
            assert config2.fusion_strategy == config.fusion_strategy
        finally:
            # Clean up
            Path(temp_path).unlink()
    
    def test_load_from_nested_json(self):
        """Test loading config from JSON with v17_detection key."""
        # Create a temporary JSON file with nested structure
        config_data = {
            "some_other_config": "value",
            "v17_detection": {
                "enable_doclayout": False,
                "doclayout_confidence": 0.5
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = DetectionConfig.from_json_file(temp_path)
            
            # Should load from v17_detection section
            assert config.enable_doclayout is False
            assert config.doclayout_confidence == 0.5
        finally:
            Path(temp_path).unlink()


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_get_default_config(self):
        """Test get_default_config function."""
        config = get_default_config()
        
        assert isinstance(config, DetectionConfig)
        assert config.enable_doclayout is True
        assert config.doclayout_confidence == 0.35
    
    def test_validate_config_valid(self):
        """Test validate_config with valid configuration."""
        config_dict = {
            'enable_doclayout': True,
            'doclayout_confidence': 0.4
        }
        
        validated = validate_config(config_dict)
        
        assert isinstance(validated, dict)
        assert validated['enable_doclayout'] is True
        assert validated['doclayout_confidence'] == 0.4
    
    def test_validate_config_invalid(self):
        """Test validate_config with invalid configuration."""
        config_dict = {
            'doclayout_confidence': 1.5  # Invalid
        }
        
        with pytest.raises(ValueError, match="must be in range"):
            validate_config(config_dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

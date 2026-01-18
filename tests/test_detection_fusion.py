#!/usr/bin/env python3
"""
Unit tests for IntelligentDetectionFusion module.

Tests the basic structure, initialization, and detector weight management
of the detection fusion module.
"""

import pytest
from src.figtabminer.detection_fusion import IntelligentDetectionFusion
from src.figtabminer.models import Detection, DetectionConfig, get_default_config


class TestIntelligentDetectionFusionInit:
    """Test initialization and configuration loading."""
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        fusion = IntelligentDetectionFusion()
        
        assert fusion.config is not None
        assert fusion.fusion_strategy == 'weighted_nms'
        assert 'doclayout' in fusion.detector_weights
        assert 'table_transformer' in fusion.detector_weights
        assert fusion.enable_adaptive_threshold is True
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = DetectionConfig(
            fusion_strategy='voting',
            detector_weights={
                'doclayout': 0.5,
                'table_transformer': 0.5,
            },
            doclayout_confidence=0.4,
            table_transformer_confidence=0.8,
        )
        
        fusion = IntelligentDetectionFusion(config)
        
        assert fusion.fusion_strategy == 'voting'
        assert fusion.detector_weights['doclayout'] == 0.5
        assert fusion.detector_weights['table_transformer'] == 0.5
        assert fusion.confidence_thresholds['doclayout'] == 0.4
        assert fusion.confidence_thresholds['table_transformer'] == 0.8
    
    def test_confidence_thresholds_loaded(self):
        """Test that confidence thresholds are properly loaded."""
        config = DetectionConfig(
            doclayout_confidence=0.3,
            table_transformer_confidence=0.7,
            legacy_confidence=0.5,
        )
        
        fusion = IntelligentDetectionFusion(config)
        
        assert fusion.confidence_thresholds['doclayout'] == 0.3
        assert fusion.confidence_thresholds['table_transformer'] == 0.7
        assert fusion.confidence_thresholds['legacy'] == 0.5
        assert fusion.confidence_thresholds['publaynet'] == 0.5  # Alias


class TestDetectorWeightManagement:
    """Test detector weight management methods."""
    
    def test_get_detector_weight_exact_match(self):
        """Test getting weight for exact detector name match."""
        fusion = IntelligentDetectionFusion()
        
        weight = fusion.get_detector_weight('doclayout')
        assert 0.0 <= weight <= 1.0
        assert weight == fusion.detector_weights['doclayout']
    
    def test_get_detector_weight_case_insensitive(self):
        """Test that detector name matching is case-insensitive."""
        fusion = IntelligentDetectionFusion()
        
        weight_lower = fusion.get_detector_weight('doclayout')
        weight_upper = fusion.get_detector_weight('DocLayout')
        weight_mixed = fusion.get_detector_weight('DocLayout')
        
        assert weight_lower == weight_upper == weight_mixed
    
    def test_get_detector_weight_with_alias(self):
        """Test getting weight using detector aliases."""
        fusion = IntelligentDetectionFusion()
        
        # 'publaynet' should map to 'legacy'
        weight_legacy = fusion.get_detector_weight('legacy')
        weight_publaynet = fusion.get_detector_weight('publaynet')
        
        assert weight_publaynet == weight_legacy
    
    def test_get_detector_weight_unknown_detector(self):
        """Test getting weight for unknown detector returns default."""
        fusion = IntelligentDetectionFusion()
        
        weight = fusion.get_detector_weight('unknown_detector')
        assert weight == 0.5  # Default weight
    
    def test_update_detector_weights(self):
        """Test updating detector weights dynamically."""
        fusion = IntelligentDetectionFusion()
        
        original_weight = fusion.detector_weights['doclayout']
        
        fusion.update_detector_weights({'doclayout': 0.6})
        
        assert fusion.detector_weights['doclayout'] == 0.6
        assert fusion.detector_weights['doclayout'] != original_weight
    
    def test_update_detector_weights_invalid_range(self):
        """Test that invalid weights are rejected."""
        fusion = IntelligentDetectionFusion()
        
        original_weight = fusion.detector_weights['doclayout']
        
        # Try to set invalid weight (should be ignored)
        fusion.update_detector_weights({'doclayout': 1.5})
        
        # Weight should remain unchanged
        assert fusion.detector_weights['doclayout'] == original_weight
    
    def test_get_detector_reliability(self):
        """Test getting detector reliability scores."""
        fusion = IntelligentDetectionFusion()
        
        reliability = fusion.get_detector_reliability('doclayout')
        assert 0.0 <= reliability <= 1.0
        assert reliability == fusion.detector_reliability['doclayout']
    
    def test_update_detector_reliability(self):
        """Test updating detector reliability scores."""
        fusion = IntelligentDetectionFusion()
        
        fusion.update_detector_reliability({'doclayout': 0.95})
        
        assert fusion.detector_reliability['doclayout'] == 0.95


class TestConfidenceFiltering:
    """Test confidence-based filtering."""
    
    def test_filter_by_confidence_empty_list(self):
        """Test filtering empty detection list."""
        fusion = IntelligentDetectionFusion()
        
        filtered = fusion.filter_by_confidence([])
        assert filtered == []
    
    def test_filter_by_confidence_above_threshold(self):
        """Test that detections above threshold are kept."""
        config = DetectionConfig(doclayout_confidence=0.5)
        fusion = IntelligentDetectionFusion(config)
        
        detections = [
            Detection(
                bbox=[0, 0, 100, 100],
                type='figure',
                score=0.8,
                detector='doclayout'
            )
        ]
        
        filtered = fusion.filter_by_confidence(detections)
        assert len(filtered) == 1
        assert filtered[0].score == 0.8
    
    def test_filter_by_confidence_below_threshold(self):
        """Test that detections below threshold are filtered out."""
        config = DetectionConfig(doclayout_confidence=0.5)
        fusion = IntelligentDetectionFusion(config)
        
        detections = [
            Detection(
                bbox=[0, 0, 100, 100],
                type='figure',
                score=0.3,
                detector='doclayout'
            )
        ]
        
        filtered = fusion.filter_by_confidence(detections)
        assert len(filtered) == 0
    
    def test_filter_by_confidence_mixed(self):
        """Test filtering with mixed confidence scores."""
        config = DetectionConfig(
            doclayout_confidence=0.5,
            table_transformer_confidence=0.7
        )
        fusion = IntelligentDetectionFusion(config)
        
        detections = [
            Detection(bbox=[0, 0, 100, 100], type='figure', score=0.8, detector='doclayout'),
            Detection(bbox=[100, 0, 200, 100], type='figure', score=0.3, detector='doclayout'),
            Detection(bbox=[0, 100, 100, 200], type='table', score=0.9, detector='table_transformer'),
            Detection(bbox=[100, 100, 200, 200], type='table', score=0.6, detector='table_transformer'),
        ]
        
        filtered = fusion.filter_by_confidence(detections)
        
        # Should keep: doclayout with 0.8, table_transformer with 0.9
        # Should filter: doclayout with 0.3, table_transformer with 0.6
        assert len(filtered) == 2
        assert all(det.score >= 0.7 for det in filtered)


class TestWeightedScoring:
    """Test weighted score calculation."""
    
    def test_calculate_weighted_score(self):
        """Test weighted score calculation."""
        fusion = IntelligentDetectionFusion()
        
        detection = Detection(
            bbox=[0, 0, 100, 100],
            type='figure',
            score=0.8,
            detector='doclayout'
        )
        
        weighted_score = fusion.calculate_weighted_score(detection)
        
        # Weighted score should be: score * weight * reliability
        expected = 0.8 * fusion.detector_weights['doclayout'] * fusion.detector_reliability['doclayout']
        
        assert abs(weighted_score - expected) < 0.001
    
    def test_calculate_weighted_score_range(self):
        """Test that weighted score is in valid range."""
        fusion = IntelligentDetectionFusion()
        
        detection = Detection(
            bbox=[0, 0, 100, 100],
            type='figure',
            score=0.9,
            detector='doclayout'
        )
        
        weighted_score = fusion.calculate_weighted_score(detection)
        
        assert 0.0 <= weighted_score <= 1.0


class TestDetectorPriority:
    """Test detector priority system."""
    
    def test_get_detector_priority_table_transformer_for_tables(self):
        """Test that Table Transformer has high priority for tables."""
        fusion = IntelligentDetectionFusion()
        
        priority = fusion.get_detector_priority('table_transformer', 'table')
        
        # Table Transformer should have highest priority for tables
        assert priority >= 3
    
    def test_get_detector_priority_doclayout_for_figures(self):
        """Test that DocLayout has good priority for figures."""
        fusion = IntelligentDetectionFusion()
        
        priority = fusion.get_detector_priority('doclayout', 'figure')
        
        # DocLayout should have decent priority for figures
        assert priority >= 2
    
    def test_get_detector_priority_unknown_detector(self):
        """Test that unknown detectors get default priority."""
        fusion = IntelligentDetectionFusion()
        
        priority = fusion.get_detector_priority('unknown', 'figure')
        
        assert priority == 1  # Default priority


class TestStringRepresentation:
    """Test string representation."""
    
    def test_repr(self):
        """Test __repr__ method."""
        fusion = IntelligentDetectionFusion()
        
        repr_str = repr(fusion)
        
        assert 'IntelligentDetectionFusion' in repr_str
        assert 'strategy' in repr_str
        assert 'weights' in repr_str
        assert 'adaptive_threshold' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

#!/usr/bin/env python3
"""
Unit tests for TextFalsePositiveFilter module.

Tests the enhanced text false positive filtering functionality including:
- Position heuristics
- OCR text pattern recognition
- Continuous text line detection
- Content feature filtering
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os

from src.figtabminer.text_false_positive_filter import TextFalsePositiveFilter
from src.figtabminer.models import Detection


class TestTextFalsePositiveFilter:
    """Test suite for TextFalsePositiveFilter"""
    
    @pytest.fixture
    def filter_instance(self):
        """Create a filter instance with default settings"""
        return TextFalsePositiveFilter()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        # Create a 1000x800 white image
        image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
        return image
    
    @pytest.fixture
    def temp_image_path(self, sample_image):
        """Save sample image to temporary file"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            cv2.imwrite(f.name, sample_image)
            yield f.name
        # Cleanup
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    def test_filter_empty_detections(self, filter_instance, temp_image_path):
        """Test filtering with empty detection list"""
        kept, removed = filter_instance.filter([], temp_image_path)
        assert len(kept) == 0
        assert len(removed) == 0
    
    def test_filter_non_table_detections(self, filter_instance, temp_image_path):
        """Test that non-table detections are kept unchanged"""
        detections = [
            Detection(
                bbox=[100, 100, 300, 300],
                type='figure',
                score=0.9,
                detector='doclayout'
            )
        ]
        kept, removed = filter_instance.filter(detections, temp_image_path)
        assert len(kept) == 1
        assert len(removed) == 0
        assert kept[0].type == 'figure'
    
    def test_filter_low_confidence_table(self, filter_instance, temp_image_path):
        """Test that low confidence tables are filtered"""
        detections = [
            Detection(
                bbox=[100, 100, 300, 300],
                type='table',
                score=0.5,  # Below default threshold of 0.7
                detector='doclayout'
            )
        ]
        kept, removed = filter_instance.filter(detections, temp_image_path)
        assert len(kept) == 0
        assert len(removed) == 1
    
    def test_filter_high_confidence_table(self, filter_instance, temp_image_path):
        """Test that high confidence tables pass initial filter"""
        detections = [
            Detection(
                bbox=[100, 100, 300, 300],
                type='table',
                score=0.9,
                detector='doclayout'
            )
        ]
        kept, removed = filter_instance.filter(detections, temp_image_path)
        # Should pass confidence check (may be filtered by other checks)
        assert len(kept) + len(removed) == 1
    
    def test_check_position_heuristics_header(self, filter_instance):
        """Test position heuristics for header region"""
        detection = Detection(
            bbox=[100, 20, 300, 40],  # Small box in header region
            type='table',
            score=0.8,
            detector='doclayout'
        )
        image_shape = (1000, 800, 3)
        
        is_fp, reason = filter_instance.check_position_heuristics(detection, image_shape)
        assert is_fp is True
        assert '页眉' in reason
    
    def test_check_position_heuristics_footer(self, filter_instance):
        """Test position heuristics for footer region"""
        detection = Detection(
            bbox=[100, 950, 300, 980],  # Small box in footer region
            type='table',
            score=0.8,
            detector='doclayout'
        )
        image_shape = (1000, 800, 3)
        
        is_fp, reason = filter_instance.check_position_heuristics(detection, image_shape)
        assert is_fp is True
        assert '页脚' in reason
    
    def test_check_position_heuristics_left_edge(self, filter_instance):
        """Test position heuristics for left edge"""
        detection = Detection(
            bbox=[10, 400, 50, 500],  # Small box at left edge
            type='table',
            score=0.8,
            detector='doclayout'
        )
        image_shape = (1000, 800, 3)
        
        is_fp, reason = filter_instance.check_position_heuristics(detection, image_shape)
        assert is_fp is True
        assert '左边缘' in reason
    
    def test_check_position_heuristics_center(self, filter_instance):
        """Test position heuristics for center region (should pass)"""
        detection = Detection(
            bbox=[200, 400, 600, 600],  # Large box in center
            type='table',
            score=0.8,
            detector='doclayout'
        )
        image_shape = (1000, 800, 3)
        
        is_fp, reason = filter_instance.check_position_heuristics(detection, image_shape)
        assert is_fp is False
    
    def test_detect_text_pattern_no_ocr(self, filter_instance, sample_image):
        """Test text pattern detection when OCR is not available"""
        detection = Detection(
            bbox=[100, 100, 300, 200],
            type='table',
            score=0.8,
            detector='doclayout'
        )
        
        # Should gracefully handle missing OCR
        is_fp, reason = filter_instance.detect_text_pattern(detection, sample_image)
        # Should return False when OCR is not available
        assert is_fp is False
    
    def test_detect_continuous_text_lines_empty_region(self, filter_instance, sample_image):
        """Test continuous text line detection on empty region"""
        detection = Detection(
            bbox=[100, 100, 300, 200],
            type='table',
            score=0.8,
            detector='doclayout'
        )
        
        # Empty white region should not be detected as text lines
        is_fp, reason = filter_instance.detect_continuous_text_lines(detection, sample_image)
        assert is_fp is False
    
    def test_is_text_false_positive_empty_region(self, filter_instance, sample_image):
        """Test content feature filtering on empty region"""
        detection = Detection(
            bbox=[100, 100, 300, 200],
            type='table',
            score=0.8,
            detector='doclayout'
        )
        
        # Empty white region should have low ink density
        is_fp, reason = filter_instance.is_text_false_positive(detection, sample_image)
        assert is_fp is False
    
    def test_filter_invalid_image_path(self, filter_instance):
        """Test filtering with invalid image path"""
        detections = [
            Detection(
                bbox=[100, 100, 300, 300],
                type='table',
                score=0.9,
                detector='doclayout'
            )
        ]
        
        kept, removed = filter_instance.filter(detections, 'nonexistent.png')
        # Should return original detections when image cannot be loaded
        assert len(kept) == 1
        assert len(removed) == 0
    
    def test_filter_with_page_text(self, filter_instance, temp_image_path):
        """Test filtering with page_text parameter"""
        detections = [
            Detection(
                bbox=[100, 100, 300, 300],
                type='table',
                score=0.9,
                detector='doclayout'
            )
        ]
        
        page_text = "Some sample page text"
        kept, removed = filter_instance.filter(detections, temp_image_path, page_text)
        # Should not crash with page_text parameter
        assert len(kept) + len(removed) == 1
    
    def test_filter_configuration_options(self):
        """Test filter with different configuration options"""
        # Test with all features disabled
        filter_disabled = TextFalsePositiveFilter(
            enable_position_heuristics=False,
            enable_ocr_pattern_matching=False,
            enable_text_line_detection=False
        )
        assert filter_disabled.enable_position_heuristics is False
        assert filter_disabled.enable_ocr_pattern_matching is False
        assert filter_disabled.enable_text_line_detection is False
        
        # Test with custom thresholds
        filter_custom = TextFalsePositiveFilter(
            table_confidence_threshold=0.5,
            text_density_threshold=0.1,
            min_table_structure_score=100,
            header_margin=0.15,
            footer_margin=0.15,
            edge_margin=0.08
        )
        assert filter_custom.table_confidence_threshold == 0.5
        assert filter_custom.header_margin == 0.15


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

#!/usr/bin/env python3
"""
Unit tests for Enhanced Quality Assessor (v1.7).

Tests comprehensive quality assessment, anomaly detection,
and quality-based filtering.
"""

import sys
import os
import numpy as np
import cv2
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from figtabminer.enhanced_quality_assessor import (
    EnhancedQualityAssessor,
    Anomaly,
    assess_detection_quality,
    filter_detections_by_quality
)


class TestQualityAssessment:
    """Test comprehensive quality assessment."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.assessor = EnhancedQualityAssessor()
        # Create a test page image
        self.page_image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    
    def test_assess_good_quality(self):
        """Test assessment of good quality detection."""
        # Create a detection with good properties
        detection = {
            'bbox': [100, 100, 400, 400],
            'type': 'figure',
            'score': 0.9
        }
        
        # Draw content in the bbox
        cv2.rectangle(self.page_image, (100, 100), (400, 400), (0, 0, 0), -1)
        
        result = self.assessor.assess_comprehensive(detection, self.page_image)
        
        assert 'overall_score' in result
        assert 'dimension_scores' in result
        assert 'issues' in result
        assert 'recommendations' in result
        
        # Good quality should have high score
        assert result['overall_score'] > 0.5
    
    def test_assess_poor_quality(self):
        """Test assessment of poor quality detection."""
        # Create a detection with poor properties
        detection = {
            'bbox': [100, 100, 800, 900],  # Very large, mostly empty
            'type': 'figure',
            'score': 0.3  # Low confidence
        }
        
        result = self.assessor.assess_comprehensive(detection, self.page_image)
        
        # Poor quality should have low score
        assert result['overall_score'] < 0.5
        
        # Should have issues identified
        assert len(result['issues']) > 0
    
    def test_dimension_scores(self):
        """Test that all dimension scores are present."""
        detection = {
            'bbox': [100, 100, 400, 400],
            'type': 'figure',
            'score': 0.8
        }
        
        result = self.assessor.assess_comprehensive(detection, self.page_image)
        
        expected_dimensions = [
            'detection_confidence',
            'content_completeness',
            'boundary_precision',
            'caption_match',
            'position_reasonableness'
        ]
        
        for dim in expected_dimensions:
            assert dim in result['dimension_scores']
            assert 0.0 <= result['dimension_scores'][dim] <= 1.0
    
    def test_with_captions(self):
        """Test assessment with caption matching."""
        detection = {
            'bbox': [100, 100, 400, 400],
            'type': 'figure',
            'score': 0.8
        }
        
        # Caption near the detection
        captions = [
            {'bbox': [100, 420, 400, 450], 'text': 'Figure 1: Test'}
        ]
        
        result = self.assessor.assess_comprehensive(
            detection, self.page_image, captions=captions
        )
        
        # Should have good caption match
        assert result['dimension_scores']['caption_match'] > 0.5


class TestAnomalyDetection:
    """Test anomaly detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.assessor = EnhancedQualityAssessor()
        self.page_image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    
    def test_detect_oversized(self):
        """Test detection of oversized bboxes."""
        detections = [
            {
                'bbox': [10, 10, 790, 990],  # Covers >90% of page
                'type': 'figure',
                'score': 0.8
            }
        ]
        
        anomalies = self.assessor.detect_anomalies(detections, self.page_image)
        
        assert len(anomalies) > 0
        assert any(a.anomaly_type == 'oversized' for a in anomalies)
    
    def test_detect_undersized(self):
        """Test detection of undersized bboxes."""
        detections = [
            {
                'bbox': [100, 100, 105, 105],  # Very small
                'type': 'figure',
                'score': 0.8
            }
        ]
        
        anomalies = self.assessor.detect_anomalies(detections, self.page_image)
        
        assert len(anomalies) > 0
        assert any(a.anomaly_type == 'undersized' for a in anomalies)
    
    def test_detect_extreme_aspect_ratio(self):
        """Test detection of extreme aspect ratios."""
        detections = [
            {
                'bbox': [100, 100, 700, 120],  # Very wide
                'type': 'figure',
                'score': 0.8
            }
        ]
        
        anomalies = self.assessor.detect_anomalies(detections, self.page_image)
        
        assert len(anomalies) > 0
        assert any(a.anomaly_type == 'extreme_aspect_ratio' for a in anomalies)
    
    def test_detect_sparse_content(self):
        """Test detection of sparse content."""
        detections = [
            {
                'bbox': [100, 100, 400, 400],  # Large but empty
                'type': 'figure',
                'score': 0.8
            }
        ]
        
        # Don't draw anything - leave it white
        anomalies = self.assessor.detect_anomalies(detections, self.page_image)
        
        assert len(anomalies) > 0
        assert any(a.anomaly_type == 'sparse_content' for a in anomalies)
    
    def test_no_anomalies(self):
        """Test that good detections have no anomalies."""
        detections = [
            {
                'bbox': [100, 100, 400, 400],  # Reasonable size
                'type': 'figure',
                'score': 0.8
            }
        ]
        
        # Draw content
        cv2.rectangle(self.page_image, (100, 100), (400, 400), (0, 0, 0), -1)
        
        anomalies = self.assessor.detect_anomalies(detections, self.page_image)
        
        # Should have no anomalies
        assert len(anomalies) == 0


class TestQualityFiltering:
    """Test quality-based filtering."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.assessor = EnhancedQualityAssessor()
        self.page_image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    
    def test_filter_by_quality(self):
        """Test filtering by quality score."""
        # Create mix of good and poor detections
        detections = [
            {
                'bbox': [100, 100, 400, 400],
                'type': 'figure',
                'score': 0.9  # Good
            },
            {
                'bbox': [100, 100, 800, 900],
                'type': 'figure',
                'score': 0.2  # Poor
            }
        ]
        
        # Draw content for first detection
        cv2.rectangle(self.page_image, (100, 100), (400, 400), (0, 0, 0), -1)
        
        passed, filtered = self.assessor.filter_by_quality(
            detections, self.page_image, min_score=0.5
        )
        
        assert len(passed) >= 1
        assert len(filtered) >= 1
    
    def test_filter_threshold(self):
        """Test that filtering respects threshold."""
        detections = [
            {'bbox': [100, 100, 400, 400], 'type': 'figure', 'score': 0.9},
            {'bbox': [100, 500, 400, 800], 'type': 'figure', 'score': 0.8}
        ]
        
        # Draw content for both
        cv2.rectangle(self.page_image, (100, 100), (400, 400), (0, 0, 0), -1)
        cv2.rectangle(self.page_image, (100, 500), (400, 800), (0, 0, 0), -1)
        
        # Filter with high threshold
        passed, filtered = self.assessor.filter_by_quality(
            detections, self.page_image, min_score=0.8
        )
        
        # All should pass with good content
        assert len(passed) == 2
    
    def test_filter_returns_both_lists(self):
        """Test that filter returns both passed and filtered lists."""
        detections = [
            {'bbox': [100, 100, 400, 400], 'type': 'figure', 'score': 0.9}
        ]
        
        result = self.assessor.filter_by_quality(detections, self.page_image)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        passed, filtered = result
        assert isinstance(passed, list)
        assert isinstance(filtered, list)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.page_image = np.ones((1000, 800, 3), dtype=np.uint8) * 255
    
    def test_assess_detection_quality(self):
        """Test convenience function for assessment."""
        detection = {
            'bbox': [100, 100, 400, 400],
            'type': 'figure',
            'score': 0.8
        }
        
        result = assess_detection_quality(detection, self.page_image)
        
        assert 'overall_score' in result
        assert isinstance(result['overall_score'], float)
    
    def test_filter_detections_by_quality(self):
        """Test convenience function for filtering."""
        detections = [
            {'bbox': [100, 100, 400, 400], 'type': 'figure', 'score': 0.9},
            {'bbox': [100, 500, 400, 800], 'type': 'figure', 'score': 0.8}
        ]
        
        filtered = filter_detections_by_quality(
            detections, self.page_image, min_score=0.5
        )
        
        assert isinstance(filtered, list)
        assert len(filtered) <= len(detections)


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("Enhanced Quality Assessor Test Suite (v1.7)")
    print("="*70)
    
    test_classes = [
        TestQualityAssessment,
        TestAnomalyDetection,
        TestQualityFiltering,
        TestConvenienceFunctions
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 70)
        
        # Get all test methods
        test_methods = [m for m in dir(test_class) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                # Create instance and run test
                instance = test_class()
                instance.setup_method()
                method = getattr(instance, method_name)
                method()
                
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except AssertionError as e:
                print(f"  ✗ {method_name}: {e}")
                failed_tests.append((test_class.__name__, method_name, str(e)))
            except Exception as e:
                print(f"  ✗ {method_name}: Unexpected error: {e}")
                failed_tests.append((test_class.__name__, method_name, str(e)))
    
    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
        return False
    else:
        print("\n✓ All tests passed!")
        return True


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

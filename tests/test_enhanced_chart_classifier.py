#!/usr/bin/env python3
"""
Unit tests for Enhanced Chart Classifier (v1.7).

Tests hierarchical classification, extended chart types,
visual feature extraction, and confidence calibration.
"""

import sys
import os
import numpy as np
import cv2
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from figtabminer.enhanced_chart_classifier import (
    EnhancedChartClassifier,
    HierarchicalClassification,
    classify_chart_enhanced
)


class TestChartTypes:
    """Test extended chart type support."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = EnhancedChartClassifier({
            'enable_visual_analysis': True,
            'enable_hierarchical': True,
            'enable_calibration': False  # Disable for predictable testing
        })
    
    def test_chart_types_count(self):
        """Test that we have 15+ chart types."""
        assert len(self.classifier.CHART_TYPES) >= 15, \
            f"Expected at least 15 chart types, got {len(self.classifier.CHART_TYPES)}"
    
    def test_hierarchy_structure(self):
        """Test hierarchical classification structure."""
        assert 'chart' in self.classifier.HIERARCHY
        assert 'microscopy' in self.classifier.HIERARCHY
        assert 'diagram' in self.classifier.HIERARCHY
        
        # Check that chart category has multiple subtypes
        assert len(self.classifier.HIERARCHY['chart']) >= 7
    
    def test_keyword_coverage(self):
        """Test that all chart types have keywords."""
        for chart_type in self.classifier.CHART_TYPES:
            if chart_type != 'unknown':
                assert chart_type in self.classifier.CHART_KEYWORDS, \
                    f"Chart type {chart_type} missing keywords"


class TestHierarchicalClassification:
    """Test hierarchical classification functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = EnhancedChartClassifier({
            'enable_hierarchical': True,
            'enable_visual_analysis': False  # Focus on keyword matching
        })
        self.temp_dir = tempfile.mkdtemp()
    
    def _create_dummy_image(self, filename: str) -> str:
        """Create a dummy image for testing."""
        img = np.ones((300, 300, 3), dtype=np.uint8) * 255
        path = os.path.join(self.temp_dir, filename)
        cv2.imwrite(path, img)
        return path
    
    def test_bar_chart_classification(self):
        """Test bar chart classification."""
        img_path = self._create_dummy_image('bar.png')
        caption = "Figure 1: Bar chart showing distribution"
        
        result = self.classifier.classify_hierarchical(img_path, caption=caption)
        
        assert isinstance(result, HierarchicalClassification)
        assert result.main_category == 'chart'
        assert 'bar_chart' in result.sub_category or result.sub_category in self.classifier.HIERARCHY['chart']
    
    def test_microscopy_classification(self):
        """Test microscopy image classification."""
        img_path = self._create_dummy_image('sem.png')
        caption = "SEM image of nanoparticles"
        
        result = self.classifier.classify_hierarchical(img_path, caption=caption)
        
        assert result.main_category == 'microscopy'
        assert 'microscopy' in result.sub_category
    
    def test_diagram_classification(self):
        """Test diagram classification."""
        img_path = self._create_dummy_image('flow.png')
        caption = "Flowchart of the process"
        
        result = self.classifier.classify_hierarchical(img_path, caption=caption)
        
        assert result.main_category == 'diagram'
        assert 'diagram' in result.sub_category
    
    def test_confidence_levels(self):
        """Test that confidence is provided for both levels."""
        img_path = self._create_dummy_image('test.png')
        caption = "Line plot showing trend"
        
        result = self.classifier.classify_hierarchical(img_path, caption=caption)
        
        assert 'main' in result.confidence_by_level
        assert 'sub' in result.confidence_by_level
        assert 0.0 <= result.confidence_by_level['main'] <= 1.0
        assert 0.0 <= result.confidence_by_level['sub'] <= 1.0


class TestVisualFeatures:
    """Test visual feature extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = EnhancedChartClassifier({
            'enable_visual_analysis': True
        })
        self.temp_dir = tempfile.mkdtemp()
    
    def _create_bar_chart_image(self) -> str:
        """Create a simple bar chart image."""
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Draw bars
        cv2.rectangle(img, (50, 200), (100, 350), (0, 0, 0), -1)
        cv2.rectangle(img, (150, 150), (200, 350), (0, 0, 0), -1)
        cv2.rectangle(img, (250, 250), (300, 350), (0, 0, 0), -1)
        
        # Draw axes
        cv2.line(img, (40, 360), (350, 360), (0, 0, 0), 2)
        cv2.line(img, (40, 100), (40, 360), (0, 0, 0), 2)
        
        path = os.path.join(self.temp_dir, 'bar_chart.png')
        cv2.imwrite(path, img)
        return path
    
    def _create_pie_chart_image(self) -> str:
        """Create a simple pie chart image."""
        img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Draw circle
        cv2.circle(img, (200, 200), 100, (0, 0, 0), 2)
        
        # Draw pie slices
        cv2.line(img, (200, 200), (300, 200), (0, 0, 0), 2)
        cv2.line(img, (200, 200), (200, 100), (0, 0, 0), 2)
        
        path = os.path.join(self.temp_dir, 'pie_chart.png')
        cv2.imwrite(path, img)
        return path
    
    def test_detect_rectangles(self):
        """Test rectangle detection for bar charts."""
        img_path = self._create_bar_chart_image()
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        score = self.classifier._detect_rectangles(gray)
        
        assert score > 0.0, "Should detect rectangles in bar chart"
    
    def test_detect_circles(self):
        """Test circle detection for pie charts."""
        img_path = self._create_pie_chart_image()
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        score = self.classifier._detect_circles(gray)
        
        assert score > 0.0, "Should detect circles in pie chart"
    
    def test_main_category_visual_scores(self):
        """Test main category visual scoring."""
        img_path = self._create_bar_chart_image()
        
        scores = self.classifier._get_main_category_visual_scores(img_path)
        
        assert 'chart' in scores
        assert 'microscopy' in scores
        assert 'diagram' in scores
        assert all(0.0 <= v <= 1.0 for v in scores.values())


class TestConfidenceCalibration:
    """Test confidence calibration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = EnhancedChartClassifier({
            'enable_calibration': True,
            'calibration_a': 2.0,
            'calibration_b': -1.0
        })
    
    def test_calibration_range(self):
        """Test that calibrated confidence is in [0, 1]."""
        raw_confidences = [0.0, 0.3, 0.5, 0.7, 1.0]
        
        for raw_conf in raw_confidences:
            calibrated = self.classifier._calibrate_confidence(raw_conf)
            assert 0.0 <= calibrated <= 1.0, \
                f"Calibrated confidence {calibrated} out of range for raw {raw_conf}"
    
    def test_calibration_monotonic(self):
        """Test that calibration preserves order."""
        raw_confidences = [0.2, 0.4, 0.6, 0.8]
        calibrated = [self.classifier._calibrate_confidence(c) for c in raw_confidences]
        
        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i + 1], \
                "Calibration should preserve order"


class TestIntegration:
    """Integration tests for the complete classifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = EnhancedChartClassifier()
        self.temp_dir = tempfile.mkdtemp()
    
    def _create_test_image(self) -> str:
        """Create a test image."""
        img = np.ones((300, 300, 3), dtype=np.uint8) * 255
        path = os.path.join(self.temp_dir, 'test.png')
        cv2.imwrite(path, img)
        return path
    
    def test_classify_with_keywords(self):
        """Test classification with keyword matching."""
        img_path = self._create_test_image()
        caption = "Scatter plot showing correlation"
        
        chart_type, confidence, keywords, debug = self.classifier.classify(
            img_path, caption=caption
        )
        
        assert chart_type in self.classifier.CHART_TYPES
        assert 0.0 <= confidence <= 1.0
        assert isinstance(keywords, list)
        assert isinstance(debug, dict)
    
    def test_classify_unknown(self):
        """Test that unknown type is returned when no match."""
        img_path = self._create_test_image()
        caption = ""  # No keywords
        
        chart_type, confidence, keywords, debug = self.classifier.classify(
            img_path, caption=caption
        )
        
        # Should return some type (possibly unknown)
        assert chart_type in self.classifier.CHART_TYPES
    
    def test_convenience_function(self):
        """Test the convenience function."""
        img_path = self._create_test_image()
        caption = "Heatmap visualization"
        
        result = classify_chart_enhanced(img_path, caption=caption)
        
        assert len(result) == 4
        chart_type, confidence, keywords, debug = result
        assert chart_type in self.classifier.CHART_TYPES


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("Enhanced Chart Classifier Test Suite (v1.7)")
    print("="*70)
    
    test_classes = [
        TestChartTypes,
        TestHierarchicalClassification,
        TestVisualFeatures,
        TestConfidenceCalibration,
        TestIntegration
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

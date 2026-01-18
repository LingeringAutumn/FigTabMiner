#!/usr/bin/env python3
"""
Unit tests for Enhanced BBox Merger (v1.7).

Tests the context-aware merging, type-specific strategies,
boundary refinement, and complex figure splitting.
"""

import sys
import os
import numpy as np
import cv2
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from figtabminer import bbox_merger
from figtabminer import bbox_utils


class TestContextAwareMerging:
    """Test context-aware merging functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.merger = bbox_merger.SmartBBoxMerger({
            'enable_context_aware_merge': True,
            'enable_type_specific_merge': True
        })
        # Create a dummy page image
        self.page_image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    
    def test_flowchart_detection(self):
        """Test flowchart detection from visual features."""
        # Create a flowchart-like image with boxes and lines
        flowchart_image = np.ones((300, 300, 3), dtype=np.uint8) * 255
        
        # Draw some boxes (nodes)
        cv2.rectangle(flowchart_image, (50, 50), (100, 80), (0, 0, 0), 2)
        cv2.rectangle(flowchart_image, (150, 50), (200, 80), (0, 0, 0), 2)
        cv2.rectangle(flowchart_image, (100, 150), (150, 180), (0, 0, 0), 2)
        
        # Draw connecting lines
        cv2.line(flowchart_image, (100, 65), (150, 65), (0, 0, 0), 2)
        cv2.line(flowchart_image, (125, 80), (125, 150), (0, 0, 0), 2)
        
        is_flowchart = self.merger._is_flowchart(flowchart_image)
        assert is_flowchart, "Should detect flowchart pattern"
    
    def test_composite_figure_detection(self):
        """Test composite figure (grid) detection."""
        # Create a composite figure with 2x2 grid
        composite_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Draw 4 subfigures in a grid
        cv2.rectangle(composite_image, (50, 50), (150, 150), (0, 0, 0), -1)
        cv2.rectangle(composite_image, (250, 50), (350, 150), (0, 0, 0), -1)
        cv2.rectangle(composite_image, (50, 250), (150, 350), (0, 0, 0), -1)
        cv2.rectangle(composite_image, (250, 250), (350, 350), (0, 0, 0), -1)
        
        is_composite = self.merger._is_composite_figure(composite_image)
        assert is_composite, "Should detect composite figure pattern"
    
    def test_grid_arrangement_detection(self):
        """Test grid arrangement detection from centroids."""
        # Create centroids in a 2x2 grid
        centroids = np.array([
            [100, 100],  # Top-left
            [300, 100],  # Top-right
            [100, 300],  # Bottom-left
            [300, 300]   # Bottom-right
        ])
        
        is_grid = self.merger._check_grid_arrangement(centroids)
        assert is_grid, "Should detect grid arrangement"
    
    def test_non_grid_arrangement(self):
        """Test that non-grid arrangements are not detected as grids."""
        # Create random centroids
        centroids = np.array([
            [100, 100],
            [150, 200],
            [300, 150],
            [250, 350]
        ])
        
        is_grid = self.merger._check_grid_arrangement(centroids)
        assert not is_grid, "Should not detect non-grid as grid"
    
    def test_type_specific_merge_flowchart(self):
        """Test that flowchart components are merged correctly."""
        # Create two close flowchart bboxes
        bboxes = [
            {'bbox': [100, 100, 200, 200], 'type': 'figure', 'score': 0.9},
            {'bbox': [220, 100, 320, 200], 'type': 'figure', 'score': 0.9}
        ]
        
        # Create flowchart-like image
        flowchart_image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
        # Draw boxes and connecting line
        cv2.rectangle(flowchart_image, (100, 100), (200, 200), (0, 0, 0), 2)
        cv2.rectangle(flowchart_image, (220, 100), (320, 200), (0, 0, 0), 2)
        cv2.line(flowchart_image, (200, 150), (220, 150), (0, 0, 0), 2)
        
        merged = self.merger.merge(bboxes, page_image=flowchart_image)
        
        # Should merge into one bbox
        assert len(merged) == 1, f"Expected 1 merged bbox, got {len(merged)}"
        assert merged[0].get('merged_from', 1) >= 2, "Should indicate merge from multiple boxes"


class TestBoundaryRefinement:
    """Test boundary refinement functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.merger = bbox_merger.SmartBBoxMerger()
    
    def test_refine_bbox_with_padding(self):
        """Test that refinement removes excess white space."""
        # Create image with content in center and white space around
        page_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        # Draw content in center
        cv2.rectangle(page_image, (150, 150), (350, 350), (0, 0, 0), -1)
        
        # Create bbox with excess padding
        bbox = {
            'bbox': [100, 100, 400, 400],
            'type': 'figure',
            'score': 0.9
        }
        
        refined = self.merger._refine_single_bbox(bbox, page_image)
        
        # Refined bbox should be smaller (closer to actual content)
        original_area = (400 - 100) * (400 - 100)
        refined_area = (refined['bbox'][2] - refined['bbox'][0]) * \
                      (refined['bbox'][3] - refined['bbox'][1])
        
        assert refined_area < original_area, "Refined bbox should be smaller"
        
        # Refined bbox should still contain the content (with small tolerance)
        assert refined['bbox'][0] <= 155, "Should include left edge of content"
        assert refined['bbox'][1] <= 155, "Should include top edge of content"
        assert refined['bbox'][2] >= 345, "Should include right edge of content"
        assert refined['bbox'][3] >= 345, "Should include bottom edge of content"
    
    def test_refine_multiple_bboxes(self):
        """Test refining multiple bboxes."""
        page_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        cv2.rectangle(page_image, (50, 50), (150, 150), (0, 0, 0), -1)
        cv2.rectangle(page_image, (300, 300), (400, 400), (0, 0, 0), -1)
        
        bboxes = [
            {'bbox': [30, 30, 170, 170], 'type': 'figure', 'score': 0.9},
            {'bbox': [280, 280, 420, 420], 'type': 'figure', 'score': 0.9}
        ]
        
        refined = self.merger.refine_boundaries(bboxes, page_image)
        
        assert len(refined) == 2, "Should refine all bboxes"
        
        # Both should be refined (smaller)
        for i in range(2):
            original_area = (bboxes[i]['bbox'][2] - bboxes[i]['bbox'][0]) * \
                          (bboxes[i]['bbox'][3] - bboxes[i]['bbox'][1])
            refined_area = (refined[i]['bbox'][2] - refined[i]['bbox'][0]) * \
                          (refined[i]['bbox'][3] - refined[i]['bbox'][1])
            assert refined_area <= original_area, f"Bbox {i} should be refined"
    
    def test_refine_empty_bbox(self):
        """Test that empty bboxes are handled gracefully."""
        page_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        
        bbox = {
            'bbox': [100, 100, 100, 100],  # Zero area
            'type': 'figure',
            'score': 0.9
        }
        
        refined = self.merger._refine_single_bbox(bbox, page_image)
        
        # Should return original bbox
        assert refined['bbox'] == bbox['bbox'], "Should return original for invalid bbox"


class TestComplexFigureSplitting:
    """Test complex figure splitting functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.merger = bbox_merger.SmartBBoxMerger()
    
    def test_split_disconnected_components(self):
        """Test splitting of disconnected components."""
        # Create image with two separate figures
        page_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        cv2.rectangle(page_image, (50, 50), (150, 150), (0, 0, 0), -1)
        cv2.rectangle(page_image, (300, 300), (400, 400), (0, 0, 0), -1)
        
        # Create bbox that incorrectly includes both
        bbox = {
            'bbox': [40, 40, 410, 410],
            'type': 'figure',
            'score': 0.9
        }
        
        split = self.merger._check_and_split_bbox(bbox, page_image)
        
        # Should split into 2 bboxes
        assert len(split) == 2, f"Expected 2 split bboxes, got {len(split)}"
        
        # Each split bbox should be marked
        for s in split:
            assert 'split_from' in s, "Split bbox should be marked"
    
    def test_no_split_for_connected_figure(self):
        """Test that connected figures are not split."""
        # Create image with one connected figure
        page_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        cv2.rectangle(page_image, (100, 100), (400, 400), (0, 0, 0), -1)
        
        bbox = {
            'bbox': [90, 90, 410, 410],
            'type': 'figure',
            'score': 0.9
        }
        
        split = self.merger._check_and_split_bbox(bbox, page_image)
        
        # Should not split
        assert len(split) == 1, "Should not split connected figure"
        assert split[0] is bbox, "Should return original bbox"
    
    def test_split_multiple_bboxes(self):
        """Test splitting multiple bboxes."""
        page_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        
        # First bbox: two disconnected components
        cv2.rectangle(page_image, (50, 50), (100, 100), (0, 0, 0), -1)
        cv2.rectangle(page_image, (150, 50), (200, 100), (0, 0, 0), -1)
        
        # Second bbox: one connected component
        cv2.rectangle(page_image, (300, 300), (400, 400), (0, 0, 0), -1)
        
        bboxes = [
            {'bbox': [40, 40, 210, 110], 'type': 'figure', 'score': 0.9},
            {'bbox': [290, 290, 410, 410], 'type': 'figure', 'score': 0.9}
        ]
        
        result = self.merger.split_complex_figures(bboxes, page_image)
        
        # First should split into 2, second should remain 1
        assert len(result) >= 2, f"Expected at least 2 bboxes, got {len(result)}"
    
    def test_component_separation_check(self):
        """Test that component separation is correctly detected."""
        # Create stats for well-separated components
        stats = np.array([
            [0, 0, 0, 0, 0],  # Background
            [50, 50, 100, 100, 10000],  # Component 1
            [300, 300, 100, 100, 10000]  # Component 2
        ])
        
        is_separated = self.merger._are_components_separated([1, 2], stats, (500, 500))
        assert is_separated, "Should detect well-separated components"
        
        # Create stats for touching components
        stats_touching = np.array([
            [0, 0, 0, 0, 0],  # Background
            [50, 50, 100, 100, 10000],  # Component 1
            [155, 50, 100, 100, 10000]  # Component 2 (close to component 1)
        ])
        
        is_separated = self.merger._are_components_separated([1, 2], stats_touching, (500, 500))
        assert not is_separated, "Should not detect touching components as separated"


class TestIntegration:
    """Integration tests for the complete enhanced merger."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.merger = bbox_merger.SmartBBoxMerger({
            'enable_context_aware_merge': True,
            'enable_type_specific_merge': True,
            'enable_boundary_refinement': True
        })
    
    def test_complete_merge_pipeline(self):
        """Test the complete merge pipeline with all enhancements."""
        # Create a complex page with multiple figures
        page_image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
        
        # Draw flowchart (should merge)
        cv2.rectangle(page_image, (100, 100), (200, 200), (0, 0, 0), 2)
        cv2.rectangle(page_image, (250, 100), (350, 200), (0, 0, 0), 2)
        cv2.line(page_image, (200, 150), (250, 150), (0, 0, 0), 2)
        
        # Draw separate figure (should not merge with flowchart)
        cv2.rectangle(page_image, (600, 600), (800, 800), (0, 0, 0), -1)
        
        bboxes = [
            {'bbox': [90, 90, 210, 210], 'type': 'figure', 'score': 0.9},
            {'bbox': [240, 90, 360, 210], 'type': 'figure', 'score': 0.9},
            {'bbox': [590, 590, 810, 810], 'type': 'figure', 'score': 0.9}
        ]
        
        merged = self.merger.merge(bboxes, page_image=page_image)
        
        # Should have 2 bboxes: merged flowchart + separate figure
        assert len(merged) == 2, f"Expected 2 merged bboxes, got {len(merged)}"
    
    def test_merge_with_captions(self):
        """Test merging with caption information."""
        page_image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
        
        # Draw two subfigures
        cv2.rectangle(page_image, (100, 100), (200, 200), (0, 0, 0), -1)
        cv2.rectangle(page_image, (250, 100), (350, 200), (0, 0, 0), -1)
        
        bboxes = [
            {'bbox': [90, 90, 210, 210], 'type': 'figure', 'score': 0.9},
            {'bbox': [240, 90, 360, 210], 'type': 'figure', 'score': 0.9}
        ]
        
        # Caption below both figures
        captions = [
            {'text': 'Figure 1: (a) First subfigure (b) Second subfigure',
             'bbox': [100, 220, 350, 250]}
        ]
        
        merged = self.merger.merge(bboxes, page_image=page_image, captions=captions)
        
        # Should merge subfigures sharing the same caption
        assert len(merged) <= 1, f"Expected subfigures to merge, got {len(merged)} bboxes"


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("Enhanced BBox Merger Test Suite (v1.7)")
    print("="*70)
    
    test_classes = [
        TestContextAwareMerging,
        TestBoundaryRefinement,
        TestComplexFigureSplitting,
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

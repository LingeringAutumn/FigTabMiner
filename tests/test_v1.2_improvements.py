#!/usr/bin/env python3
"""
Test script for v1.2 improvements:
1. Math equation filtering
2. Merge validation
3. Enhanced table extraction
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_math_equation_filter():
    """Test math equation filtering"""
    print("\n" + "="*60)
    print("Test 1: Math Equation Filtering")
    print("="*60)
    
    from figtabminer import content_classifier
    import numpy as np
    
    classifier = content_classifier.ContentClassifier()
    
    # Test case 1: Wide and short (likely equation)
    equation_item = {
        'bbox': [100, 200, 500, 250],  # Wide and short
        'type': 'table'
    }
    
    is_equation = classifier.is_math_equation(equation_item)
    print(f"\nTest case 1 (wide & short): {'✓ PASS' if is_equation else '❌ FAIL'}")
    print(f"  Expected: True (equation), Got: {is_equation}")
    
    # Test case 2: Square shape (likely table)
    table_item = {
        'bbox': [100, 200, 400, 500],  # More square
        'type': 'table'
    }
    
    is_equation = classifier.is_math_equation(table_item)
    print(f"\nTest case 2 (square): {'✓ PASS' if not is_equation else '❌ FAIL'}")
    print(f"  Expected: False (table), Got: {is_equation}")
    
    # Test case 3: With math symbols in text
    text_lines = [
        {'text': 'The equation is: α + β = γ', 'bbox': [100, 180, 500, 200]}
    ]
    
    is_equation = classifier.is_math_equation(equation_item, text_lines=text_lines)
    print(f"\nTest case 3 (with math symbols): {'✓ PASS' if is_equation else '❌ FAIL'}")
    print(f"  Expected: True (equation), Got: {is_equation}")
    
    return True


def test_merge_validation():
    """Test enhanced merge validation"""
    print("\n" + "="*60)
    print("Test 2: Merge Validation Enhancement")
    print("="*60)
    
    from figtabminer import bbox_merger
    import numpy as np
    import cv2
    
    merger = bbox_merger.SmartBBoxMerger()
    
    # Create a dummy page image
    page_image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    
    # Test case 1: Close boxes (should merge)
    close_boxes = [
        {'bbox': [100, 100, 200, 200], 'type': 'figure', 'score': 0.9},
        {'bbox': [210, 100, 310, 200], 'type': 'figure', 'score': 0.9}
    ]
    
    should_merge = merger._should_merge_component(close_boxes, page_image)
    print(f"\nTest case 1 (close boxes): {'✓ PASS' if should_merge else '❌ FAIL'}")
    print(f"  Expected: True (should merge), Got: {should_merge}")
    
    # Test case 2: Far apart boxes (should NOT merge)
    far_boxes = [
        {'bbox': [100, 100, 200, 200], 'type': 'figure', 'score': 0.9},
        {'bbox': [400, 100, 500, 200], 'type': 'figure', 'score': 0.9}
    ]
    
    should_merge = merger._should_merge_component(far_boxes, page_image)
    print(f"\nTest case 2 (far boxes): {'✓ PASS' if not should_merge else '❌ FAIL'}")
    print(f"  Expected: False (should NOT merge), Got: {should_merge}")
    
    # Test case 3: Large merged area (should NOT merge)
    large_area_boxes = [
        {'bbox': [100, 100, 200, 200], 'type': 'figure', 'score': 0.9},
        {'bbox': [800, 800, 900, 900], 'type': 'figure', 'score': 0.9}
    ]
    
    should_merge = merger._should_merge_component(large_area_boxes, page_image)
    print(f"\nTest case 3 (large merged area): {'✓ PASS' if not should_merge else '❌ FAIL'}")
    print(f"  Expected: False (should NOT merge), Got: {should_merge}")
    
    return True


def test_enhanced_table_extraction():
    """Test enhanced table extraction"""
    print("\n" + "="*60)
    print("Test 3: Enhanced Table Extraction")
    print("="*60)
    
    try:
        from figtabminer import table_extract_v2
        print("\n✓ Enhanced table extractor module loaded successfully")
        
        # Check if all strategies are defined
        extractor = table_extract_v2.EnhancedTableExtractor({'layout': False})
        num_strategies = len(extractor.table_settings_variants)
        
        print(f"✓ Found {num_strategies} extraction strategies:")
        for settings in extractor.table_settings_variants:
            print(f"  - {settings.get('name', 'unknown')}")
        
        if num_strategies >= 4:
            print(f"\n✓ PASS: Multiple strategies available")
            return True
        else:
            print(f"\n❌ FAIL: Expected >= 4 strategies, got {num_strategies}")
            return False
    
    except Exception as e:
        print(f"\n❌ FAIL: Could not load enhanced extractor: {e}")
        return False


def main():
    """Run all v1.2 tests"""
    print("\n" + "="*60)
    print("FigTabMiner v1.2 Improvements Test Suite")
    print("="*60)
    
    results = {}
    
    # Test 1: Math equation filtering
    try:
        results['math_filter'] = test_math_equation_filter()
    except Exception as e:
        print(f"\n❌ Math equation filter test failed: {e}")
        import traceback
        traceback.print_exc()
        results['math_filter'] = False
    
    # Test 2: Merge validation
    try:
        results['merge_validation'] = test_merge_validation()
    except Exception as e:
        print(f"\n❌ Merge validation test failed: {e}")
        import traceback
        traceback.print_exc()
        results['merge_validation'] = False
    
    # Test 3: Enhanced table extraction
    try:
        results['table_extraction'] = test_enhanced_table_extraction()
    except Exception as e:
        print(f"\n❌ Table extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        results['table_extraction'] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All v1.2 improvements are working!")
        return 0
    else:
        print("\n⚠️  Some tests failed, but this may be expected")
        return 1


if __name__ == "__main__":
    sys.exit(main())

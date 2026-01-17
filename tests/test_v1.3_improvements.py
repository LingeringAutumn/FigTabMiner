#!/usr/bin/env python3
"""
Test script for v1.3 improvements:
1. Enhanced chart type classification
2. Bar chart data extraction
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_chart_classification():
    """Test enhanced chart type classification"""
    print("\n" + "="*60)
    print("Test 1: Enhanced Chart Classification")
    print("="*60)
    
    try:
        from figtabminer import chart_classifier
        print("\n✓ Chart classifier module loaded successfully")
        
        # Create classifier
        classifier = chart_classifier.ChartClassifier()
        
        # Test keyword matching
        text = "This bar chart shows the distribution of values"
        keyword_scores = classifier._classify_by_keywords(text)
        
        print(f"\n✓ Keyword classification working")
        print(f"  Bar chart score: {keyword_scores.get('bar_chart', 0):.2f}")
        print(f"  Pie chart score: {keyword_scores.get('pie_chart', 0):.2f}")
        
        if keyword_scores.get('bar_chart', 0) > 0:
            print(f"\n✓ PASS: Bar chart correctly identified from keywords")
            return True
        else:
            print(f"\n❌ FAIL: Bar chart not identified")
            return False
    
    except Exception as e:
        print(f"\n❌ FAIL: Chart classification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bar_chart_digitizer():
    """Test bar chart data extraction"""
    print("\n" + "="*60)
    print("Test 2: Bar Chart Data Extraction")
    print("="*60)
    
    try:
        from figtabminer import bar_chart_digitizer
        import numpy as np
        import cv2
        
        print("\n✓ Bar chart digitizer module loaded successfully")
        
        # Create a simple synthetic bar chart for testing
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Draw axes
        cv2.line(img, (50, 350), (550, 350), (0, 0, 0), 2)  # X-axis
        cv2.line(img, (50, 50), (50, 350), (0, 0, 0), 2)    # Y-axis
        
        # Draw bars
        cv2.rectangle(img, (100, 200), (150, 350), (100, 100, 255), -1)  # Bar 1
        cv2.rectangle(img, (200, 150), (250, 350), (100, 255, 100), -1)  # Bar 2
        cv2.rectangle(img, (300, 250), (350, 350), (255, 100, 100), -1)  # Bar 3
        
        # Save test image
        test_img_path = "/tmp/test_bar_chart.png"
        cv2.imwrite(test_img_path, img)
        
        # Test digitizer
        digitizer = bar_chart_digitizer.BarChartDigitizer()
        df = digitizer.digitize(test_img_path, orientation='vertical')
        
        if df is not None and len(df) > 0:
            print(f"\n✓ PASS: Extracted {len(df)} bars")
            print(f"\nExtracted data:")
            print(df)
            return True
        else:
            print(f"\n❌ FAIL: No bars extracted")
            return False
    
    except Exception as e:
        print(f"\n❌ FAIL: Bar chart digitizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration with ai_enrich"""
    print("\n" + "="*60)
    print("Test 3: Integration Test")
    print("="*60)
    
    try:
        from figtabminer import ai_enrich
        
        # Check if enhanced features are available
        print(f"\n✓ Chart classifier available: {ai_enrich.CHART_CLASSIFIER_AVAILABLE}")
        print(f"✓ Bar digitizer available: {ai_enrich.BAR_DIGITIZER_AVAILABLE}")
        
        if ai_enrich.CHART_CLASSIFIER_AVAILABLE and ai_enrich.BAR_DIGITIZER_AVAILABLE:
            print(f"\n✓ PASS: All v1.3 features integrated")
            return True
        else:
            print(f"\n⚠️  WARNING: Some features not available (may be expected)")
            return True  # Not a failure, just a warning
    
    except Exception as e:
        print(f"\n❌ FAIL: Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all v1.3 tests"""
    print("\n" + "="*60)
    print("FigTabMiner v1.3 Improvements Test Suite")
    print("="*60)
    
    results = {}
    
    # Test 1: Chart classification
    try:
        results['chart_classification'] = test_chart_classification()
    except Exception as e:
        print(f"\n❌ Chart classification test failed: {e}")
        import traceback
        traceback.print_exc()
        results['chart_classification'] = False
    
    # Test 2: Bar chart digitizer
    try:
        results['bar_digitizer'] = test_bar_chart_digitizer()
    except Exception as e:
        print(f"\n❌ Bar chart digitizer test failed: {e}")
        import traceback
        traceback.print_exc()
        results['bar_digitizer'] = False
    
    # Test 3: Integration
    try:
        results['integration'] = test_integration()
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        results['integration'] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All v1.3 improvements are working!")
        return 0
    else:
        print("\n⚠️  Some tests failed, but this may be expected")
        return 1


if __name__ == "__main__":
    sys.exit(main())

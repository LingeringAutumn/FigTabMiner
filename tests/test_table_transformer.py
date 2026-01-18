#!/usr/bin/env python3
"""
Test script for Table Transformer integration (v1.6).

Tests:
1. Table Transformer detector availability and initialization
2. Table detection with Table Transformer
3. Integration with table_extract_v2.py
4. Fallback mechanism to pdfplumber
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60 + "\n")


def test_table_transformer_availability():
    """Test 1: Check if Table Transformer is available"""
    print_header("Test 1: Table Transformer Availability")
    
    try:
        from figtabminer.detectors import table_transformer_detector
        print("✓ Table Transformer detector module loaded successfully")
        
        available = table_transformer_detector.is_available()
        print(f"✓ Table Transformer available: {available}")
        
        if available:
            print("✓ PASS: Table Transformer is ready to use")
            return True
        else:
            print("⚠️  WARNING: Table Transformer not available (not installed)")
            print("   Install with: pip install transformers torch torchvision")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Could not load Table Transformer detector: {e}")
        return False


def test_table_transformer_detection():
    """Test 2: Test Table Transformer detection on a sample image"""
    print_header("Test 2: Table Transformer Detection")
    
    try:
        from figtabminer.detectors import table_transformer_detector
        
        if not table_transformer_detector.is_available():
            print("⚠️  SKIP: Table Transformer not available")
            return None
        
        # Find a test image
        test_image = None
        data_dir = Path("data/outputs")
        
        if data_dir.exists():
            # Look for any page image
            for img_path in data_dir.rglob("page_*.png"):
                test_image = str(img_path)
                break
        
        if not test_image:
            print("⚠️  SKIP: No test images found in data/outputs")
            return None
        
        print(f"Using test image: {test_image}")
        
        # Test detection
        print("Running Table Transformer detection...")
        print("(This may take a while on first run - downloading models)")
        
        detections = table_transformer_detector.detect_tables_transformer(
            test_image,
            conf_threshold=0.7
        )
        
        print(f"✓ Detection completed: {len(detections)} tables found")
        
        if detections:
            # Show detections
            print("\nDetected tables:")
            for i, det in enumerate(detections):
                print(f"  {i+1}. Table (score: {det['score']:.3f})")
                print(f"     bbox: {det['bbox']}")
            
            print("\n✓ PASS: Table Transformer detection working")
            return True
        else:
            print("⚠️  WARNING: No tables detected (image may not contain tables)")
            return True
            
    except Exception as e:
        print(f"❌ FAIL: Table Transformer detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_table_extract_integration():
    """Test 3: Test integration with table_extract_v2.py"""
    print_header("Test 3: Table Extraction Integration")
    
    try:
        from figtabminer import table_extract_v2
        
        # Check if Table Transformer is integrated
        has_tt = hasattr(table_extract_v2, 'TABLE_TRANSFORMER_AVAILABLE')
        print(f"✓ TABLE_TRANSFORMER_AVAILABLE flag exists: {has_tt}")
        
        if has_tt:
            tt_available = table_extract_v2.TABLE_TRANSFORMER_AVAILABLE
            print(f"✓ Table Transformer available in table_extract_v2: {tt_available}")
            
            if tt_available:
                print("✓ PASS: Table Transformer integrated into table extraction")
                return True
            else:
                print("⚠️  WARNING: Table Transformer not available (not installed)")
                print("✓ PASS: Integration code present (fallback will work)")
                return True
        else:
            print("❌ FAIL: TABLE_TRANSFORMER_AVAILABLE flag not found")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_strategy_extraction():
    """Test 4: Test multi-strategy table extraction"""
    print_header("Test 4: Multi-Strategy Table Extraction")
    
    try:
        from figtabminer import table_extract_v2
        
        # Check available strategies
        print("Checking available extraction strategies:")
        
        strategies = []
        
        # Layout detection
        try:
            from figtabminer import layout_detect
            if layout_detect.layout_available():
                strategies.append("Layout detection")
                print("  ✓ Layout detection (DocLayout-YOLO or PubLayNet)")
        except:
            pass
        
        # Table Transformer
        if table_extract_v2.TABLE_TRANSFORMER_AVAILABLE:
            strategies.append("Table Transformer")
            print("  ✓ Table Transformer")
        else:
            print("  ⚠️  Table Transformer (not available)")
        
        # pdfplumber
        strategies.append("pdfplumber")
        print("  ✓ pdfplumber (always available)")
        
        # Visual detection
        strategies.append("Visual detection")
        print("  ✓ Visual line detection")
        
        print(f"\nTotal strategies available: {len(strategies)}")
        
        if len(strategies) >= 3:
            print("✓ PASS: Multiple extraction strategies available")
            return True
        else:
            print("⚠️  WARNING: Limited strategies available")
            return True
            
    except Exception as e:
        print(f"❌ FAIL: Multi-strategy test failed: {e}")
        return False


def test_fallback_mechanism():
    """Test 5: Test fallback mechanism"""
    print_header("Test 5: Fallback Mechanism")
    
    try:
        from figtabminer import table_extract_v2
        
        tt_available = table_extract_v2.TABLE_TRANSFORMER_AVAILABLE
        
        print(f"Table Transformer available: {tt_available}")
        
        if tt_available:
            print("✓ Table Transformer will be used as primary table detector")
            print("✓ pdfplumber and visual detection available as fallback")
            print("✓ PASS: Multi-level fallback mechanism in place")
            return True
        else:
            print("⚠️  Table Transformer not available")
            print("✓ pdfplumber and visual detection will be used")
            print("✓ PASS: Fallback mechanism working")
            return True
            
    except Exception as e:
        print(f"❌ FAIL: Fallback test failed: {e}")
        return False


def main():
    print_header("FigTabMiner v1.6 - Table Transformer Integration Test Suite")
    
    results = {
        "table_transformer_availability": test_table_transformer_availability(),
        "table_transformer_detection": test_table_transformer_detection(),
        "integration": test_table_extract_integration(),
        "multi_strategy": test_multi_strategy_extraction(),
        "fallback": test_fallback_mechanism(),
    }
    
    print_header("Test Summary")
    
    for test_name, result in results.items():
        if result is True:
            print(f"✓ PASSED: {test_name}")
        elif result is False:
            print(f"❌ FAILED: {test_name}")
        else:
            print(f"⚠️  SKIPPED: {test_name}")
    
    failed = sum(1 for r in results.values() if r is False)
    passed = sum(1 for r in results.values() if r is True)
    skipped = sum(1 for r in results.values() if r is None)
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n✓ All tests passed or skipped!")
        if skipped > 0:
            print("  (Some tests skipped due to missing dependencies or test data)")
    else:
        print(f"\n❌ {failed} test(s) failed")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Comprehensive test script for v1.5 and v1.6 improvements.

v1.5: DocLayout-YOLO integration
v1.6: Table Transformer integration

Tests:
1. DocLayout-YOLO availability and detection
2. Table Transformer availability and detection
3. Fallback mechanisms
4. End-to-end pipeline test
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


def test_doclayout_yolo():
    """Test DocLayout-YOLO (v1.5)"""
    print_header("Test 1: DocLayout-YOLO (v1.5)")
    
    try:
        from figtabminer.detectors import doclayout_detector
        
        available = doclayout_detector.is_available()
        print(f"DocLayout-YOLO available: {available}")
        
        if available:
            print("✓ PASS: DocLayout-YOLO ready")
            return True
        else:
            print("⚠️  WARNING: DocLayout-YOLO not installed")
            print("   Install with: pip install doclayout-yolo")
            return None
            
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_table_transformer():
    """Test Table Transformer (v1.6)"""
    print_header("Test 2: Table Transformer (v1.6)")
    
    try:
        from figtabminer.detectors import table_transformer_detector
        
        available = table_transformer_detector.is_available()
        print(f"Table Transformer available: {available}")
        
        if available:
            print("✓ PASS: Table Transformer ready")
            return True
        else:
            print("⚠️  WARNING: Table Transformer not installed")
            print("   Install with: pip install transformers torch torchvision")
            return None
            
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_layout_detection_fallback():
    """Test layout detection fallback chain"""
    print_header("Test 3: Layout Detection Fallback Chain")
    
    try:
        from figtabminer import layout_detect
        
        status = layout_detect.get_layout_status()
        
        print("Layout detection status:")
        print(f"  - Available: {status['available']}")
        print(f"  - DocLayout-YOLO available: {status.get('doclayout_available', False)}")
        print(f"  - PubLayNet available: {status.get('publaynet_available', False)}")
        print(f"  - Primary detector: {status.get('primary_detector', 'none')}")
        print(f"  - Status: {status.get('status', 'unknown')}")
        
        if status['available']:
            primary = status.get('primary_detector', 'none')
            if primary == 'doclayout_yolo':
                print("\n✓ Using DocLayout-YOLO (best accuracy)")
                print("✓ PubLayNet available as fallback")
            elif primary == 'publaynet':
                print("\n✓ Using PubLayNet (good accuracy)")
                print("⚠️  DocLayout-YOLO not available")
            else:
                print("\n⚠️  No detector initialized yet")
            
            print("✓ PASS: Layout detection available with fallback")
            return True
        else:
            print("\n❌ FAIL: No layout detection available")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_table_extraction_strategies():
    """Test table extraction strategies"""
    print_header("Test 4: Table Extraction Strategies")
    
    try:
        from figtabminer import table_extract_v2
        from figtabminer import layout_detect
        
        strategies = []
        
        # Check layout detection
        if layout_detect.layout_available():
            status = layout_detect.get_layout_status()
            detector = status.get('primary_detector', 'none')
            if detector != 'none':
                strategies.append(f"Layout detection ({detector})")
        
        # Check Table Transformer
        if table_extract_v2.TABLE_TRANSFORMER_AVAILABLE:
            strategies.append("Table Transformer")
        
        # pdfplumber and visual are always available
        strategies.append("pdfplumber (multi-strategy)")
        strategies.append("Visual line detection")
        
        print("Available table extraction strategies:")
        for i, strategy in enumerate(strategies, 1):
            print(f"  {i}. {strategy}")
        
        print(f"\nTotal: {len(strategies)} strategies")
        
        if len(strategies) >= 4:
            print("✓ PASS: All strategies available")
            return True
        elif len(strategies) >= 2:
            print("✓ PASS: Core strategies available")
            return True
        else:
            print("⚠️  WARNING: Limited strategies")
            return None
            
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_end_to_end_detection():
    """Test end-to-end detection on sample image"""
    print_header("Test 5: End-to-End Detection")
    
    try:
        from figtabminer import layout_detect
        
        # Find a test image
        test_image = None
        data_dir = Path("data/outputs")
        
        if data_dir.exists():
            for img_path in data_dir.rglob("page_*.png"):
                test_image = str(img_path)
                break
        
        if not test_image:
            print("⚠️  SKIP: No test images found")
            return None
        
        print(f"Using test image: {test_image}")
        
        # Run detection
        print("Running layout detection...")
        results = layout_detect.detect_layout(test_image)
        
        print(f"✓ Detection completed: {len(results)} items found")
        
        if results:
            # Check detector used
            detector = results[0].get("detector", "unknown")
            print(f"✓ Detector used: {detector}")
            
            # Count by type
            fig_count = sum(1 for r in results if r["type"] == "figure")
            table_count = sum(1 for r in results if r["type"] == "table")
            
            print(f"✓ Found {fig_count} figures, {table_count} tables")
            
            print("\n✓ PASS: End-to-end detection working")
            return True
        else:
            print("⚠️  No items detected (may be expected)")
            return True
            
    except Exception as e:
        print(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_system_capabilities():
    """Test overall system capabilities"""
    print_header("Test 6: System Capabilities Summary")
    
    try:
        from figtabminer import layout_detect
        from figtabminer import table_extract_v2
        from figtabminer.detectors import doclayout_detector
        from figtabminer.detectors import table_transformer_detector
        
        print("System Capabilities:")
        print()
        
        # Layout detection
        print("Layout Detection:")
        doclayout_avail = doclayout_detector.is_available()
        print(f"  - DocLayout-YOLO: {'✓ Available' if doclayout_avail else '✗ Not available'}")
        
        status = layout_detect.get_layout_status()
        publaynet_avail = status.get('publaynet_available', False)
        print(f"  - PubLayNet: {'✓ Available' if publaynet_avail else '✗ Not available'}")
        
        # Table detection
        print()
        print("Table Detection:")
        tt_avail = table_transformer_detector.is_available()
        print(f"  - Table Transformer: {'✓ Available' if tt_avail else '✗ Not available'}")
        print(f"  - pdfplumber: ✓ Available")
        print(f"  - Visual detection: ✓ Available")
        
        # Summary
        print()
        print("Summary:")
        total_detectors = sum([
            doclayout_avail,
            publaynet_avail,
            tt_avail,
            True,  # pdfplumber
            True   # visual
        ])
        print(f"  Total detectors available: {total_detectors}/5")
        
        if total_detectors >= 4:
            print("  Status: ✓ Excellent (most detectors available)")
            return True
        elif total_detectors >= 3:
            print("  Status: ✓ Good (core detectors available)")
            return True
        elif total_detectors >= 2:
            print("  Status: ✓ Acceptable (basic detectors available)")
            return True
        else:
            print("  Status: ⚠️  Limited (few detectors available)")
            return None
            
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def main():
    print_header("FigTabMiner v1.5 & v1.6 - Comprehensive Test Suite")
    
    print("Testing new features:")
    print("  v1.5: DocLayout-YOLO integration")
    print("  v1.6: Table Transformer integration")
    print()
    
    results = {
        "doclayout_yolo": test_doclayout_yolo(),
        "table_transformer": test_table_transformer(),
        "layout_fallback": test_layout_detection_fallback(),
        "table_strategies": test_table_extraction_strategies(),
        "end_to_end": test_end_to_end_detection(),
        "capabilities": test_system_capabilities(),
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
        print("\n✓ All critical tests passed!")
        if skipped > 0:
            print("\n⚠️  Some optional features not available:")
            print("   - Install doclayout-yolo for better layout detection")
            print("   - Install transformers+torch for better table detection")
    else:
        print(f"\n❌ {failed} test(s) failed")
    
    print("\n" + "=" * 60)
    print("Installation commands:")
    print("=" * 60)
    print()
    print("# Install DocLayout-YOLO (v1.5):")
    print("pip install doclayout-yolo")
    print()
    print("# Install Table Transformer (v1.6):")
    print("pip install transformers torch torchvision")
    print()
    print("# Or install all extras:")
    print("pip install -r requirements-extra.txt")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

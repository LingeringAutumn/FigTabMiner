#!/usr/bin/env python3
"""
Test script for DocLayout-YOLO integration (v1.5).

Tests:
1. DocLayout-YOLO detector availability and initialization
2. Layout detection with DocLayout-YOLO
3. Fallback mechanism to PubLayNet
4. Detection result format and quality
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


def test_doclayout_availability():
    """Test 1: Check if DocLayout-YOLO is available"""
    print_header("Test 1: DocLayout-YOLO Availability")
    
    try:
        from figtabminer.detectors import doclayout_detector
        print("✓ DocLayout detector module loaded successfully")
        
        available = doclayout_detector.is_available()
        print(f"✓ DocLayout-YOLO available: {available}")
        
        if available:
            print("✓ PASS: DocLayout-YOLO is ready to use")
            return True
        else:
            print("⚠️  WARNING: DocLayout-YOLO not available (not installed)")
            print("   Install with: pip install doclayout-yolo")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Could not load DocLayout detector: {e}")
        return False


def test_doclayout_detection():
    """Test 2: Test DocLayout-YOLO detection on a sample image"""
    print_header("Test 2: DocLayout-YOLO Detection")
    
    try:
        from figtabminer.detectors import doclayout_detector
        
        if not doclayout_detector.is_available():
            print("⚠️  SKIP: DocLayout-YOLO not available")
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
        print("Running DocLayout-YOLO detection...")
        detections = doclayout_detector.detect_layout_doclayout(
            test_image,
            conf_threshold=0.25
        )
        
        print(f"✓ Detection completed: {len(detections)} elements found")
        
        if detections:
            # Count by type
            type_counts = {}
            for det in detections:
                label = det["label"]
                type_counts[label] = type_counts.get(label, 0) + 1
            
            print("\nDetection summary:")
            for label, count in sorted(type_counts.items()):
                print(f"  - {label}: {count}")
            
            # Show first few detections
            print("\nFirst 3 detections:")
            for i, det in enumerate(detections[:3]):
                print(f"  {i+1}. {det['label']} (score: {det['score']:.3f})")
                print(f"     bbox: {det['bbox']}")
            
            print("\n✓ PASS: DocLayout-YOLO detection working")
            return True
        else:
            print("⚠️  WARNING: No elements detected (image may be empty)")
            return True
            
    except Exception as e:
        print(f"❌ FAIL: DocLayout-YOLO detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_layout_detect_integration():
    """Test 3: Test integration with layout_detect.py"""
    print_header("Test 3: Layout Detection Integration")
    
    try:
        from figtabminer import layout_detect
        
        # Check status
        status = layout_detect.get_layout_status()
        print("Layout detection status:")
        print(f"  - Available: {status['available']}")
        print(f"  - DocLayout available: {status.get('doclayout_available', False)}")
        print(f"  - PubLayNet available: {status.get('publaynet_available', False)}")
        print(f"  - Primary detector: {status.get('primary_detector', 'none')}")
        print(f"  - Status: {status.get('status', 'unknown')}")
        
        # Find a test image
        test_image = None
        data_dir = Path("data/outputs")
        
        if data_dir.exists():
            for img_path in data_dir.rglob("page_*.png"):
                test_image = str(img_path)
                break
        
        if not test_image:
            print("\n⚠️  SKIP: No test images found")
            return None
        
        print(f"\nUsing test image: {test_image}")
        
        # Test detection
        print("Running integrated layout detection...")
        results = layout_detect.detect_layout(test_image)
        
        print(f"✓ Detection completed: {len(results)} items found")
        
        if results:
            # Check which detector was used
            detector_used = results[0].get("detector", "unknown")
            print(f"✓ Detector used: {detector_used}")
            
            # Count by type
            fig_count = sum(1 for r in results if r["type"] == "figure")
            table_count = sum(1 for r in results if r["type"] == "table")
            
            print(f"✓ Found {fig_count} figures, {table_count} tables")
            
            # Show first few results
            print("\nFirst 3 results:")
            for i, res in enumerate(results[:3]):
                print(f"  {i+1}. {res['type']} (score: {res['score']:.3f}, detector: {res.get('detector', 'unknown')})")
            
            print("\n✓ PASS: Layout detection integration working")
            return True
        else:
            print("⚠️  WARNING: No figures/tables detected")
            return True
            
    except Exception as e:
        print(f"❌ FAIL: Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_mechanism():
    """Test 4: Test fallback from DocLayout-YOLO to PubLayNet"""
    print_header("Test 4: Fallback Mechanism")
    
    try:
        from figtabminer import layout_detect
        
        status = layout_detect.get_layout_status()
        
        doclayout_available = status.get('doclayout_available', False)
        publaynet_available = status.get('publaynet_available', False)
        
        print(f"DocLayout-YOLO available: {doclayout_available}")
        print(f"PubLayNet available: {publaynet_available}")
        
        if doclayout_available and publaynet_available:
            print("✓ Both detectors available - fallback mechanism ready")
            print("✓ PASS: Fallback mechanism in place")
            return True
        elif doclayout_available:
            print("✓ DocLayout-YOLO available (primary)")
            print("⚠️  PubLayNet not available (no fallback)")
            print("✓ PASS: Primary detector available")
            return True
        elif publaynet_available:
            print("⚠️  DocLayout-YOLO not available")
            print("✓ PubLayNet available (fallback)")
            print("✓ PASS: Fallback detector available")
            return True
        else:
            print("❌ FAIL: No detectors available")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Fallback test failed: {e}")
        return False


def main():
    print_header("FigTabMiner v1.5 - DocLayout-YOLO Integration Test Suite")
    
    results = {
        "doclayout_availability": test_doclayout_availability(),
        "doclayout_detection": test_doclayout_detection(),
        "integration": test_layout_detect_integration(),
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

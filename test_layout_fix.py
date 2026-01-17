#!/usr/bin/env python3
"""
Test script to verify layout detection model loading fix.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from figtabminer import layout_detect

def test_layout_status():
    """Test layout detection status"""
    print("\n" + "="*60)
    print("Testing Layout Detection Status")
    print("="*60)
    
    status = layout_detect.get_layout_status()
    print(f"\nLayout Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    return status

def test_model_loading():
    """Test model loading"""
    print("\n" + "="*60)
    print("Testing Model Loading")
    print("="*60)
    
    if not layout_detect.layout_available():
        print("\n⚠️  Layout detection dependencies not available")
        print("This is expected if layoutparser/detectron2 are not installed")
        return False
    
    print("\n✓ Layout detection dependencies available")
    
    # Try to get the model
    model = layout_detect._get_model()
    
    if model is None:
        print("\n❌ Model loading failed")
        return False
    else:
        print("\n✓ Model loaded successfully")
        return True

def test_weights_normalization():
    """Test weights file normalization"""
    print("\n" + "="*60)
    print("Testing Weights File Normalization")
    print("="*60)
    
    weights_path = layout_detect._normalize_cached_weights()
    
    if weights_path:
        print(f"\n✓ Found weights at: {weights_path}")
        
        # Check if file exists
        from pathlib import Path
        if Path(weights_path).exists():
            print(f"✓ Weights file exists and is accessible")
            file_size = Path(weights_path).stat().st_size / (1024 * 1024)
            print(f"  File size: {file_size:.1f} MB")
        else:
            print(f"❌ Weights file path returned but file doesn't exist!")
            return False
    else:
        print("\n⚠️  No cached weights found")
        print("Model will download weights on first use")
    
    return True

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("FigTabMiner Layout Detection Fix Verification")
    print("="*60)
    
    results = {}
    
    # Test 1: Check status
    try:
        status = test_layout_status()
        results['status_check'] = True
    except Exception as e:
        print(f"\n❌ Status check failed: {e}")
        results['status_check'] = False
    
    # Test 2: Check weights normalization
    try:
        results['weights_norm'] = test_weights_normalization()
    except Exception as e:
        print(f"\n❌ Weights normalization failed: {e}")
        results['weights_norm'] = False
    
    # Test 3: Try to load model
    try:
        results['model_loading'] = test_model_loading()
    except Exception as e:
        print(f"\n❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        results['model_loading'] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status_icon = "✓" if passed else "❌"
        print(f"{status_icon} {test_name}: {'PASSED' if passed else 'FAILED'}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed, but this may be expected if dependencies are not installed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

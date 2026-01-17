#!/bin/bash
# Quick test runner for FigTabMiner improvements

echo "======================================================================"
echo "FigTabMiner Improvement Tests"
echo "======================================================================"
echo ""

# Test 1: Model loading fix
echo "Test 1: Model Loading Fix"
echo "----------------------------------------------------------------------"
python tests/test_layout_fix.py
if [ $? -eq 0 ]; then
    echo "✓ Model loading test PASSED"
else
    echo "❌ Model loading test FAILED"
    exit 1
fi
echo ""

# Test 2: Improved extraction
echo "Test 2: Improved Extraction"
echo "----------------------------------------------------------------------"
python tests/test_improved_extraction.py
if [ $? -eq 0 ]; then
    echo "✓ Improved extraction test PASSED"
else
    echo "❌ Improved extraction test FAILED"
    exit 1
fi
echo ""

# Test 3: Comparison test (optional, can be slow)
read -p "Run comparison test on all samples? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Test 3: Comparison Test"
    echo "----------------------------------------------------------------------"
    python tests/test_comparison.py
    if [ $? -eq 0 ]; then
        echo "✓ Comparison test PASSED"
    else
        echo "❌ Comparison test FAILED"
        exit 1
    fi
    echo ""
fi

echo "======================================================================"
echo "All tests completed!"
echo "======================================================================"

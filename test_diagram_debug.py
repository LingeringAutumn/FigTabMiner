#!/usr/bin/env python3
"""Debug script for diagram classification test."""

import sys
import os
import numpy as np
import cv2
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from figtabminer.enhanced_chart_classifier import EnhancedChartClassifier

# Create classifier
classifier = EnhancedChartClassifier({
    'enable_hierarchical': True,
    'enable_visual_analysis': False  # Focus on keyword matching
})

# Create temp directory and dummy image
temp_dir = tempfile.mkdtemp()
img = np.ones((300, 300, 3), dtype=np.uint8) * 255
img_path = os.path.join(temp_dir, 'flow.png')
cv2.imwrite(img_path, img)

# Test with flowchart caption
caption = "Flowchart of the process"

print(f"Testing with caption: '{caption}'")
print(f"Image path: {img_path}")

result = classifier.classify_hierarchical(img_path, caption=caption)

print(f"\nResult:")
print(f"  Main category: {result.main_category}")
print(f"  Sub category: {result.sub_category}")
print(f"  Main confidence: {result.confidence_by_level.get('main', 0.0):.3f}")
print(f"  Sub confidence: {result.confidence_by_level.get('sub', 0.0):.3f}")
print(f"\nMain scores: {result.debug_info.get('main_scores', {})}")
print(f"Sub scores: {result.debug_info.get('sub_scores', {})}")

# Check assertion
print(f"\nAssertion check:")
print(f"  result.main_category == 'diagram': {result.main_category == 'diagram'}")
print(f"  'diagram' in result.sub_category: {'diagram' in result.sub_category}")

if result.main_category != 'diagram':
    print(f"\n❌ FAILED: Expected main_category='diagram', got '{result.main_category}'")
else:
    print(f"\n✓ PASSED: Main category is 'diagram'")

if 'diagram' not in result.sub_category:
    print(f"❌ FAILED: Expected 'diagram' in sub_category, got '{result.sub_category}'")
else:
    print(f"✓ PASSED: Sub category contains 'diagram'")

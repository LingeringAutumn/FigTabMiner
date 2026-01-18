#!/usr/bin/env python3
"""Debug keyword matching."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from figtabminer.enhanced_chart_classifier import EnhancedChartClassifier

classifier = EnhancedChartClassifier()

# Test text
text = "flowchart of the process"

print(f"Testing text: '{text}'")
print(f"\nKeyword scores:")

scores = classifier._get_main_category_keyword_scores(text)
for category, score in scores.items():
    print(f"  {category}: {score:.3f}")

print(f"\nDiagram keywords: {classifier.CHART_KEYWORDS.get('diagram_flowchart', [])}")
print(f"\nChecking matches:")
for kw in ['flowchart', 'flow chart', 'diagram']:
    print(f"  '{kw}' in text: {kw in text}")

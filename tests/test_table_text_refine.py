#!/usr/bin/env python3
"""
Tests for table bbox text-line refinement.
"""

from src.figtabminer.table_extract_v2 import EnhancedTableExtractor


def test_refine_bbox_with_text_lines():
    extractor = EnhancedTableExtractor({"layout": False})

    bbox = [0, 0, 200, 200]
    text_lines = [
        {"bbox": [10, 10, 190, 20], "text": "Table 1: Caption"},
        {"bbox": [10, 80, 190, 90], "text": "1 2 3"},
        {"bbox": [10, 100, 190, 110], "text": "4 5 6"},
    ]

    refined = extractor._refine_bbox_with_text_lines(bbox, text_lines)
    # Caption line should be excluded; refined box should move downward.
    assert refined[1] > 40
    assert refined[3] < 140

#!/usr/bin/env python3
"""
Tests for figure text-band splitting logic.
"""

import numpy as np
import cv2

from src.figtabminer.figure_extract import _split_bbox_by_text_band


def test_split_bbox_by_text_band():
    # Create a synthetic page with two figure blocks and a text band in between.
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (20, 20), (180, 70), (0, 0, 0), -1)
    cv2.rectangle(img, (20, 130), (180, 180), (0, 0, 0), -1)

    text_lines = [
        {"bbox": [20, 90, 180, 100], "text": "Figure 1"},
        {"bbox": [20, 102, 180, 112], "text": "continued"},
    ]

    bbox = {"bbox": [0, 0, 200, 200], "type": "figure", "score": 0.9}
    split = _split_bbox_by_text_band(bbox, text_lines, img)

    assert len(split) == 2
    top, bottom = split
    assert top["bbox"][3] < bottom["bbox"][1]

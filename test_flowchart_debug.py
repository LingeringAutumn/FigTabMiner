#!/usr/bin/env python3
"""Debug flowchart detection."""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from figtabminer import bbox_merger

# Create a flowchart-like image with boxes and lines
flowchart_image = np.ones((300, 300, 3), dtype=np.uint8) * 255

# Draw some boxes (nodes)
cv2.rectangle(flowchart_image, (50, 50), (100, 80), (0, 0, 0), 2)
cv2.rectangle(flowchart_image, (150, 50), (200, 80), (0, 0, 0), 2)
cv2.rectangle(flowchart_image, (100, 150), (150, 180), (0, 0, 0), 2)

# Draw connecting lines
cv2.line(flowchart_image, (100, 65), (150, 65), (0, 0, 0), 2)
cv2.line(flowchart_image, (125, 80), (125, 150), (0, 0, 0), 2)

# Test detection
merger = bbox_merger.SmartBBoxMerger({
    'enable_context_aware_merge': True,
    'enable_type_specific_merge': True
})

# Debug the detection
gray = cv2.cvtColor(flowchart_image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
edge_ratio = np.count_nonzero(edges) / edges.size

print(f"Edge ratio: {edge_ratio}")
print(f"Edge ratio in range (0.02, 0.4)? {0.02 < edge_ratio < 0.4}")

# Detect lines
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                       minLineLength=15, maxLineGap=15)

print(f"Number of lines detected: {len(lines) if lines is not None else 0}")

# Detect rectangles
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rectangles = 0
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    if len(approx) == 4:
        rectangles += 1

print(f"Number of rectangles detected: {rectangles}")

# Test the actual method
is_flowchart = merger._is_flowchart(flowchart_image)
print(f"\nIs flowchart? {is_flowchart}")

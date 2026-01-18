#!/usr/bin/env python3
"""Debug script for split test."""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from figtabminer import bbox_merger

# Create test image
page_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
cv2.rectangle(page_image, (50, 50), (150, 150), (0, 0, 0), -1)
cv2.rectangle(page_image, (300, 300), (400, 400), (0, 0, 0), -1)

# Create bbox
bbox = {
    'bbox': [40, 40, 410, 410],
    'type': 'figure',
    'score': 0.9
}

# Test split
merger = bbox_merger.SmartBBoxMerger()

# Extract crop to debug
x0, y0, x1, y1 = 40, 40, 410, 410
crop = page_image[y0:y1, x0:x1]

# Convert to grayscale and threshold
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

print(f"Number of labels: {num_labels}")
print(f"Crop size: {crop.size}")
print(f"Min area threshold (5%): {crop.size * 0.05}")

# Filter components by size
min_area = crop.size * 0.05
large_components = []

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    print(f"Component {i}: area={area}, min_area={min_area}, passes={area >= min_area}")
    if area >= min_area:
        large_components.append(i)

print(f"\nLarge components: {large_components}")
print(f"Number of large components: {len(large_components)}")

# Check separation
if 2 <= len(large_components) <= 4:
    print("\nChecking separation...")
    is_separated = merger._are_components_separated(large_components, stats, crop.shape)
    print(f"Are components separated? {is_separated}")

# Now run the actual split
split = merger._check_and_split_bbox(bbox, page_image)

print(f"\nNumber of split bboxes: {len(split)}")
for i, s in enumerate(split):
    print(f"Bbox {i}: {s}")

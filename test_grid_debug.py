#!/usr/bin/env python3
"""Debug script for grid arrangement detection."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from figtabminer import bbox_merger

merger = bbox_merger.SmartBBoxMerger()

# Test 1: Grid arrangement (should return True)
print("Test 1: Grid arrangement (2x2)")
centroids_grid = np.array([
    [100, 100],  # Top-left
    [300, 100],  # Top-right
    [100, 300],  # Bottom-left
    [300, 300]   # Bottom-right
])

is_grid = merger._check_grid_arrangement(centroids_grid)
print(f"Centroids: {centroids_grid}")
print(f"Is grid? {is_grid}")
print(f"Expected: True\n")

# Test 2: Non-grid arrangement (should return False)
print("Test 2: Non-grid arrangement (random)")
centroids_random = np.array([
    [100, 100],
    [150, 200],
    [300, 150],
    [250, 350]
])

is_grid = merger._check_grid_arrangement(centroids_random)
print(f"Centroids: {centroids_random}")
print(f"Is grid? {is_grid}")
print(f"Expected: False\n")

# Debug the logic for random centroids
print("Debugging random centroids:")
sorted_by_y = sorted(enumerate(centroids_random), key=lambda x: x[1][1])
print(f"Sorted by y: {[(i, c) for i, c in sorted_by_y]}")

rows = []
current_row = [sorted_by_y[0]]
y_threshold = 50

for i in range(1, len(sorted_by_y)):
    y_diff = abs(sorted_by_y[i][1][1] - current_row[-1][1][1])
    print(f"  Comparing y={sorted_by_y[i][1][1]} with y={current_row[-1][1][1]}, diff={y_diff}")
    if y_diff < y_threshold:
        current_row.append(sorted_by_y[i])
    else:
        rows.append(current_row)
        current_row = [sorted_by_y[i]]

if current_row:
    rows.append(current_row)

print(f"Rows: {len(rows)}")
for i, row in enumerate(rows):
    print(f"  Row {i}: {len(row)} elements, y-coords: {[c[1][1] for _, c in row]}")

row_sizes = [len(row) for row in rows]
print(f"Row sizes: {row_sizes}")
print(f"Max - Min: {max(row_sizes) - min(row_sizes)}")
print(f"Consistent row sizes? {max(row_sizes) - min(row_sizes) <= 1}")

if len(rows) >= 2 and len(rows[0]) >= 2:
    print("\nChecking column alignment:")
    for row in rows:
        row.sort(key=lambda x: x[1][0])
    
    x_threshold = 50
    for col_idx in range(min(len(rows[0]), len(rows[1]))):
        x1 = rows[0][col_idx][1][0]
        x2 = rows[1][col_idx][1][0]
        x_diff = abs(x1 - x2)
        print(f"  Column {col_idx}: x1={x1}, x2={x2}, diff={x_diff}, aligned={x_diff <= x_threshold}")

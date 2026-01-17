"""
Bounding box utility functions for layout analysis.
Provides geometric operations and spatial relationship analysis.
"""

from typing import List, Tuple, Optional
import numpy as np


def bbox_area(bbox: List[float]) -> float:
    """Calculate area of a bounding box."""
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0) * max(0, y1 - y0)


def bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        bbox1, bbox2: [x0, y0, x1, y1] format
        
    Returns:
        IoU value between 0 and 1
    """
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])
    
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    
    if intersection == 0:
        return 0.0
    
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def bbox_overlap_ratio(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate overlap ratio (intersection / smaller box area).
    Useful for detecting containment relationships.
    """
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])
    
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    
    if intersection == 0:
        return 0.0
    
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    smaller_area = min(area1, area2)
    
    if smaller_area == 0:
        return 0.0
    
    return intersection / smaller_area


def bbox_contains(bbox1: List[float], bbox2: List[float], threshold: float = 0.9) -> bool:
    """
    Check if bbox1 contains bbox2.
    
    Args:
        bbox1: Potentially containing box
        bbox2: Potentially contained box
        threshold: Overlap ratio threshold (default 0.9)
        
    Returns:
        True if bbox1 contains bbox2
    """
    overlap = bbox_overlap_ratio(bbox2, bbox1)
    return overlap >= threshold


def bbox_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate minimum distance between two bounding boxes.
    Returns 0 if boxes overlap.
    """
    # If boxes overlap, distance is 0
    if bbox_iou(bbox1, bbox2) > 0:
        return 0.0
    
    # Calculate center points
    c1_x = (bbox1[0] + bbox1[2]) / 2
    c1_y = (bbox1[1] + bbox1[3]) / 2
    c2_x = (bbox2[0] + bbox2[2]) / 2
    c2_y = (bbox2[1] + bbox2[3]) / 2
    
    # Calculate horizontal and vertical distances
    if bbox1[2] < bbox2[0]:  # bbox1 is to the left
        dx = bbox2[0] - bbox1[2]
    elif bbox2[2] < bbox1[0]:  # bbox1 is to the right
        dx = bbox1[0] - bbox2[2]
    else:  # Horizontally overlapping
        dx = 0
    
    if bbox1[3] < bbox2[1]:  # bbox1 is above
        dy = bbox2[1] - bbox1[3]
    elif bbox2[3] < bbox1[1]:  # bbox1 is below
        dy = bbox1[1] - bbox2[3]
    else:  # Vertically overlapping
        dy = 0
    
    return np.sqrt(dx**2 + dy**2)


def bbox_horizontal_alignment(bbox1: List[float], bbox2: List[float], threshold: float = 0.5) -> bool:
    """
    Check if two boxes are horizontally aligned.
    
    Args:
        threshold: Minimum overlap ratio in vertical direction
    """
    y_overlap = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
    if y_overlap <= 0:
        return False
    
    h1 = bbox1[3] - bbox1[1]
    h2 = bbox2[3] - bbox2[1]
    min_h = min(h1, h2)
    
    if min_h == 0:
        return False
    
    return (y_overlap / min_h) >= threshold


def bbox_vertical_alignment(bbox1: List[float], bbox2: List[float], threshold: float = 0.5) -> bool:
    """
    Check if two boxes are vertically aligned.
    
    Args:
        threshold: Minimum overlap ratio in horizontal direction
    """
    x_overlap = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
    if x_overlap <= 0:
        return False
    
    w1 = bbox1[2] - bbox1[0]
    w2 = bbox2[2] - bbox2[0]
    min_w = min(w1, w2)
    
    if min_w == 0:
        return False
    
    return (x_overlap / min_w) >= threshold


def merge_bboxes(bbox1: List[float], bbox2: List[float]) -> List[float]:
    """Merge two bounding boxes into their union."""
    return [
        min(bbox1[0], bbox2[0]),
        min(bbox1[1], bbox2[1]),
        max(bbox1[2], bbox2[2]),
        max(bbox1[3], bbox2[3])
    ]


def merge_bbox_list(bboxes: List[List[float]]) -> List[float]:
    """Merge a list of bounding boxes into their union."""
    if not bboxes:
        return [0, 0, 0, 0]
    
    result = bboxes[0][:]
    for bbox in bboxes[1:]:
        result = merge_bboxes(result, bbox)
    
    return result


def bbox_aspect_ratio(bbox: List[float]) -> float:
    """Calculate aspect ratio (width / height) of a bounding box."""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    if height == 0:
        return float('inf')
    
    return width / height


def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Get center point of a bounding box."""
    return (
        (bbox[0] + bbox[2]) / 2,
        (bbox[1] + bbox[3]) / 2
    )


def bbox_relative_position(bbox1: List[float], bbox2: List[float]) -> str:
    """
    Determine relative position of bbox2 with respect to bbox1.
    
    Returns:
        One of: 'above', 'below', 'left', 'right', 'inside', 'overlap'
    """
    # Check for containment
    if bbox_contains(bbox1, bbox2, threshold=0.8):
        return 'inside'
    
    # Check for significant overlap
    if bbox_iou(bbox1, bbox2) > 0.1:
        return 'overlap'
    
    c1_x, c1_y = bbox_center(bbox1)
    c2_x, c2_y = bbox_center(bbox2)
    
    dx = c2_x - c1_x
    dy = c2_y - c1_y
    
    # Determine primary direction
    if abs(dx) > abs(dy):
        return 'right' if dx > 0 else 'left'
    else:
        return 'below' if dy > 0 else 'above'


def expand_bbox(bbox: List[float], padding: float, max_w: Optional[float] = None, 
                max_h: Optional[float] = None) -> List[float]:
    """
    Expand a bounding box by padding on all sides.
    
    Args:
        bbox: [x0, y0, x1, y1]
        padding: Pixels to add on each side
        max_w, max_h: Optional maximum bounds
    """
    x0 = max(0, bbox[0] - padding)
    y0 = max(0, bbox[1] - padding)
    x1 = bbox[2] + padding
    y1 = bbox[3] + padding
    
    if max_w is not None:
        x1 = min(max_w, x1)
    if max_h is not None:
        y1 = min(max_h, y1)
    
    return [x0, y0, x1, y1]


def clamp_bbox(bbox: List[float], max_w: float, max_h: float) -> List[float]:
    """Clamp bounding box coordinates to valid range."""
    return [
        max(0, min(bbox[0], max_w)),
        max(0, min(bbox[1], max_h)),
        max(0, min(bbox[2], max_w)),
        max(0, min(bbox[3], max_h))
    ]


def bbox_gap(bbox1: List[float], bbox2: List[float]) -> Tuple[float, float]:
    """
    Calculate horizontal and vertical gaps between two boxes.
    Returns (horizontal_gap, vertical_gap).
    Negative values indicate overlap.
    """
    # Horizontal gap
    if bbox1[2] < bbox2[0]:  # bbox1 is to the left
        h_gap = bbox2[0] - bbox1[2]
    elif bbox2[2] < bbox1[0]:  # bbox1 is to the right
        h_gap = bbox1[0] - bbox2[2]
    else:  # Horizontally overlapping
        h_gap = -(min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
    
    # Vertical gap
    if bbox1[3] < bbox2[1]:  # bbox1 is above
        v_gap = bbox2[1] - bbox1[3]
    elif bbox2[3] < bbox1[1]:  # bbox1 is below
        v_gap = bbox1[1] - bbox2[3]
    else:  # Vertically overlapping
        v_gap = -(min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
    
    return h_gap, v_gap


def are_bboxes_close(bbox1: List[float], bbox2: List[float], 
                     max_distance: float = 50) -> bool:
    """
    Check if two bounding boxes are close to each other.
    
    Args:
        max_distance: Maximum distance in pixels to be considered close
    """
    return bbox_distance(bbox1, bbox2) <= max_distance

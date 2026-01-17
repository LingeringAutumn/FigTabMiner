"""
Smart bounding box merger with multi-dimensional analysis.
Handles figure/table detection merging with semantic and visual understanding.
"""

from typing import List, Dict, Optional, Set, Tuple
import numpy as np
import cv2
from collections import defaultdict

from . import bbox_utils
from . import utils

logger = utils.setup_logging(__name__)


class SmartBBoxMerger:
    """
    Intelligent bounding box merger that considers:
    - Spatial relationships (distance, alignment, containment)
    - Semantic information (caption association, subfigure labels)
    - Visual continuity (connections, color, texture)
    - Noise filtering (arrows, small artifacts)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize merger with configuration.
        
        Args:
            config: Configuration dictionary with merge parameters
        """
        self.config = config or {}
        
        # Merge thresholds
        self.iou_threshold = self.config.get('iou_threshold', 0.3)
        self.overlap_threshold = self.config.get('overlap_threshold', 0.7)
        self.distance_threshold = self.config.get('distance_threshold', 50)
        
        # Semantic merge settings
        self.enable_semantic_merge = self.config.get('enable_semantic_merge', True)
        self.enable_visual_merge = self.config.get('enable_visual_merge', True)
        self.enable_noise_filter = self.config.get('enable_noise_filter', True)
        
        # Noise filtering thresholds
        self.arrow_aspect_ratio_min = self.config.get('arrow_aspect_ratio_min', 5.0)
        self.arrow_aspect_ratio_max = self.config.get('arrow_aspect_ratio_max', 0.2)
        self.arrow_ink_ratio_max = self.config.get('arrow_ink_ratio_max', 0.05)
        self.min_area_threshold = self.config.get('min_area_threshold', 1000)
    
    def merge(self, bboxes: List[Dict], page_image: Optional[np.ndarray] = None,
              captions: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Multi-stage merging strategy.
        
        Args:
            bboxes: List of bbox dicts with 'bbox', 'type', 'score', etc.
            page_image: Optional page image for visual analysis
            captions: Optional list of caption dicts with 'text' and 'bbox'
            
        Returns:
            Merged list of bboxes
        """
        if not bboxes:
            return []
        
        logger.debug(f"Starting merge with {len(bboxes)} boxes")
        
        # Stage 1: Forced merge (high overlap or containment)
        bboxes = self._merge_by_overlap(bboxes)
        logger.debug(f"After overlap merge: {len(bboxes)} boxes")
        
        # Stage 2: Semantic merge (caption association)
        if self.enable_semantic_merge and captions:
            bboxes = self._merge_by_caption(bboxes, captions)
            logger.debug(f"After semantic merge: {len(bboxes)} boxes")
        
        # Stage 3: Visual merge (connections, proximity)
        if self.enable_visual_merge and page_image is not None:
            bboxes = self._merge_by_visual(bboxes, page_image)
            logger.debug(f"After visual merge: {len(bboxes)} boxes")
        
        # Stage 4: Noise filtering
        if self.enable_noise_filter and page_image is not None:
            bboxes = self._filter_noise(bboxes, page_image)
            logger.debug(f"After noise filtering: {len(bboxes)} boxes")
        
        return bboxes
    
    def _merge_by_overlap(self, bboxes: List[Dict]) -> List[Dict]:
        """
        Merge boxes with high IoU or overlap ratio.
        This handles obvious cases where boxes should be merged.
        """
        if len(bboxes) <= 1:
            return bboxes
        
        merged = []
        used = set()
        
        for i, bbox1 in enumerate(bboxes):
            if i in used:
                continue
            
            # Start a merge group with this box
            group = [bbox1]
            group_indices = {i}
            
            # Find all boxes that should merge with this one
            for j, bbox2 in enumerate(bboxes):
                if j <= i or j in used:
                    continue
                
                # Check if should merge
                iou = bbox_utils.bbox_iou(bbox1['bbox'], bbox2['bbox'])
                overlap = bbox_utils.bbox_overlap_ratio(bbox1['bbox'], bbox2['bbox'])
                
                if iou >= self.iou_threshold or overlap >= self.overlap_threshold:
                    group.append(bbox2)
                    group_indices.add(j)
            
            # Merge the group
            if len(group) > 1:
                merged_bbox = self._merge_group(group)
                merged.append(merged_bbox)
                used.update(group_indices)
            else:
                merged.append(bbox1)
                used.add(i)
        
        # Iterative merging until no more merges
        changed = True
        iterations = 0
        max_iterations = 10
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            next_merged = []
            used = set()
            
            for i, bbox1 in enumerate(merged):
                if i in used:
                    continue
                
                group = [bbox1]
                group_indices = {i}
                
                for j, bbox2 in enumerate(merged):
                    if j <= i or j in used:
                        continue
                    
                    iou = bbox_utils.bbox_iou(bbox1['bbox'], bbox2['bbox'])
                    overlap = bbox_utils.bbox_overlap_ratio(bbox1['bbox'], bbox2['bbox'])
                    
                    if iou >= self.iou_threshold or overlap >= self.overlap_threshold:
                        group.append(bbox2)
                        group_indices.add(j)
                        changed = True
                
                if len(group) > 1:
                    next_merged.append(self._merge_group(group))
                    used.update(group_indices)
                else:
                    next_merged.append(bbox1)
                    used.add(i)
            
            merged = next_merged
        
        return merged
    
    def _merge_by_caption(self, bboxes: List[Dict], captions: List[Dict]) -> List[Dict]:
        """
        Merge boxes that share the same caption.
        Useful for subfigures like Figure 1(a), (b), (c).
        """
        if not captions:
            return bboxes
        
        # Map each bbox to its nearest caption
        bbox_caption_map = {}
        for i, bbox in enumerate(bboxes):
            nearest_caption = self._find_nearest_caption(bbox['bbox'], captions)
            if nearest_caption:
                caption_id = id(nearest_caption)
                if caption_id not in bbox_caption_map:
                    bbox_caption_map[caption_id] = []
                bbox_caption_map[caption_id].append((i, bbox))
        
        # Merge boxes sharing the same caption
        merged = []
        used = set()
        
        for caption_id, bbox_list in bbox_caption_map.items():
            if len(bbox_list) <= 1:
                continue
            
            # Check if this looks like a subfigure group
            indices, boxes = zip(*bbox_list)
            
            if self._is_subfigure_group(boxes):
                # Merge all boxes in this group
                merged_bbox = self._merge_group(list(boxes))
                merged.append(merged_bbox)
                used.update(indices)
                logger.debug(f"Merged {len(boxes)} subfigures sharing caption")
        
        # Add unmerged boxes
        for i, bbox in enumerate(bboxes):
            if i not in used:
                merged.append(bbox)
        
        return merged
    
    def _merge_by_visual(self, bboxes: List[Dict], page_image: np.ndarray) -> List[Dict]:
        """
        Merge boxes that are visually connected.
        Detects connecting lines, arrows, and visual continuity.
        """
        if len(bboxes) <= 1:
            return bboxes
        
        # Build connection graph
        connections = self._detect_connections(bboxes, page_image)
        
        if not connections:
            return bboxes
        
        # Find connected components
        components = self._find_connected_components(len(bboxes), connections)
        
        # Merge each component
        merged = []
        used = set()
        
        for component in components:
            if len(component) <= 1:
                continue
            
            # Get boxes in this component
            boxes = [bboxes[i] for i in component]
            
            # Verify this is a valid merge (not too spread out)
            if self._should_merge_component(boxes, page_image):
                merged_bbox = self._merge_group(boxes)
                merged.append(merged_bbox)
                used.update(component)
                logger.debug(f"Merged {len(boxes)} visually connected boxes")
        
        # Add unmerged boxes
        for i, bbox in enumerate(bboxes):
            if i not in used:
                merged.append(bbox)
        
        return merged
    
    def _filter_noise(self, bboxes: List[Dict], page_image: np.ndarray) -> List[Dict]:
        """
        Filter out noise detections like arrows, small artifacts, etc.
        """
        filtered = []
        
        for bbox in bboxes:
            # Check if it's an arrow
            if self._is_arrow(bbox, page_image):
                logger.debug(f"Filtered arrow: {bbox.get('bbox')}")
                continue
            
            # Check if it's too small
            area = bbox_utils.bbox_area(bbox['bbox'])
            if area < self.min_area_threshold:
                logger.debug(f"Filtered small box (area={area}): {bbox.get('bbox')}")
                continue
            
            # Check if it's other noise
            if self._is_noise(bbox, page_image):
                logger.debug(f"Filtered noise: {bbox.get('bbox')}")
                continue
            
            filtered.append(bbox)
        
        return filtered
    
    def _is_arrow(self, bbox: Dict, page_image: np.ndarray) -> bool:
        """
        Detect if a bbox is likely an arrow.
        Arrows typically have:
        - Extreme aspect ratio (very thin)
        - Low ink ratio (sparse content)
        - Triangle shape at one end
        """
        bbox_coords = bbox['bbox']
        aspect_ratio = bbox_utils.bbox_aspect_ratio(bbox_coords)
        
        # Check aspect ratio
        if not (aspect_ratio > self.arrow_aspect_ratio_min or 
                aspect_ratio < self.arrow_aspect_ratio_max):
            return False
        
        # Extract crop
        try:
            x0, y0, x1, y1 = [int(c) for c in bbox_coords]
            h, w = page_image.shape[:2]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            
            if x1 <= x0 or y1 <= y0:
                return False
            
            crop = page_image[y0:y1, x0:x1]
            
            if crop.size == 0:
                return False
            
            # Calculate ink ratio
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
            ink_ratio = np.count_nonzero(binary) / binary.size
            
            # Arrows have low ink ratio
            if ink_ratio < self.arrow_ink_ratio_max:
                return True
            
        except Exception as e:
            logger.debug(f"Error in arrow detection: {e}")
        
        return False
    
    def _is_noise(self, bbox: Dict, page_image: np.ndarray) -> bool:
        """
        Detect other types of noise (decorative elements, page numbers, etc.)
        """
        bbox_coords = bbox['bbox']
        area = bbox_utils.bbox_area(bbox_coords)
        
        # Very small boxes are likely noise
        if area < 500:
            return True
        
        # Check if mostly empty
        try:
            x0, y0, x1, y1 = [int(c) for c in bbox_coords]
            h, w = page_image.shape[:2]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            
            if x1 <= x0 or y1 <= y0:
                return True
            
            crop = page_image[y0:y1, x0:x1]
            
            if crop.size == 0:
                return True
            
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
            ink_ratio = np.count_nonzero(binary) / binary.size
            
            # Mostly empty
            if ink_ratio < 0.005:
                return True
            
        except Exception as e:
            logger.debug(f"Error in noise detection: {e}")
        
        return False
    
    def _find_nearest_caption(self, bbox: List[float], captions: List[Dict]) -> Optional[Dict]:
        """Find the nearest caption to a bbox."""
        if not captions:
            return None
        
        min_distance = float('inf')
        nearest = None
        
        for caption in captions:
            if 'bbox' not in caption:
                continue
            
            distance = bbox_utils.bbox_distance(bbox, caption['bbox'])
            
            # Prefer captions below the figure
            position = bbox_utils.bbox_relative_position(bbox, caption['bbox'])
            if position == 'above':
                distance += 100  # Penalty for captions above
            
            if distance < min_distance:
                min_distance = distance
                nearest = caption
        
        return nearest
    
    def _is_subfigure_group(self, boxes: List[Dict]) -> bool:
        """
        Check if boxes form a subfigure group.
        Subfigures are typically:
        - Close to each other
        - Similar in size
        - Arranged in a grid or row
        """
        if len(boxes) < 2:
            return False
        
        # Check if boxes are close
        max_distance = 100
        for i, box1 in enumerate(boxes):
            for box2 in boxes[i+1:]:
                if bbox_utils.bbox_distance(box1['bbox'], box2['bbox']) > max_distance:
                    return False
        
        # Check if similar sizes (within 2x)
        areas = [bbox_utils.bbox_area(box['bbox']) for box in boxes]
        max_area = max(areas)
        min_area = min(areas)
        
        if max_area > 4 * min_area:  # More than 4x difference
            return False
        
        return True
    
    def _detect_connections(self, bboxes: List[Dict], page_image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect visual connections between boxes.
        Returns list of (i, j) pairs indicating connected boxes.
        """
        connections = []
        
        for i, bbox1 in enumerate(bboxes):
            for j, bbox2 in enumerate(bboxes):
                if j <= i:
                    continue
                
                # Check if boxes are close enough to potentially be connected
                distance = bbox_utils.bbox_distance(bbox1['bbox'], bbox2['bbox'])
                if distance > 200:  # Too far apart
                    continue
                
                # Check for visual connection
                if self._has_visual_connection(bbox1['bbox'], bbox2['bbox'], page_image):
                    connections.append((i, j))
        
        return connections
    
    def _has_visual_connection(self, bbox1: List[float], bbox2: List[float], 
                               page_image: np.ndarray) -> bool:
        """
        Check if two boxes have a visual connection (line, arrow, etc.)
        """
        # Extract region between boxes
        try:
            # Get bounding box of the gap
            x_min = min(bbox1[0], bbox2[0])
            y_min = min(bbox1[1], bbox2[1])
            x_max = max(bbox1[2], bbox2[2])
            y_max = max(bbox1[3], bbox2[3])
            
            x0, y0, x1, y1 = [int(c) for c in [x_min, y_min, x_max, y_max]]
            h, w = page_image.shape[:2]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            
            if x1 <= x0 or y1 <= y0:
                return False
            
            region = page_image[y0:y1, x0:x1]
            
            if region.size == 0:
                return False
            
            # Convert to grayscale and detect edges
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
            edges = cv2.Canny(gray, 50, 150)
            
            # Use Hough line detection
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                                   minLineLength=20, maxLineGap=10)
            
            # If we detect lines, there might be a connection
            if lines is not None and len(lines) > 0:
                return True
            
        except Exception as e:
            logger.debug(f"Error detecting visual connection: {e}")
        
        return False
    
    def _find_connected_components(self, n: int, connections: List[Tuple[int, int]]) -> List[Set[int]]:
        """
        Find connected components in a graph.
        Uses Union-Find algorithm.
        """
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Build union-find structure
        for i, j in connections:
            union(i, j)
        
        # Group by root
        components = defaultdict(set)
        for i in range(n):
            root = find(i)
            components[root].add(i)
        
        return list(components.values())
    
    def _should_merge_component(self, boxes: List[Dict], page_image: np.ndarray) -> bool:
        """
        Verify if a connected component should actually be merged.
        Prevents merging boxes that are too spread out.
        """
        if len(boxes) <= 1:
            return False
        
        # Calculate merged bbox
        merged_bbox = bbox_utils.merge_bbox_list([box['bbox'] for box in boxes])
        merged_area = bbox_utils.bbox_area(merged_bbox)
        
        # Calculate sum of individual areas
        individual_area = sum(bbox_utils.bbox_area(box['bbox']) for box in boxes)
        
        # If merged area is much larger than sum of parts, probably shouldn't merge
        if merged_area > 3 * individual_area:
            return False
        
        return True
    
    def _merge_group(self, boxes: List[Dict]) -> Dict:
        """
        Merge a group of boxes into one.
        Combines bboxes and aggregates metadata.
        """
        if len(boxes) == 1:
            return boxes[0]
        
        # Merge bboxes
        merged_bbox = bbox_utils.merge_bbox_list([box['bbox'] for box in boxes])
        
        # Aggregate scores (use max)
        scores = [box.get('score', 0.5) for box in boxes]
        merged_score = max(scores)
        
        # Use type from highest scoring box
        best_box = max(boxes, key=lambda b: b.get('score', 0.5))
        merged_type = best_box.get('type', 'figure')
        
        return {
            'bbox': merged_bbox,
            'type': merged_type,
            'score': merged_score,
            'merged_from': len(boxes)
        }

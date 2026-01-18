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
    - Context-aware merging (title association, layout structure, visual continuity)
    - Type-specific strategies (flowcharts, composite figures, tables, single figures)
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
        
        # Enhanced v1.7 settings
        self.enable_context_aware_merge = self.config.get('enable_context_aware_merge', True)
        self.enable_type_specific_merge = self.config.get('enable_type_specific_merge', True)
        
        # Noise filtering thresholds
        self.arrow_aspect_ratio_min = self.config.get('arrow_aspect_ratio_min', 5.0)
        self.arrow_aspect_ratio_max = self.config.get('arrow_aspect_ratio_max', 0.2)
        self.arrow_ink_ratio_max = self.config.get('arrow_ink_ratio_max', 0.05)
        self.min_area_threshold = self.config.get('min_area_threshold', 1000)
        
        # Type-specific merge parameters
        self.flowchart_connection_threshold = self.config.get('flowchart_connection_threshold', 200)
        self.composite_grid_threshold = self.config.get('composite_grid_threshold', 0.8)
        self.table_text_overlap_threshold = self.config.get('table_text_overlap_threshold', 0.3)
    
    def merge(self, bboxes: List[Dict], page_image: Optional[np.ndarray] = None,
              captions: Optional[List[Dict]] = None, page_layout: Optional[Dict] = None) -> List[Dict]:
        """
        Multi-stage merging strategy.
        
        Args:
            bboxes: List of bbox dicts with 'bbox', 'type', 'score', etc.
            page_image: Optional page image for visual analysis
            captions: Optional list of caption dicts with 'text' and 'bbox'
            page_layout: Optional page layout information (columns, text regions, etc.)
            
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
        
        # Stage 3: Context-aware merge (v1.7 enhancement)
        if self.enable_context_aware_merge and page_image is not None:
            bboxes = self._merge_with_context(bboxes, page_image, captions, page_layout)
            logger.debug(f"After context-aware merge: {len(bboxes)} boxes")
        
        # Stage 4: Visual merge (connections, proximity)
        if self.enable_visual_merge and page_image is not None:
            bboxes = self._merge_by_visual(bboxes, page_image)
            logger.debug(f"After visual merge: {len(bboxes)} boxes")
        
        # Stage 5: Noise filtering
        if self.enable_noise_filter and page_image is not None:
            bboxes = self._filter_noise(bboxes, page_image)
            logger.debug(f"After noise filtering: {len(bboxes)} boxes")
        
        return bboxes
    
    def refine_boundaries(self, bboxes: List[Dict], page_image: np.ndarray) -> List[Dict]:
        """
        Refine bounding box boundaries using image processing techniques.
        
        This method:
        - Removes excess white space around figures
        - Ensures all relevant content is included
        - Tightens boundaries to actual content
        
        Args:
            bboxes: List of bbox dicts
            page_image: Page image
            
        Returns:
            List of bboxes with refined boundaries
        """
        refined = []
        
        for bbox in bboxes:
            try:
                refined_bbox = self._refine_single_bbox(bbox, page_image)
                refined.append(refined_bbox)
            except Exception as e:
                logger.warning(f"Failed to refine bbox: {e}")
                refined.append(bbox)  # Keep original if refinement fails
        
        return refined
    
    def _refine_single_bbox(self, bbox: Dict, page_image: np.ndarray) -> Dict:
        """
        Refine a single bounding box.
        
        Uses image processing to:
        1. Remove white space padding
        2. Ensure all content is included
        3. Align to content edges
        
        Args:
            bbox: Bbox dict
            page_image: Page image
            
        Returns:
            Refined bbox dict
        """
        bbox_coords = bbox['bbox']
        x0, y0, x1, y1 = [int(c) for c in bbox_coords]
        h, w = page_image.shape[:2]
        
        # Clamp to image bounds
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        
        if x1 <= x0 or y1 <= y0:
            return bbox
        
        # Extract crop
        crop = page_image[y0:y1, x0:x1]
        
        if crop.size == 0:
            return bbox
        
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        
        # Threshold to binary (invert so content is white)
        _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
        
        # Find content bounds
        coords = cv2.findNonZero(binary)
        
        if coords is None or len(coords) == 0:
            # No content found, return original
            return bbox
        
        # Get bounding rectangle of content
        content_x, content_y, content_w, content_h = cv2.boundingRect(coords)
        
        # Add small padding (5 pixels)
        padding = 5
        content_x = max(0, content_x - padding)
        content_y = max(0, content_y - padding)
        content_w = min(crop.shape[1] - content_x, content_w + 2 * padding)
        content_h = min(crop.shape[0] - content_y, content_h + 2 * padding)
        
        # Convert back to page coordinates
        refined_x0 = x0 + content_x
        refined_y0 = y0 + content_y
        refined_x1 = refined_x0 + content_w
        refined_y1 = refined_y0 + content_h
        
        # Create refined bbox
        refined_bbox = bbox.copy()
        refined_bbox['bbox'] = [float(refined_x0), float(refined_y0), 
                               float(refined_x1), float(refined_y1)]
        
        # Log refinement
        original_area = (x1 - x0) * (y1 - y0)
        refined_area = content_w * content_h
        reduction = (1 - refined_area / original_area) * 100 if original_area > 0 else 0
        
        if reduction > 5:  # Only log significant reductions
            logger.debug(f"Refined bbox: reduced area by {reduction:.1f}%")
        
        return refined_bbox
    
    def split_complex_figures(self, bboxes: List[Dict], page_image: np.ndarray) -> List[Dict]:
        """
        Detect and split erroneously merged complex figures.
        
        This method identifies cases where multiple independent figures
        were incorrectly merged into one detection and splits them back.
        
        Args:
            bboxes: List of bbox dicts
            page_image: Page image
            
        Returns:
            List of bboxes with complex figures split if needed
        """
        result = []
        
        for bbox in bboxes:
            # Check if this bbox should be split
            split_bboxes = self._check_and_split_bbox(bbox, page_image)
            
            if len(split_bboxes) > 1:
                logger.info(f"Split bbox into {len(split_bboxes)} parts")
                result.extend(split_bboxes)
            else:
                result.append(bbox)
        
        return result
    
    def _check_and_split_bbox(self, bbox: Dict, page_image: np.ndarray) -> List[Dict]:
        """
        Check if a bbox should be split and perform the split if needed.
        
        A bbox should be split if:
        - It contains multiple disconnected regions
        - The regions are separated by significant white space
        - Each region is large enough to be a separate figure
        
        Args:
            bbox: Bbox dict
            page_image: Page image
            
        Returns:
            List of bbox dicts (original if no split, multiple if split)
        """
        bbox_coords = bbox['bbox']
        x0, y0, x1, y1 = [int(c) for c in bbox_coords]
        h, w = page_image.shape[:2]
        
        # Clamp to image bounds
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        
        if x1 <= x0 or y1 <= y0:
            return [bbox]
        
        # Extract crop
        crop = page_image[y0:y1, x0:x1]
        
        if crop.size == 0:
            return [bbox]
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Filter components by size (must be at least 2% of bbox area or 1000 pixels)
        # Use binary.size (2D) not crop.size (which includes color channels)
        min_area = max(binary.size * 0.02, 1000)
        large_components = []
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                large_components.append(i)
        
        # If we have 2-4 large disconnected components, consider splitting
        if 2 <= len(large_components) <= 4:
            # Check if components are well-separated
            if self._are_components_separated(large_components, stats, crop.shape):
                # Create separate bboxes for each component
                split_bboxes = []
                
                for comp_id in large_components:
                    comp_x = stats[comp_id, cv2.CC_STAT_LEFT]
                    comp_y = stats[comp_id, cv2.CC_STAT_TOP]
                    comp_w = stats[comp_id, cv2.CC_STAT_WIDTH]
                    comp_h = stats[comp_id, cv2.CC_STAT_HEIGHT]
                    
                    # Add padding
                    padding = 10
                    comp_x = max(0, comp_x - padding)
                    comp_y = max(0, comp_y - padding)
                    comp_w = min(crop.shape[1] - comp_x, comp_w + 2 * padding)
                    comp_h = min(crop.shape[0] - comp_y, comp_h + 2 * padding)
                    
                    # Convert to page coordinates
                    split_x0 = x0 + comp_x
                    split_y0 = y0 + comp_y
                    split_x1 = split_x0 + comp_w
                    split_y1 = split_y0 + comp_h
                    
                    # Create new bbox
                    split_bbox = bbox.copy()
                    split_bbox['bbox'] = [float(split_x0), float(split_y0),
                                         float(split_x1), float(split_y1)]
                    split_bbox['split_from'] = 1  # Mark as split
                    
                    split_bboxes.append(split_bbox)
                
                logger.debug(f"Split bbox into {len(split_bboxes)} components")
                return split_bboxes
        
        # No split needed
        return [bbox]
    
    def _are_components_separated(self, component_ids: List[int], 
                                  stats: np.ndarray, shape: Tuple[int, int]) -> bool:
        """
        Check if components are well-separated (not just touching).
        
        Args:
            component_ids: List of component IDs
            stats: Component statistics from connectedComponentsWithStats
            shape: Shape of the image (height, width)
            
        Returns:
            True if components are well-separated
        """
        if len(component_ids) < 2:
            return False
        
        # Get bounding boxes of components
        comp_bboxes = []
        for comp_id in component_ids:
            x = stats[comp_id, cv2.CC_STAT_LEFT]
            y = stats[comp_id, cv2.CC_STAT_TOP]
            w = stats[comp_id, cv2.CC_STAT_WIDTH]
            h = stats[comp_id, cv2.CC_STAT_HEIGHT]
            comp_bboxes.append([x, y, x + w, y + h])
        
        # Check distances between all pairs
        min_separation = float('inf')
        
        for i in range(len(comp_bboxes)):
            for j in range(i + 1, len(comp_bboxes)):
                distance = bbox_utils.bbox_distance(comp_bboxes[i], comp_bboxes[j])
                min_separation = min(min_separation, distance)
        
        # Components should be separated by at least 20 pixels
        return min_separation >= 20
    
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
        Prevents merging boxes that are too spread out or semantically different.
        """
        if len(boxes) <= 1:
            return False
        
        # Calculate merged bbox
        merged_bbox = bbox_utils.merge_bbox_list([box['bbox'] for box in boxes])
        merged_area = bbox_utils.bbox_area(merged_bbox)
        
        # Calculate sum of individual areas
        individual_area = sum(bbox_utils.bbox_area(box['bbox']) for box in boxes)
        
        # Rule 1: If merged area is much larger than sum of parts, probably shouldn't merge
        if merged_area > 3 * individual_area:
            logger.debug(f"Merge rejected: merged area {merged_area} > 3x individual {individual_area}")
            return False
        
        # Rule 2: Check if boxes are too far apart
        max_distance = 0
        for i, box1 in enumerate(boxes):
            for box2 in boxes[i+1:]:
                dist = bbox_utils.bbox_distance(box1['bbox'], box2['bbox'])
                max_distance = max(max_distance, dist)
        
        if max_distance > 150:  # Too far apart
            logger.debug(f"Merge rejected: max distance {max_distance} > 150")
            return False
        
        # Rule 3: Check visual similarity
        try:
            # Extract crops for each box
            crops = []
            for box in boxes:
                x0, y0, x1, y1 = [int(c) for c in box['bbox']]
                h, w = page_image.shape[:2]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                
                if x1 > x0 and y1 > y0:
                    crop = page_image[y0:y1, x0:x1]
                    if crop.size > 0:
                        crops.append(crop)
            
            if len(crops) >= 2:
                # Compare visual features (color histograms)
                hists = []
                for crop in crops:
                    if len(crop.shape) == 3:
                        hist = cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                        hist = cv2.normalize(hist, hist).flatten()
                        hists.append(hist)
                
                if len(hists) >= 2:
                    # Compare first two histograms
                    similarity = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL)
                    
                    # If very different visually, don't merge
                    if similarity < 0.3:
                        logger.debug(f"Merge rejected: low visual similarity {similarity:.3f}")
                        return False
        
        except Exception as e:
            logger.debug(f"Error in visual similarity check: {e}")
        
        return True
    
    def _merge_with_context(self, bboxes: List[Dict], page_image: np.ndarray,
                           captions: Optional[List[Dict]], page_layout: Optional[Dict]) -> List[Dict]:
        """
        Context-aware merging based on title association, layout structure, and visual continuity.
        
        This is a v1.7 enhancement that considers:
        - Title/caption proximity and association
        - Page layout structure (columns, sections)
        - Visual continuity and grouping
        - Type-specific merging strategies
        
        Args:
            bboxes: List of bbox dicts
            page_image: Page image for visual analysis
            captions: Optional list of caption dicts
            page_layout: Optional page layout information
            
        Returns:
            Merged list of bboxes
        """
        if len(bboxes) <= 1:
            return bboxes
        
        # Detect figure types for type-specific merging
        bbox_types = self._detect_figure_types(bboxes, page_image, captions)
        
        # Apply type-specific merging strategies
        if self.enable_type_specific_merge:
            bboxes = self._apply_type_specific_merge(bboxes, bbox_types, page_image, captions, page_layout)
        
        return bboxes
    
    def _detect_figure_types(self, bboxes: List[Dict], page_image: np.ndarray,
                            captions: Optional[List[Dict]]) -> Dict[int, str]:
        """
        Detect the type of each figure for type-specific merging.
        
        Types:
        - 'flowchart': Flowchart or diagram with nodes and connections
        - 'composite': Multiple subfigures arranged in a grid
        - 'table': Table structure
        - 'single': Single standalone figure
        
        Args:
            bboxes: List of bbox dicts
            page_image: Page image
            captions: Optional captions
            
        Returns:
            Dict mapping bbox index to detected type
        """
        bbox_types = {}
        
        for i, bbox in enumerate(bboxes):
            # Default to single
            fig_type = 'single'
            
            # Check if it's a table (based on type field)
            if bbox.get('type') == 'table':
                fig_type = 'table'
            else:
                # Analyze visual features
                try:
                    x0, y0, x1, y1 = [int(c) for c in bbox['bbox']]
                    h, w = page_image.shape[:2]
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(w, x1), min(h, y1)
                    
                    if x1 > x0 and y1 > y0:
                        crop = page_image[y0:y1, x0:x1]
                        
                        if crop.size > 0:
                            # Detect flowchart: many edges and connections
                            if self._is_flowchart(crop):
                                fig_type = 'flowchart'
                            # Detect composite: grid structure
                            elif self._is_composite_figure(crop):
                                fig_type = 'composite'
                
                except Exception as e:
                    logger.debug(f"Error detecting figure type: {e}")
            
            bbox_types[i] = fig_type
            logger.debug(f"Bbox {i} detected as type: {fig_type}")
        
        return bbox_types
    
    def _is_flowchart(self, crop: np.ndarray) -> bool:
        """
        Detect if a crop is a flowchart based on visual features.
        
        Flowcharts typically have:
        - Many edges (boxes and connections)
        - Regular geometric shapes
        - Text inside boxes
        """
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.count_nonzero(edges) / edges.size
            
            # Flowcharts have moderate edge density (lowered threshold for simple flowcharts)
            if 0.005 < edge_ratio < 0.4:
                # Detect lines using Hough transform
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                                       minLineLength=15, maxLineGap=15)
                
                if lines is not None and len(lines) >= 3:
                    # Detect rectangles/boxes using contours on a cleaner binary image
                    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
                    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Count rectangular shapes with reasonable size
                    min_area = crop.size * 0.001  # At least 0.1% of image
                    rectangles = 0
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area < min_area:
                            continue
                            
                        # Approximate contour to polygon
                        peri = cv2.arcLength(contour, True)
                        if peri > 0:
                            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                            
                            # Rectangles have 4 vertices
                            if len(approx) == 4:
                                rectangles += 1
                    
                    # Flowchart: multiple boxes + connecting lines
                    if rectangles >= 2 and len(lines) >= 2:
                        return True
        
        except Exception as e:
            logger.debug(f"Error in flowchart detection: {e}")
        
        return False
    
    def _is_composite_figure(self, crop: np.ndarray) -> bool:
        """
        Detect if a crop is a composite figure (multiple subfigures in a grid).
        
        Composite figures typically have:
        - Regular grid structure
        - Multiple similar-sized regions
        - Separating white space
        """
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            
            # Threshold to binary
            _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            # Filter out small components (noise)
            min_area = crop.size * 0.01  # At least 1% of image
            large_components = [i for i in range(1, num_labels) 
                              if stats[i, cv2.CC_STAT_AREA] > min_area]
            
            # If we have 2-9 large components, might be composite
            if 2 <= len(large_components) <= 9:
                # Check if components are arranged in a grid
                if self._check_grid_arrangement(centroids[large_components]):
                    return True
        
        except Exception as e:
            logger.debug(f"Error in composite figure detection: {e}")
        
        return False
    
    def _check_grid_arrangement(self, centroids: np.ndarray) -> bool:
        """
        Check if centroids are arranged in a regular grid pattern.
        """
        if len(centroids) < 2:
            return False
        
        # Sort by y-coordinate to find rows
        sorted_by_y = sorted(enumerate(centroids), key=lambda x: x[1][1])
        
        # Group into rows (centroids with similar y-coordinates)
        rows = []
        current_row = [sorted_by_y[0]]
        y_threshold = 50  # pixels
        
        for i in range(1, len(sorted_by_y)):
            if abs(sorted_by_y[i][1][1] - current_row[-1][1][1]) < y_threshold:
                current_row.append(sorted_by_y[i])
            else:
                rows.append(current_row)
                current_row = [sorted_by_y[i]]
        
        if current_row:
            rows.append(current_row)
        
        # Check if we have multiple rows with similar number of elements
        if len(rows) >= 2:
            row_sizes = [len(row) for row in rows]
            # Grid should have consistent row sizes
            if max(row_sizes) - min(row_sizes) <= 1:
                # Additional check: columns should also be aligned
                # Sort each row by x-coordinate
                for row in rows:
                    row.sort(key=lambda x: x[1][0])
                
                # Check column alignment (x-coordinates should be similar across rows)
                # Only check if we have at least 2 rows with at least 2 elements each
                if len(rows) >= 2 and all(len(row) >= 2 for row in rows):
                    # Compare x-coordinates of first row with second row
                    x_threshold = 50  # pixels
                    num_cols = min(len(rows[0]), len(rows[1]))
                    
                    aligned_cols = 0
                    for col_idx in range(num_cols):
                        x1 = rows[0][col_idx][1][0]
                        x2 = rows[1][col_idx][1][0]
                        if abs(x1 - x2) <= x_threshold:
                            aligned_cols += 1
                    
                    # At least half of the columns should be aligned
                    if aligned_cols >= num_cols / 2:
                        return True
                elif len(rows) >= 2 and all(len(row) == 1 for row in rows):
                    # Special case: single column grid (all rows have 1 element)
                    # Check if x-coordinates are aligned
                    x_coords = [row[0][1][0] for row in rows]
                    x_variance = max(x_coords) - min(x_coords)
                    if x_variance <= y_threshold:
                        return True
        
        return False
    
    def _apply_type_specific_merge(self, bboxes: List[Dict], bbox_types: Dict[int, str],
                                   page_image: np.ndarray, captions: Optional[List[Dict]],
                                   page_layout: Optional[Dict]) -> List[Dict]:
        """
        Apply type-specific merging strategies.
        
        Args:
            bboxes: List of bbox dicts
            bbox_types: Dict mapping bbox index to type
            page_image: Page image
            captions: Optional captions
            page_layout: Optional page layout
            
        Returns:
            Merged list of bboxes
        """
        # Group bboxes by type
        flowcharts = [i for i, t in bbox_types.items() if t == 'flowchart']
        composites = [i for i, t in bbox_types.items() if t == 'composite']
        tables = [i for i, t in bbox_types.items() if t == 'table']
        
        merged = []
        used = set()
        
        # Strategy 1: Merge flowchart components
        if flowcharts:
            flowchart_groups = self._merge_flowchart_components(
                [bboxes[i] for i in flowcharts], page_image
            )
            for group in flowchart_groups:
                if len(group) > 1:
                    merged_bbox = self._merge_group(group)
                    merged.append(merged_bbox)
                    # Mark original indices as used
                    for bbox in group:
                        for i, orig_bbox in enumerate(bboxes):
                            if orig_bbox is bbox:
                                used.add(i)
                                break
                else:
                    merged.append(group[0])
                    for i, orig_bbox in enumerate(bboxes):
                        if orig_bbox is group[0]:
                            used.add(i)
                            break
        
        # Strategy 2: Merge composite subfigures
        if composites:
            composite_groups = self._merge_composite_subfigures(
                [bboxes[i] for i in composites], page_image, captions
            )
            for group in composite_groups:
                if len(group) > 1:
                    merged_bbox = self._merge_group(group)
                    merged.append(merged_bbox)
                    for bbox in group:
                        for i, orig_bbox in enumerate(bboxes):
                            if orig_bbox is bbox:
                                used.add(i)
                                break
                else:
                    merged.append(group[0])
                    for i, orig_bbox in enumerate(bboxes):
                        if orig_bbox is group[0]:
                            used.add(i)
                            break
        
        # Strategy 3: Keep tables separate (avoid merging with text)
        for i in tables:
            if i not in used:
                merged.append(bboxes[i])
                used.add(i)
        
        # Add remaining bboxes
        for i, bbox in enumerate(bboxes):
            if i not in used:
                merged.append(bbox)
        
        return merged
    
    def _merge_flowchart_components(self, flowchart_bboxes: List[Dict],
                                   page_image: np.ndarray) -> List[List[Dict]]:
        """
        Merge flowchart components that are connected.
        
        Flowcharts should be merged if:
        - They have visual connections (lines, arrows)
        - They are close to each other
        - They form a logical flow
        
        Args:
            flowchart_bboxes: List of flowchart bbox dicts
            page_image: Page image
            
        Returns:
            List of groups, where each group is a list of bboxes to merge
        """
        if len(flowchart_bboxes) <= 1:
            return [[bbox] for bbox in flowchart_bboxes]
        
        # Build connection graph
        connections = []
        for i, bbox1 in enumerate(flowchart_bboxes):
            for j, bbox2 in enumerate(flowchart_bboxes):
                if j <= i:
                    continue
                
                # Check distance
                distance = bbox_utils.bbox_distance(bbox1['bbox'], bbox2['bbox'])
                if distance > self.flowchart_connection_threshold:
                    continue
                
                # Check for visual connection
                if self._has_visual_connection(bbox1['bbox'], bbox2['bbox'], page_image):
                    connections.append((i, j))
        
        # Find connected components
        components = self._find_connected_components(len(flowchart_bboxes), connections)
        
        # Convert to groups of bboxes
        groups = []
        for component in components:
            group = [flowchart_bboxes[i] for i in component]
            groups.append(group)
        
        return groups
    
    def _merge_composite_subfigures(self, composite_bboxes: List[Dict],
                                   page_image: np.ndarray,
                                   captions: Optional[List[Dict]]) -> List[List[Dict]]:
        """
        Merge composite subfigures that belong to the same figure.
        
        Subfigures should be merged if:
        - They share the same caption
        - They are arranged in a grid
        - They are similar in size
        
        Args:
            composite_bboxes: List of composite bbox dicts
            page_image: Page image
            captions: Optional captions
            
        Returns:
            List of groups, where each group is a list of bboxes to merge
        """
        if len(composite_bboxes) <= 1:
            return [[bbox] for bbox in composite_bboxes]
        
        # Group by caption
        caption_groups = {}
        for i, bbox in enumerate(composite_bboxes):
            if captions:
                nearest_caption = self._find_nearest_caption(bbox['bbox'], captions)
                if nearest_caption:
                    caption_id = id(nearest_caption)
                    if caption_id not in caption_groups:
                        caption_groups[caption_id] = []
                    caption_groups[caption_id].append(i)
        
        # For each caption group, check if they form a grid
        groups = []
        used = set()
        
        for caption_id, indices in caption_groups.items():
            if len(indices) >= 2:
                # Check if they form a grid
                boxes = [composite_bboxes[i] for i in indices]
                if self._is_subfigure_group(boxes):
                    groups.append(boxes)
                    used.update(indices)
        
        # Add remaining bboxes as individual groups
        for i, bbox in enumerate(composite_bboxes):
            if i not in used:
                groups.append([bbox])
        
        return groups
    
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

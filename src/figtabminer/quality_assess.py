"""
Quality assessment module for extracted figures and tables.
Evaluates detection quality and filters low-quality results.
"""

from typing import List, Dict, Optional
import numpy as np
import cv2

from . import bbox_utils
from . import utils

logger = utils.setup_logging(__name__)


class QualityAssessor:
    """
    Assess quality of extracted figures and tables.
    Provides multi-dimensional quality scoring.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize quality assessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Quality weights
        self.weights = self.config.get('weights', {
            'detection_conf': 0.3,
            'content_completeness': 0.3,
            'caption_match': 0.2,
            'size_reasonableness': 0.1,
            'position_reasonableness': 0.1
        })
        
        # Thresholds
        self.min_quality_score = self.config.get('min_quality_score', 0.3)
        self.min_ink_ratio = self.config.get('min_ink_ratio', 0.005)
        self.max_page_ratio = self.config.get('max_page_ratio', 0.95)
    
    def assess(self, item: Dict, page_image: Optional[np.ndarray] = None,
               captions: Optional[List[Dict]] = None) -> float:
        """
        Assess quality of an extracted item.
        
        Args:
            item: Item dictionary with bbox, type, score, etc.
            page_image: Optional page image for visual analysis
            captions: Optional list of captions for matching
            
        Returns:
            Quality score between 0 and 1
        """
        scores = {}
        
        # 1. Detection confidence
        scores['detection_conf'] = item.get('detection_score', 0.5)
        
        # 2. Content completeness
        if page_image is not None:
            scores['content_completeness'] = self._assess_content(
                item['bbox'], page_image
            )
        else:
            scores['content_completeness'] = 0.5  # Neutral if no image
        
        # 3. Caption match
        if captions:
            scores['caption_match'] = self._assess_caption_match(
                item, captions
            )
        else:
            scores['caption_match'] = 0.5  # Neutral if no captions
        
        # 4. Size reasonableness
        if page_image is not None:
            scores['size_reasonableness'] = self._assess_size(
                item['bbox'], page_image.shape
            )
        else:
            scores['size_reasonableness'] = 0.7  # Assume reasonable
        
        # 5. Position reasonableness
        if page_image is not None:
            scores['position_reasonableness'] = self._assess_position(
                item['bbox'], page_image.shape
            )
        else:
            scores['position_reasonableness'] = 0.7  # Assume reasonable
        
        # Calculate weighted score
        total_score = sum(scores[k] * self.weights[k] for k in scores)
        
        # Store details in item
        item['quality_score'] = total_score
        item['quality_details'] = scores
        
        logger.debug(f"Quality assessment for {item.get('item_id', 'unknown')}: {total_score:.3f}")
        
        return total_score
    
    def filter_low_quality(self, items: List[Dict], 
                          page_images: Optional[Dict[int, np.ndarray]] = None,
                          captions_by_page: Optional[Dict[int, List[Dict]]] = None) -> List[Dict]:
        """
        Filter out low-quality items.
        
        Args:
            items: List of items to filter
            page_images: Optional dict mapping page_index to page image
            captions_by_page: Optional dict mapping page_index to captions
            
        Returns:
            Filtered list of items
        """
        filtered = []
        
        for item in items:
            page_idx = item.get('page_index', 0)
            page_image = page_images.get(page_idx) if page_images else None
            captions = captions_by_page.get(page_idx) if captions_by_page else None
            
            # Assess quality
            score = self.assess(item, page_image, captions)
            
            # Filter by threshold
            if score >= self.min_quality_score:
                filtered.append(item)
            else:
                logger.debug(f"Filtered low quality item: {item.get('item_id')}, score: {score:.3f}")
        
        logger.info(f"Quality filtering: {len(items)} -> {len(filtered)} items (removed {len(items) - len(filtered)})")
        
        return filtered
    
    def _assess_content(self, bbox: List[float], page_image: np.ndarray) -> float:
        """
        Assess content completeness.
        Checks if the bbox contains meaningful content (not mostly empty).
        """
        try:
            x0, y0, x1, y1 = [int(c) for c in bbox]
            h, w = page_image.shape[:2]
            
            # Clamp to image bounds
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            
            if x1 <= x0 or y1 <= y0:
                return 0.0
            
            crop = page_image[y0:y1, x0:x1]
            
            if crop.size == 0:
                return 0.0
            
            # Convert to grayscale
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop
            
            # Calculate ink ratio (non-white pixels)
            _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
            ink_ratio = np.count_nonzero(binary) / binary.size
            
            # Score based on ink ratio
            if ink_ratio < self.min_ink_ratio:
                return 0.0  # Too empty
            elif ink_ratio > 0.5:
                return 1.0  # Very dense content
            else:
                # Linear scale between min and 0.5
                return (ink_ratio - self.min_ink_ratio) / (0.5 - self.min_ink_ratio)
        
        except Exception as e:
            logger.debug(f"Error assessing content: {e}")
            return 0.5  # Neutral on error
    
    def _assess_caption_match(self, item: Dict, captions: List[Dict]) -> float:
        """
        Assess how well the item matches with a caption.
        """
        if not captions:
            return 0.5  # Neutral if no captions
        
        bbox = item['bbox']
        item_type = item['type']
        
        # Find nearest caption
        min_distance = float('inf')
        best_match = None
        
        for caption in captions:
            if 'bbox' not in caption:
                continue
            
            # Check if caption mentions the right type
            caption_text = caption.get('text', '').lower()
            if item_type == 'figure':
                if 'figure' not in caption_text and 'fig' not in caption_text:
                    continue
            elif item_type == 'table':
                if 'table' not in caption_text and 'tab' not in caption_text:
                    continue
            
            distance = bbox_utils.bbox_distance(bbox, caption['bbox'])
            
            # Prefer captions below the item
            position = bbox_utils.bbox_relative_position(bbox, caption['bbox'])
            if position == 'above':
                distance += 100  # Penalty
            
            if distance < min_distance:
                min_distance = distance
                best_match = caption
        
        if best_match is None:
            return 0.3  # No matching caption found
        
        # Score based on distance
        if min_distance < 50:
            return 1.0  # Very close
        elif min_distance < 150:
            return 0.8  # Close
        elif min_distance < 300:
            return 0.6  # Moderate
        else:
            return 0.4  # Far
    
    def _assess_size(self, bbox: List[float], page_shape: tuple) -> float:
        """
        Assess if the size is reasonable.
        Too small or too large boxes are suspicious.
        """
        page_h, page_w = page_shape[:2]
        page_area = page_w * page_h
        
        bbox_area = bbox_utils.bbox_area(bbox)
        
        if bbox_area == 0:
            return 0.0
        
        ratio = bbox_area / page_area
        
        # Too small (< 0.5% of page)
        if ratio < 0.005:
            return 0.3
        
        # Too large (> 95% of page)
        if ratio > self.max_page_ratio:
            return 0.3
        
        # Reasonable size (1% - 80% of page)
        if 0.01 <= ratio <= 0.8:
            return 1.0
        
        # Moderate size
        return 0.7
    
    def _assess_position(self, bbox: List[float], page_shape: tuple) -> float:
        """
        Assess if the position is reasonable.
        Items too close to page edges might be artifacts.
        """
        page_h, page_w = page_shape[:2]
        
        x0, y0, x1, y1 = bbox
        
        # Calculate margins
        left_margin = x0
        right_margin = page_w - x1
        top_margin = y0
        bottom_margin = page_h - y1
        
        min_margin = min(left_margin, right_margin, top_margin, bottom_margin)
        
        # Score based on minimum margin
        if min_margin < 5:
            return 0.5  # Very close to edge
        elif min_margin < 20:
            return 0.7  # Close to edge
        else:
            return 1.0  # Good margin
    
    def get_statistics(self, items: List[Dict]) -> Dict:
        """
        Get quality statistics for a list of items.
        
        Returns:
            Dictionary with quality statistics
        """
        if not items:
            return {
                'count': 0,
                'mean_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'low_quality_count': 0
            }
        
        scores = [item.get('quality_score', 0.5) for item in items]
        low_quality = [s for s in scores if s < self.min_quality_score]
        
        return {
            'count': len(items),
            'mean_score': np.mean(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'std_score': np.std(scores),
            'low_quality_count': len(low_quality),
            'low_quality_ratio': len(low_quality) / len(items) if items else 0
        }

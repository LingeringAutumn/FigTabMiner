#!/usr/bin/env python3
"""
Bar chart data extraction for v1.3.

Automatically extracts numerical data from bar charts:
- Vertical bar charts
- Horizontal bar charts
- Grouped bar charts (multiple series)

Uses OpenCV for visual analysis + optional OCR for labels.
"""

import cv2
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from . import utils

logger = utils.setup_logging(__name__)


class BarChartDigitizer:
    """Extract data from bar charts."""
    
    def __init__(self, config: dict = None):
        """
        Initialize digitizer.
        
        Args:
            config: Configuration dict with options:
                - min_bar_width: int (minimum bar width in pixels)
                - min_bar_height: int (minimum bar height in pixels)
                - axis_detection_threshold: float (0-1)
                - enable_ocr: bool (use OCR for labels)
        """
        self.config = config or {}
        self.min_bar_width = self.config.get('min_bar_width', 5)
        self.min_bar_height = self.config.get('min_bar_height', 10)
        self.axis_threshold = self.config.get('axis_detection_threshold', 0.5)
        self.enable_ocr = self.config.get('enable_ocr', False)
    
    def digitize(
        self,
        image_path: str,
        orientation: str = 'auto'
    ) -> Optional[pd.DataFrame]:
        """
        Extract data from bar chart.
        
        Args:
            image_path: Path to bar chart image
            orientation: 'vertical', 'horizontal', or 'auto'
        
        Returns:
            DataFrame with columns ['category', 'value'] or None if failed
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Detect orientation if auto
            if orientation == 'auto':
                orientation = self._detect_orientation(gray)
                logger.debug(f"Detected orientation: {orientation}")
            
            # 2. Detect axes
            axes = self._detect_axes(gray, orientation)
            if axes is None:
                logger.warning("Failed to detect axes")
                return None
            
            # 3. Detect bars
            bars = self._detect_bars(gray, orientation, axes)
            if not bars:
                logger.warning("No bars detected")
                return None
            
            logger.info(f"Detected {len(bars)} bars")
            
            # 4. Extract values
            data = self._extract_values(bars, axes, orientation)
            
            # 5. Extract labels (optional, with OCR)
            if self.enable_ocr:
                labels = self._extract_labels_ocr(image_path, bars, orientation)
                if labels and len(labels) == len(data):
                    data['category'] = labels
            
            # 6. Create DataFrame
            df = pd.DataFrame(data)
            
            return df
        
        except Exception as e:
            logger.error(f"Bar chart digitization failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _detect_orientation(self, gray: np.ndarray) -> str:
        """Detect if bars are vertical or horizontal."""
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10
        )
        
        if lines is None:
            return 'vertical'  # Default
        
        # Count vertical and horizontal lines
        vertical_count = 0
        horizontal_count = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
            
            # Vertical line (angle close to π/2)
            if 1.3 < angle < 1.8:
                vertical_count += 1
            # Horizontal line (angle close to 0 or π)
            elif angle < 0.3 or angle > 2.8:
                horizontal_count += 1
        
        # More vertical lines → vertical bars
        # More horizontal lines → horizontal bars
        if vertical_count > horizontal_count * 1.5:
            return 'vertical'
        elif horizontal_count > vertical_count * 1.5:
            return 'horizontal'
        else:
            return 'vertical'  # Default
    
    def _detect_axes(
        self,
        gray: np.ndarray,
        orientation: str
    ) -> Optional[Dict]:
        """
        Detect X and Y axes.
        
        Returns:
            Dict with 'x_axis', 'y_axis' positions, or None
        """
        h, w = gray.shape
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 100, minLineLength=w//3, maxLineGap=20
        )
        
        if lines is None:
            # Fallback: assume axes at image borders
            return {
                'x_axis': h - 50,  # Bottom
                'y_axis': 50,      # Left
                'x_min': 50,
                'x_max': w - 50,
                'y_min': 50,
                'y_max': h - 50
            }
        
        # Find horizontal lines (X-axis candidates)
        h_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
            if angle < 0.2 or angle > 2.9:  # Nearly horizontal
                h_lines.append((y1 + y2) / 2)
        
        # Find vertical lines (Y-axis candidates)
        v_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
            if 1.4 < angle < 1.7:  # Nearly vertical
                v_lines.append((x1 + x2) / 2)
        
        # X-axis is typically at the bottom
        x_axis = max(h_lines) if h_lines else h - 50
        
        # Y-axis is typically at the left
        y_axis = min(v_lines) if v_lines else 50
        
        return {
            'x_axis': int(x_axis),
            'y_axis': int(y_axis),
            'x_min': int(y_axis),
            'x_max': w - 50,
            'y_min': 50,
            'y_max': int(x_axis)
        }
    
    def _detect_bars(
        self,
        gray: np.ndarray,
        orientation: str,
        axes: Dict
    ) -> List[Dict]:
        """
        Detect bar rectangles using multiple strategies.
        
        Returns:
            List of bar dicts with 'bbox' and 'value'
        """
        h, w = gray.shape
        bars = []
        
        # Strategy 1: Threshold-based detection (dark bars)
        bars_dark = self._detect_bars_threshold(gray, orientation, axes, invert=True)
        bars.extend(bars_dark)
        
        # Strategy 2: Threshold-based detection (light bars)
        bars_light = self._detect_bars_threshold(gray, orientation, axes, invert=False)
        bars.extend(bars_light)
        
        # Strategy 3: Edge-based detection
        bars_edge = self._detect_bars_edges(gray, orientation, axes)
        bars.extend(bars_edge)
        
        # Remove duplicates (bars detected by multiple strategies)
        bars = self._deduplicate_bars(bars)
        
        # Sort bars by position
        if orientation == 'vertical':
            bars.sort(key=lambda b: b['center_x'])
        else:
            bars.sort(key=lambda b: b['center_y'])
        
        logger.debug(f"Detected {len(bars)} bars using multi-strategy approach")
        
        return bars
    
    def _detect_bars_threshold(
        self,
        gray: np.ndarray,
        orientation: str,
        axes: Dict,
        invert: bool = True
    ) -> List[Dict]:
        """Detect bars using thresholding."""
        # Adaptive threshold for better handling of varying lighting
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
            11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        bars = []
        
        for cnt in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter by size (more lenient)
            if w < self.min_bar_width or h < self.min_bar_height:
                continue
            
            # Filter by position and aspect ratio
            if self._is_valid_bar(x, y, w, h, orientation, axes):
                bars.append({
                    'bbox': (x, y, w, h),
                    'center_x': x + w // 2,
                    'center_y': y + h // 2,
                    'height': h,
                    'width': w
                })
        
        return bars
    
    def _detect_bars_edges(
        self,
        gray: np.ndarray,
        orientation: str,
        axes: Dict
    ) -> List[Dict]:
        """Detect bars using edge detection."""
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        bars = []
        
        for cnt in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter by size
            if w < self.min_bar_width or h < self.min_bar_height:
                continue
            
            # Filter by position and aspect ratio
            if self._is_valid_bar(x, y, w, h, orientation, axes):
                bars.append({
                    'bbox': (x, y, w, h),
                    'center_x': x + w // 2,
                    'center_y': y + h // 2,
                    'height': h,
                    'width': w
                })
        
        return bars
    
    def _is_valid_bar(
        self,
        x: int, y: int, w: int, h: int,
        orientation: str,
        axes: Dict
    ) -> bool:
        """Check if a bounding box is a valid bar."""
        # Filter by position (should be in plot area)
        if orientation == 'vertical':
            # Bars should be above X-axis and right of Y-axis
            if y + h < axes['y_min'] or y > axes['x_axis']:
                return False
            if x < axes['y_axis'] or x > axes['x_max']:
                return False
            
            # Check aspect ratio (vertical bars are tall) - more lenient
            if h < w * 0.8:  # Changed from 1.5 to 0.8
                return False
        
        else:  # horizontal
            # Bars should be left of right edge and below top
            if x + w < axes['y_axis'] or x > axes['x_max']:
                return False
            if y < axes['y_min'] or y > axes['x_axis']:
                return False
            
            # Check aspect ratio (horizontal bars are wide) - more lenient
            if w < h * 0.8:  # Changed from 1.5 to 0.8
                return False
        
        return True
    
    def _deduplicate_bars(self, bars: List[Dict]) -> List[Dict]:
        """Remove duplicate bars detected by multiple strategies."""
        if not bars:
            return []
        
        unique_bars = []
        
        for bar in bars:
            is_duplicate = False
            
            for existing in unique_bars:
                # Check if bounding boxes overlap significantly
                iou = self._compute_iou(bar['bbox'], existing['bbox'])
                
                if iou > 0.5:  # 50% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_bars.append(bar)
        
        return unique_bars
    
    def _compute_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Compute Intersection over Union of two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Compute intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _extract_values(
        self,
        bars: List[Dict],
        axes: Dict,
        orientation: str
    ) -> Dict:
        """
        Extract numerical values from bars.
        
        Returns:
            Dict with 'category' and 'value' lists
        """
        categories = []
        values = []
        
        for i, bar in enumerate(bars):
            # Category (just index for now, OCR can replace)
            categories.append(f"Bar_{i+1}")
            
            # Value (normalized to 0-100 scale)
            if orientation == 'vertical':
                # Height of bar relative to axis
                bar_top = bar['bbox'][1]
                bar_bottom = bar['bbox'][1] + bar['bbox'][3]
                
                # Value is proportional to height
                axis_range = axes['x_axis'] - axes['y_min']
                bar_height = axes['x_axis'] - bar_top
                
                if axis_range > 0:
                    value = (bar_height / axis_range) * 100
                else:
                    value = 0
            
            else:  # horizontal
                # Width of bar relative to axis
                bar_left = bar['bbox'][0]
                bar_right = bar['bbox'][0] + bar['bbox'][2]
                
                # Value is proportional to width
                axis_range = axes['x_max'] - axes['y_axis']
                bar_width = bar_right - axes['y_axis']
                
                if axis_range > 0:
                    value = (bar_width / axis_range) * 100
                else:
                    value = 0
            
            values.append(round(value, 2))
        
        return {
            'category': categories,
            'value': values
        }
    
    def _extract_labels_ocr(
        self,
        image_path: str,
        bars: List[Dict],
        orientation: str
    ) -> Optional[List[str]]:
        """Extract category labels using OCR."""
        if not self.enable_ocr:
            return None
        
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False)
            
            # Read all text
            results = reader.readtext(image_path)
            
            # Match text to bars based on position
            labels = []
            
            for bar in bars:
                # Find text near bar
                if orientation == 'vertical':
                    # Look below bar (X-axis labels)
                    target_x = bar['center_x']
                    target_y = bar['bbox'][1] + bar['bbox'][3] + 20
                else:
                    # Look left of bar (Y-axis labels)
                    target_x = bar['bbox'][0] - 20
                    target_y = bar['center_y']
                
                # Find closest text
                closest_text = None
                min_dist = float('inf')
                
                for bbox, text, conf in results:
                    # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text_x = (bbox[0][0] + bbox[2][0]) / 2
                    text_y = (bbox[0][1] + bbox[2][1]) / 2
                    
                    dist = np.sqrt((text_x - target_x)**2 + (text_y - target_y)**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_text = text
                
                if closest_text and min_dist < 100:
                    labels.append(closest_text)
                else:
                    labels.append(f"Bar_{len(labels)+1}")
            
            return labels
        
        except Exception as e:
            logger.warning(f"OCR label extraction failed: {e}")
            return None


def digitize_bar_chart(
    image_path: str,
    orientation: str = 'auto',
    config: dict = None
) -> Optional[pd.DataFrame]:
    """
    Extract data from bar chart.
    
    Args:
        image_path: Path to bar chart image
        orientation: 'vertical', 'horizontal', or 'auto'
        config: Configuration dict
    
    Returns:
        DataFrame with columns ['category', 'value'] or None if failed
    """
    try:
        digitizer = BarChartDigitizer(config)
        df = digitizer.digitize(image_path, orientation)
        
        if df is not None and len(df) > 0:
            logger.info(f"Successfully extracted {len(df)} bars")
            return df
        else:
            logger.warning("Bar chart digitization returned no data")
            return None
    
    except Exception as e:
        logger.error(f"Bar chart digitization failed: {e}")
        return None

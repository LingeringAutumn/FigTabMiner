#!/usr/bin/env python3
"""
Enhanced chart type classifier for v1.3.

Supports precise classification of:
- bar_chart (vertical/horizontal)
- pie_chart
- line_plot
- scatter_plot
- heatmap
- box_plot
- microscopy
- diagram
- unknown

Uses keyword matching + visual analysis + optional OCR.
Falls back to ai_enrich.classify_figure_subtype if needed.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict
from . import utils

logger = utils.setup_logging(__name__)


class ChartClassifier:
    """Enhanced chart type classifier with visual analysis."""
    
    # Keyword patterns for each chart type
    CHART_KEYWORDS = {
        'bar_chart': [
            'bar chart', 'bar graph', 'histogram', 'column chart',
            'vertical bar', 'horizontal bar', 'bar plot'
        ],
        'pie_chart': [
            'pie chart', 'pie graph', 'pie diagram', 'donut chart',
            'circular chart'
        ],
        'line_plot': [
            'line chart', 'line graph', 'line plot', 'curve',
            'time series', 'trend'
        ],
        'scatter_plot': [
            'scatter plot', 'scatter chart', 'scatter diagram',
            'scatter graph', 'correlation plot'
        ],
        'heatmap': [
            'heat map', 'heatmap', 'color map', 'intensity map',
            'correlation matrix'
        ],
        'box_plot': [
            'box plot', 'box chart', 'box-and-whisker',
            'boxplot', 'quartile plot'
        ],
        'microscopy': [
            'sem', 'tem', 'afm', 'microscopy', 'micrograph',
            'electron microscope', 'optical microscope'
        ],
        'diagram': [
            'schematic', 'diagram', 'flowchart', 'flow chart',
            'illustration', 'sketch'
        ]
    }
    
    def __init__(self, config: dict = None):
        """
        Initialize classifier.
        
        Args:
            config: Configuration dict with options:
                - enable_visual_analysis: bool
                - enable_ocr_assist: bool
                - visual_weight: float (0-1)
                - keyword_weight: float (0-1)
        """
        self.config = config or {}
        self.enable_visual = self.config.get('enable_visual_analysis', True)
        self.enable_ocr = self.config.get('enable_ocr_assist', False)
        self.visual_weight = self.config.get('visual_weight', 0.6)
        self.keyword_weight = self.config.get('keyword_weight', 0.4)
    
    def classify(
        self,
        image_path: str,
        caption: str = "",
        snippet: str = "",
        ocr_text: str = ""
    ) -> Tuple[str, float, List[str], Dict]:
        """
        Classify chart type.
        
        Args:
            image_path: Path to chart image
            caption: Caption text
            snippet: Snippet text
            ocr_text: OCR text from image (optional)
        
        Returns:
            (chart_type, confidence, matched_keywords, debug_info)
        """
        # Combine all text
        text_combined = f"{caption} {snippet} {ocr_text}".lower()
        
        # 1. Keyword-based classification
        keyword_scores = self._classify_by_keywords(text_combined)
        
        # 2. Visual-based classification
        visual_scores = {}
        if self.enable_visual:
            try:
                visual_scores = self._classify_by_visual(image_path)
            except Exception as e:
                logger.warning(f"Visual classification failed: {e}")
                visual_scores = {k: 0.0 for k in self.CHART_KEYWORDS.keys()}
        else:
            visual_scores = {k: 0.0 for k in self.CHART_KEYWORDS.keys()}
        
        # 3. Combine scores
        combined_scores = {}
        for chart_type in self.CHART_KEYWORDS.keys():
            kw_score = keyword_scores.get(chart_type, 0.0)
            vis_score = visual_scores.get(chart_type, 0.0)
            combined_scores[chart_type] = (
                self.keyword_weight * kw_score +
                self.visual_weight * vis_score
            )
        
        # 4. Determine winner
        best_type = 'unknown'
        max_score = 0.0
        matched_kws = []
        
        for chart_type, score in combined_scores.items():
            if score > max_score:
                max_score = score
                best_type = chart_type
        
        # Collect matched keywords for best type
        if best_type != 'unknown':
            for kw in self.CHART_KEYWORDS[best_type]:
                if kw in text_combined:
                    matched_kws.append(kw)
        
        # Normalize confidence
        confidence = min(1.0, max_score)
        
        # Debug info
        debug = {
            'keyword_scores': keyword_scores,
            'visual_scores': visual_scores,
            'combined_scores': combined_scores,
            'method': 'enhanced_classifier_v1.3'
        }
        
        logger.debug(f"Chart classified as {best_type} (conf={confidence:.2f})")
        
        return best_type, confidence, matched_kws, debug
    
    def _classify_by_keywords(self, text: str) -> Dict[str, float]:
        """Classify based on keyword matching."""
        scores = {k: 0.0 for k in self.CHART_KEYWORDS.keys()}
        
        for chart_type, keywords in self.CHART_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    scores[chart_type] += 1.0
        
        # Normalize by max possible score
        max_kw_count = max(len(kws) for kws in self.CHART_KEYWORDS.values())
        for chart_type in scores:
            scores[chart_type] = scores[chart_type] / max_kw_count
        
        return scores
    
    def _classify_by_visual(self, image_path: str) -> Dict[str, float]:
        """Classify based on visual features."""
        img = cv2.imread(image_path)
        if img is None:
            return {k: 0.0 for k in self.CHART_KEYWORDS.keys()}
        
        scores = {k: 0.0 for k in self.CHART_KEYWORDS.keys()}
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Feature 1: Detect rectangles (for bar charts)
        rect_score = self._detect_rectangles(gray)
        scores['bar_chart'] = rect_score
        
        # Feature 2: Detect circles (for pie charts)
        circle_score = self._detect_circles(gray)
        scores['pie_chart'] = circle_score
        
        # Feature 3: Detect continuous lines (for line plots)
        line_score = self._detect_continuous_lines(gray)
        scores['line_plot'] = line_score
        
        # Feature 4: Detect scattered points (for scatter plots)
        scatter_score = self._detect_scattered_points(gray)
        scores['scatter_plot'] = scatter_score
        
        # Feature 5: Detect grid pattern (for heatmaps)
        grid_score = self._detect_grid_pattern(gray)
        scores['heatmap'] = grid_score
        
        # Feature 6: Detect high texture (for microscopy)
        texture_score = self._detect_high_texture(gray)
        scores['microscopy'] = texture_score
        
        # Feature 7: Detect diagram-like structure
        diagram_score = self._detect_diagram_structure(gray)
        scores['diagram'] = diagram_score
        
        return scores
    
    def _detect_rectangles(self, gray: np.ndarray) -> float:
        """Detect rectangular shapes (bar charts)."""
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Count rectangular contours
        rect_count = 0
        total_area = gray.shape[0] * gray.shape[1]
        
        for cnt in contours:
            # Approximate contour
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
            # Check if it's a rectangle (4 vertices)
            if len(approx) == 4:
                area = cv2.contourArea(cnt)
                # Filter by size (not too small, not too large)
                if 0.001 * total_area < area < 0.3 * total_area:
                    rect_count += 1
        
        # Normalize score
        # Bar charts typically have 3-20 bars
        if rect_count >= 3:
            score = min(1.0, rect_count / 10.0)
        else:
            score = 0.0
        
        return score
    
    def _detect_circles(self, gray: np.ndarray) -> float:
        """Detect circular shapes (pie charts)."""
        # Hough Circle Transform
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=100,
            param2=30,
            minRadius=20,
            maxRadius=min(gray.shape) // 2
        )
        
        if circles is not None:
            # Pie charts typically have 1 large circle
            return min(1.0, len(circles[0]) / 2.0)
        
        return 0.0
    
    def _detect_continuous_lines(self, gray: np.ndarray) -> float:
        """Detect continuous curves (line plots)."""
        # Edge detection
        edges = cv2.Canny(gray, 30, 100)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )
        
        if lines is None:
            return 0.0
        
        # Count non-axis lines (not perfectly horizontal/vertical)
        curve_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
            # Not horizontal (0 or π) and not vertical (π/2)
            if 0.1 < angle < 1.47 or 1.67 < angle < 3.04:
                curve_lines += 1
        
        # Normalize
        return min(1.0, curve_lines / 20.0)
    
    def _detect_scattered_points(self, gray: np.ndarray) -> float:
        """Detect scattered points (scatter plots)."""
        # Threshold to find dark points
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours (potential points)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Count small circular contours
        point_count = 0
        total_area = gray.shape[0] * gray.shape[1]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Small points
            if 0.00001 * total_area < area < 0.001 * total_area:
                # Check circularity
                peri = cv2.arcLength(cnt, True)
                if peri > 0:
                    circularity = 4 * np.pi * area / (peri * peri)
                    if circularity > 0.5:
                        point_count += 1
        
        # Scatter plots typically have 10-100+ points
        if point_count >= 10:
            return min(1.0, point_count / 50.0)
        
        return 0.0
    
    def _detect_grid_pattern(self, gray: np.ndarray) -> float:
        """Detect grid pattern (heatmaps)."""
        # Detect horizontal and vertical lines
        edges = cv2.Canny(gray, 50, 150)
        
        # Horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
        
        # Vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
        
        # Count grid intersections
        grid = cv2.bitwise_and(h_lines, v_lines)
        grid_pixels = np.count_nonzero(grid)
        
        # Normalize
        total_pixels = gray.shape[0] * gray.shape[1]
        grid_ratio = grid_pixels / total_pixels
        
        # Heatmaps have moderate grid structure
        if 0.001 < grid_ratio < 0.1:
            return min(1.0, grid_ratio * 50)
        
        return 0.0
    
    def _detect_high_texture(self, gray: np.ndarray) -> float:
        """Detect high texture (microscopy images)."""
        # Calculate texture using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Microscopy images have high texture variance
        # Normalize (typical range: 0-1000)
        texture_score = min(1.0, variance / 500.0)
        
        # Also check if image fills most of the frame (not much whitespace)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        white_ratio = np.sum(thresh == 255) / thresh.size
        
        if white_ratio < 0.3 and texture_score > 0.5:
            return texture_score
        
        return 0.0
    
    def _detect_diagram_structure(self, gray: np.ndarray) -> float:
        """Detect diagram-like structure (flowcharts, schematics)."""
        # Diagrams have:
        # 1. Multiple shapes (boxes, circles)
        # 2. Connecting lines
        # 3. Moderate whitespace
        
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Count shapes
        shape_count = len([c for c in contours if cv2.contourArea(c) > 100])
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10
        )
        line_count = len(lines) if lines is not None else 0
        
        # Diagrams have multiple shapes and lines
        if shape_count >= 3 and line_count >= 5:
            return min(1.0, (shape_count + line_count) / 30.0)
        
        return 0.0


def classify_chart_type(
    image_path: str,
    caption: str = "",
    snippet: str = "",
    ocr_text: str = "",
    config: dict = None,
    fallback_classifier=None
) -> Tuple[str, float, List[str], Dict]:
    """
    Classify chart type with fallback support.
    
    Args:
        image_path: Path to chart image
        caption: Caption text
        snippet: Snippet text
        ocr_text: OCR text (optional)
        config: Configuration dict
        fallback_classifier: Fallback function (e.g., ai_enrich.classify_figure_subtype)
    
    Returns:
        (chart_type, confidence, matched_keywords, debug_info)
    """
    try:
        classifier = ChartClassifier(config)
        chart_type, confidence, keywords, debug = classifier.classify(
            image_path, caption, snippet, ocr_text
        )
        
        # If confidence is too low and fallback is available, use it
        if confidence < 0.3 and fallback_classifier is not None:
            logger.info("Low confidence, using fallback classifier")
            try:
                fallback_result = fallback_classifier(
                    image_path, caption, snippet, ocr_text
                )
                # fallback returns (subtype, conf, kws, debug)
                return fallback_result
            except Exception as e:
                logger.warning(f"Fallback classifier failed: {e}")
        
        return chart_type, confidence, keywords, debug
    
    except Exception as e:
        logger.error(f"Chart classification failed: {e}")
        
        # Use fallback if available
        if fallback_classifier is not None:
            try:
                return fallback_classifier(image_path, caption, snippet, ocr_text)
            except Exception as e2:
                logger.error(f"Fallback classifier also failed: {e2}")
        
        # Last resort
        return 'unknown', 0.0, [], {'error': str(e)}

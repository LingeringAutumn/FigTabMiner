"""
Content classifier to distinguish between different types of content.
Helps filter out false positives like math equations misidentified as tables.
"""

from typing import Dict, List, Optional
import numpy as np
import cv2
import re

from . import utils
from . import bbox_utils

logger = utils.setup_logging(__name__)


class ContentClassifier:
    """
    Classify content to distinguish:
    - Tables vs Math equations
    - Figures vs Diagrams vs Photos
    - Text blocks vs Other content
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Math equation indicators (English + Chinese)
        self.math_keywords = [
            '=', '≠', '≈', '≤', '≥', '<', '>',
            '+', '-', '×', '÷', '∫', '∑', '∏',
            'α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'π', 'σ', 'φ', 'ω',
            '∂', '∇', '∞', '√', '∈', '∉', '⊂', '⊃',
            '\\frac', '\\sum', '\\int', '\\prod', '\\sqrt',
            '\\alpha', '\\beta', '\\gamma', '\\delta',
            # Chinese punctuation and math terms
            '，', '。', '、', '；', '：', '（', '）', '【', '】',
            '公式', '方程', '函数', '变量', '常数', '系数', '参数'
        ]
        
        # Table indicators (English + Chinese)
        self.table_keywords = [
            'Table', 'Tab.', '表', '表格',
            'Sample', 'Condition', 'Parameter', 'Value', 'Result',
            'Method', 'Material', 'Property', 'Measurement',
            '样品', '条件', '参数', '数值', '结果', '方法', '材料', '性质', '测量'
        ]
        
        # Text paragraph indicators (Chinese + English)
        self.text_patterns = [
            'abstract', 'introduction', 'conclusion', 'discussion',
            'methods', 'results', 'acknowledgment', 'appendix',
            'references', 'bibliography',
            '摘要', '引言', '介绍', '结论', '讨论', '方法', '结果', '致谢', '附录', '参考文献'
        ]
    
    def is_math_equation(self, item: Dict, page_image: Optional[np.ndarray] = None,
                        text_lines: Optional[List[Dict]] = None) -> bool:
        """
        Determine if an item is likely a math equation rather than a table.
        
        Args:
            item: Item dict with bbox, type, etc.
            page_image: Optional page image for visual analysis
            text_lines: Optional text lines from the page
            
        Returns:
            True if likely a math equation
        """
        bbox = item['bbox']
        
        # Check 1: Size - equations are usually smaller and more horizontal
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        aspect_ratio = width / height if height > 0 else 0
        
        # Equations tend to be wide and short
        if aspect_ratio > 4 and height < 100:
            logger.debug(f"Math equation indicator: high aspect ratio {aspect_ratio:.2f}")
            return True
        
        # Check 2: Text content analysis
        if text_lines:
            # Get text near this bbox
            nearby_text = self._get_nearby_text(bbox, text_lines, distance=50)
            
            if nearby_text:
                # Count math symbols
                math_symbol_count = sum(1 for symbol in self.math_keywords 
                                       if symbol in nearby_text)
                
                # Count table keywords
                table_keyword_count = sum(1 for keyword in self.table_keywords 
                                         if keyword.lower() in nearby_text.lower())
                
                # If more math symbols than table keywords, likely equation
                if math_symbol_count > table_keyword_count and math_symbol_count >= 2:
                    logger.debug(f"Math equation indicator: {math_symbol_count} math symbols vs {table_keyword_count} table keywords")
                    return True
        
        # Check 3: Visual analysis
        if page_image is not None:
            try:
                x0, y0, x1, y1 = [int(c) for c in bbox]
                h, w = page_image.shape[:2]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                
                if x1 > x0 and y1 > y0:
                    crop = page_image[y0:y1, x0:x1]
                    
                    # Equations usually have:
                    # - Low density (sparse content)
                    # - Horizontal structure
                    # - Few vertical lines
                    
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
                    
                    # Detect lines
                    edges = cv2.Canny(gray, 50, 150)
                    
                    # Count horizontal vs vertical edges
                    horizontal_kernel = np.ones((1, 5), np.uint8)
                    vertical_kernel = np.ones((5, 1), np.uint8)
                    
                    horizontal_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
                    vertical_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)
                    
                    h_count = np.count_nonzero(horizontal_edges)
                    v_count = np.count_nonzero(vertical_edges)
                    
                    # Equations have more horizontal structure, tables have grid structure
                    if h_count > 0 and v_count > 0:
                        ratio = h_count / v_count
                        if ratio > 3:  # Much more horizontal than vertical
                            logger.debug(f"Math equation indicator: horizontal/vertical ratio {ratio:.2f}")
                            return True
            
            except Exception as e:
                logger.debug(f"Error in visual analysis: {e}")
        
        return False
    
    def is_valid_table(self, item: Dict, page_image: Optional[np.ndarray] = None,
                      text_lines: Optional[List[Dict]] = None) -> bool:
        """
        Determine if an item is a valid table.
        
        Returns:
            True if likely a valid table
        """
        bbox = item['bbox']
        
        # Check 1: Size - tables should be reasonably sized
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        
        if area < 5000:  # Increased from 2000 - too small
            logger.debug(f"Invalid table: too small (area={area})")
            return False
        
        # Check 2: Aspect ratio - tables shouldn't be too extreme
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio > 8 or aspect_ratio < 0.15:  # Stricter limits
            logger.debug(f"Invalid table: extreme aspect ratio {aspect_ratio:.2f}")
            return False
        
        # Check 3: Must have actual table data
        if 'row_count' in item and 'col_count' in item:
            if item['row_count'] < 2 or item['col_count'] < 2:
                logger.debug(f"Invalid table: insufficient rows/cols ({item['row_count']}x{item['col_count']})")
                return False
        else:
            # If no table data extracted, it's not a valid table
            logger.debug(f"Invalid table: no table data extracted")
            return False
        
        # Check 4: Position heuristics - filter header/footer/edge regions
        if page_image is not None:
            h, w = page_image.shape[:2]
            x0, y0, x1, y1 = bbox
            center_y = (y0 + y1) / 2
            center_x = (x0 + x1) / 2
            
            # Header region (top 10%)
            if center_y < h * 0.1 and height < h * 0.05:
                logger.debug(f"Invalid table: in header region")
                return False
            
            # Footer region (bottom 10%)
            if center_y > h * 0.9 and height < h * 0.05:
                logger.debug(f"Invalid table: in footer region")
                return False
            
            # Edge regions (left/right 5%)
            if (center_x < w * 0.05 or center_x > w * 0.95) and width < w * 0.1:
                logger.debug(f"Invalid table: in edge region")
                return False
        
        # Check 5: Text content - should not be mostly references, author info, or section titles
        if text_lines:
            nearby_text = self._get_nearby_text(bbox, text_lines, distance=20)
            
            if nearby_text:
                # Check for reference patterns
                ref_patterns = [
                    r'\[\d+\]',  # [1], [2], etc.
                    r'\(\d{4}\)',  # (2020), (2021), etc.
                    r'et al\.',  # et al.
                    r'@',  # email addresses
                    r'\.edu',  # academic emails
                    r'Department of',
                    r'University',
                    r'Institute',
                    r'References',
                    r'Bibliography'
                ]
                
                ref_count = sum(1 for pattern in ref_patterns 
                               if re.search(pattern, nearby_text, re.IGNORECASE))
                
                if ref_count >= 2:
                    logger.debug(f"Invalid table: looks like references or author info")
                    return False
                
                # Check for text paragraph patterns (Chinese + English)
                text_pattern_count = sum(1 for pattern in self.text_patterns 
                                        if pattern.lower() in nearby_text.lower())
                
                if text_pattern_count >= 1 and height < h * 0.08:
                    logger.debug(f"Invalid table: looks like section title or paragraph")
                    return False
        
        # Check 6: Visual structure - ENHANCED with stricter criteria
        if page_image is not None:
            try:
                x0, y0, x1, y1 = [int(c) for c in bbox]
                h, w = page_image.shape[:2]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                
                if x1 > x0 and y1 > y0:
                    crop = page_image[y0:y1, x0:x1]
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
                    
                    # ENHANCED: Check for continuous text lines (stricter thresholds)
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    
                    # Detect text lines using morphology
                    kernel_width = max(int(gray.shape[1] * 0.02), 5)
                    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
                    dilated = cv2.dilate(binary, h_kernel, iterations=1)
                    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Filter contours to find text lines (STRICTER: 20% width instead of 30%)
                    min_line_width = gray.shape[1] * 0.2  # Reduced from 0.3
                    min_line_height = 5
                    text_lines_found = []
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        if w >= min_line_width and h >= min_line_height:
                            text_lines_found.append((y, y + h))
                    
                    # If we have 3+ continuous text lines, it's likely a paragraph (STRICTER: 30% density instead of 40%)
                    if len(text_lines_found) >= 3:
                        text_lines_found.sort(key=lambda line: line[0])
                        line_gaps = []
                        for i in range(len(text_lines_found) - 1):
                            gap = text_lines_found[i + 1][0] - text_lines_found[i][1]
                            line_gaps.append(gap)
                        
                        if line_gaps:
                            avg_gap = np.mean(line_gaps)
                            std_gap = np.std(line_gaps)
                            
                            # Check for uniform spacing (characteristic of paragraphs)
                            crop_height = gray.shape[0]
                            reasonable_gap_min = max(5, crop_height * 0.05)
                            reasonable_gap_max = min(30, crop_height * 0.15)
                            
                            is_uniform_spacing = std_gap < avg_gap * 0.5 if avg_gap > 0 else False
                            is_reasonable_gap = reasonable_gap_min <= avg_gap <= reasonable_gap_max
                            
                            if is_uniform_spacing and is_reasonable_gap:
                                # Calculate text line density (STRICTER: 30% instead of 40%)
                                total_line_height = sum(line[1] - line[0] for line in text_lines_found)
                                line_density = total_line_height / crop_height if crop_height > 0 else 0
                                
                                if line_density > 0.3:  # Reduced from 0.4
                                    logger.info(f"Invalid table: detected {len(text_lines_found)} continuous text lines (density={line_density:.1%})")
                                    return False
                    
                    # ENHANCED: Check ink density and structure score (STRICTER thresholds)
                    _, binary_inv = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
                    ink_pixels = np.count_nonzero(binary_inv)
                    ink_density = ink_pixels / binary_inv.size if binary_inv.size > 0 else 0
                    
                    # Detect table structure (lines)
                    edges = cv2.Canny(gray, 50, 150)
                    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
                    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
                    h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
                    v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
                    
                    h_line_pixels = np.count_nonzero(h_lines)
                    v_line_pixels = np.count_nonzero(v_lines)
                    structure_score = h_line_pixels + v_line_pixels
                    
                    # STRICTER: High ink density + low structure score = likely text paragraph
                    if ink_density > 0.04 and structure_score < 400:  # Stricter: 0.04 instead of 0.05, 400 instead of 300
                        logger.info(f"Invalid table: high ink density ({ink_density:.1%}) but low structure score ({structure_score})")
                        return False
                    
                    # STRICTER: Extremely high ink density is suspicious
                    if ink_density > 0.08 and structure_score < 800:  # Stricter: 0.08 instead of 0.10, 800 instead of 600
                        logger.info(f"Invalid table: very high ink density ({ink_density:.1%}) with insufficient structure ({structure_score})")
                        return False
                    
                    # STRICTER: Very low structure score
                    if structure_score < 200 and ink_density > 0.015:  # Stricter: 200 instead of 150, 0.015 instead of 0.02
                        logger.info(f"Invalid table: extremely low structure score ({structure_score})")
                        return False
                    
                    # Original grid structure check (STRICTER: 50 instead of 30)
                    horizontal_kernel = np.ones((1, 10), np.uint8)
                    vertical_kernel = np.ones((10, 1), np.uint8)
                    
                    horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
                    vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, vertical_kernel)
                    
                    h_count = np.count_nonzero(horizontal_lines)
                    v_count = np.count_nonzero(vertical_lines)
                    
                    # Tables should have both horizontal and vertical lines (STRICTER: 50 instead of 30)
                    if h_count < 50 and v_count < 50:
                        logger.info(f"Invalid table: insufficient grid structure (h={h_count}, v={v_count})")
                        return False
            
            except Exception as e:
                logger.debug(f"Error in visual validation: {e}")
        
        # Check 7: Math equation check (moved to end, after all other checks)
        if self.is_math_equation(item, page_image, text_lines):
            logger.info(f"Invalid table: detected as math equation")
            return False
        
        return True
    
    def classify_figure_type(self, item: Dict, page_image: Optional[np.ndarray] = None) -> str:
        """
        Classify figure into subtypes: photo, diagram, chart, plot, etc.
        
        Returns:
            Subtype string
        """
        if page_image is None:
            return "unknown"
        
        bbox = item['bbox']
        
        try:
            x0, y0, x1, y1 = [int(c) for c in bbox]
            h, w = page_image.shape[:2]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            
            if x1 <= x0 or y1 <= y0:
                return "unknown"
            
            crop = page_image[y0:y1, x0:x1]
            
            if crop.size == 0:
                return "unknown"
            
            # Convert to grayscale
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            
            # Feature 1: Color variance (photos have high variance)
            if len(crop.shape) == 3:
                color_std = np.std(crop, axis=(0, 1)).mean()
                if color_std > 50:
                    return "photo"
            
            # Feature 2: Edge density (diagrams have moderate edges, photos have many)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / edges.size
            
            if edge_density > 0.15:
                return "photo"
            elif edge_density > 0.05:
                return "diagram"
            else:
                return "chart"
        
        except Exception as e:
            logger.debug(f"Error classifying figure: {e}")
            return "unknown"
    
    def _get_nearby_text(self, bbox: List[float], text_lines: List[Dict], distance: float = 50) -> str:
        """Get text content near a bounding box."""
        nearby_texts = []
        
        for line in text_lines:
            if 'bbox' not in line or 'text' not in line:
                continue
            
            line_bbox = line['bbox']
            dist = bbox_utils.bbox_distance(bbox, line_bbox)
            
            if dist <= distance:
                nearby_texts.append(line['text'])
        
        return ' '.join(nearby_texts)

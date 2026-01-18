"""
Enhanced table extraction with multiple strategies and better data extraction.
"""

import pdfplumber
import pandas as pd
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np
import cv2

from . import config
from . import utils
from . import layout_detect
from . import bbox_merger
from . import content_classifier

logger = utils.setup_logging(__name__)

# Try to import Table Transformer detector
try:
    from .detectors import table_transformer_detector
    TABLE_TRANSFORMER_AVAILABLE = table_transformer_detector.is_available()
except ImportError:
    TABLE_TRANSFORMER_AVAILABLE = False
    logger.debug("Table Transformer detector not available")


class EnhancedTableExtractor:
    """
    Enhanced table extractor with multiple strategies:
    1. Layout-guided extraction (using layout detection)
    2. pdfplumber with optimized settings
    3. Visual table detection (line-based)
    4. Text alignment analysis
    5. Content classification (filter math equations)
    """
    
    def __init__(self, capabilities: Dict):
        self.capabilities = capabilities
        
        # Initialize merger with DISABLED merging for tables
        # Tables should NOT be merged to avoid combining separate tables
        merger_config = {
            'iou_threshold': 0.8,  # Very high threshold - only merge if almost identical
            'overlap_threshold': 0.9,  # Very high - only merge if almost complete overlap
            'distance_threshold': 5,  # Very small - tables must be touching
            'enable_semantic_merge': False,  # Disabled
            'enable_visual_merge': False,  # Disabled
            'enable_noise_filter': False,  # Disabled
            'min_area_threshold': config.MIN_TABLE_AREA
        }
        self.merger = bbox_merger.SmartBBoxMerger(merger_config)
        
        # Initialize content classifier
        self.classifier = content_classifier.ContentClassifier()
        
        # Multiple pdfplumber settings to try
        self.table_settings_variants = [
            # Strategy 1: Lines-based (for tables with borders)
            {
                "name": "lines_strict",
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_tolerance": 3,
                "snap_tolerance": 2,
                "join_tolerance": 2,
                "edge_min_length": 10,
                "min_words_vertical": 1,
                "min_words_horizontal": 1,
            },
            # Strategy 2: Lines-based relaxed
            {
                "name": "lines_relaxed",
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_tolerance": 8,
                "snap_tolerance": 5,
                "join_tolerance": 5,
                "edge_min_length": 5,
                "min_words_vertical": 1,
                "min_words_horizontal": 1,
            },
            # Strategy 3: Text-based (for borderless tables)
            {
                "name": "text_strict",
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "intersection_tolerance": 5,
                "snap_tolerance": 3,
                "join_tolerance": 3,
                "min_words_vertical": 3,
                "min_words_horizontal": 2,
                "text_tolerance": 3,
                "text_x_tolerance": 2,
            },
            # Strategy 4: Text-based relaxed
            {
                "name": "text_relaxed",
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "intersection_tolerance": 8,
                "snap_tolerance": 5,
                "join_tolerance": 5,
                "min_words_vertical": 2,
                "min_words_horizontal": 1,
                "text_tolerance": 5,
                "text_x_tolerance": 3,
            },
            # Strategy 5: Explicit (most aggressive)
            {
                "name": "explicit",
                "vertical_strategy": "explicit",
                "horizontal_strategy": "explicit",
                "explicit_vertical_lines": [],
                "explicit_horizontal_lines": [],
            }
        ]
    
    def extract_tables(self, pdf_path: str, ingest_data: dict) -> list:
        """
        Extract tables using multiple strategies.
        
        Strategies (in order):
        1. Layout-guided extraction (DocLayout-YOLO or PubLayNet)
        2. Table Transformer (specialized for tables)
        3. pdfplumber with multiple settings
        4. Visual line detection
        """
        logger.info("Extracting tables with enhanced extractor...")
        
        doc_id = ingest_data["doc_id"]
        output_dir = config.OUTPUT_DIR / doc_id / "items"
        
        all_tables = []
        table_counter = 0
        
        # Strategy 1: Layout-guided extraction
        if self.capabilities.get("layout"):
            layout_tables = self._extract_with_layout(pdf_path, ingest_data, output_dir)
            all_tables.extend(layout_tables)
            logger.info(f"Layout detection found {len(layout_tables)} tables")
        
        # Strategy 2: Table Transformer (NEW!)
        # Check if enabled in config (use environment variable or default)
        enable_tt = os.getenv('FIGTABMINER_ENABLE_TABLE_TRANSFORMER', 'false').lower() in ('true', '1', 'yes')
        
        if TABLE_TRANSFORMER_AVAILABLE and enable_tt:
            tt_tables = self._extract_with_table_transformer(pdf_path, ingest_data, output_dir)
            all_tables.extend(tt_tables)
            logger.info(f"Table Transformer found {len(tt_tables)} tables")
        elif TABLE_TRANSFORMER_AVAILABLE and not enable_tt:
            logger.info("Table Transformer is available but disabled in config")
        else:
            logger.debug("Table Transformer not available")
        
        # Strategy 3: pdfplumber with multiple settings
        pdfplumber_tables = self._extract_with_pdfplumber_multi(pdf_path, ingest_data, output_dir)
        all_tables.extend(pdfplumber_tables)
        logger.info(f"pdfplumber found {len(pdfplumber_tables)} tables")
        
        # Strategy 4: Visual line detection
        visual_tables = self._extract_with_visual_detection(pdf_path, ingest_data, output_dir)
        all_tables.extend(visual_tables)
        logger.info(f"Visual detection found {len(visual_tables)} tables")
        
        # Deduplicate tables
        unique_tables = self._deduplicate_tables(all_tables)
        logger.info(f"After deduplication: {len(unique_tables)} tables")
        
        # Filter out math equations and invalid tables
        valid_tables = self._filter_invalid_tables(unique_tables, ingest_data)
        logger.info(f"After filtering: {len(valid_tables)} valid tables")
        
        # Assign IDs and rename directories
        for i, table in enumerate(valid_tables, 1):
            new_id = f"table_{i:04d}"
            old_id = table.get('temp_id')
            
            if old_id and old_id != new_id:
                # Rename directory
                old_dir = output_dir / old_id
                new_dir = output_dir / new_id
                
                if old_dir.exists():
                    try:
                        old_dir.rename(new_dir)
                        
                        # Update artifacts paths
                        if 'artifacts' in table:
                            for key, path in table['artifacts'].items():
                                table['artifacts'][key] = path.replace(old_id, new_id)
                    except Exception as e:
                        logger.warning(f"Failed to rename {old_id} to {new_id}: {e}")
            
            table['item_id'] = new_id
            
            # Remove temp_id
            if 'temp_id' in table:
                del table['temp_id']
        
        return valid_tables
    
    def _extract_with_layout(self, pdf_path: str, ingest_data: dict, output_dir: Path) -> list:
        """Extract tables using layout detection."""
        tables = []
        
        layout_boxes_by_page = {}
        for page_idx in range(ingest_data["num_pages"]):
            page_img_path = ingest_data["page_images"][page_idx]
            
            # Get page text for enhanced filtering
            page_text_lines = ingest_data.get("page_text_lines", [[]])[page_idx] if page_idx < len(ingest_data.get("page_text_lines", [])) else []
            page_text = " ".join([line.get("text", "") for line in page_text_lines]) if page_text_lines else None
            
            layout_blocks = layout_detect.detect_layout(page_img_path, page_text)
            table_boxes = [{'bbox': b["bbox"], 'type': 'table', 'score': b.get('score', 0.5)} 
                          for b in layout_blocks if b["type"] == "table"]
            if table_boxes:
                table_boxes = self.merger._merge_by_overlap(table_boxes)
                layout_boxes_by_page[page_idx] = table_boxes
        
        if not layout_boxes_by_page:
            return tables
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, table_boxes in layout_boxes_by_page.items():
                page = pdf.pages[page_idx]
                page_w, page_h = ingest_data["page_sizes"][page_idx]
                zoom = ingest_data["zoom"]
                
                for box_dict in table_boxes:
                    bbox_rendered = box_dict['bbox']
                    x0, y0, x1, y1 = [int(c) for c in bbox_rendered]
                    
                    # Size filter
                    if (x1 - x0) < config.MIN_TABLE_DIM or (y1 - y0) < config.MIN_TABLE_DIM:
                        continue
                    if utils.bbox_area([x0, y0, x1, y1]) < config.MIN_TABLE_AREA:
                        continue
                    
                    # Expand slightly
                    x0, y0, x1, y1 = utils.expand_bbox(
                        [x0, y0, x1, y1], config.TABLE_CROP_PAD,
                        max_w=page_w, max_h=page_h
                    )
                    
                    bbox_pdf = [c / zoom for c in [x0, y0, x1, y1]]
                    
                    # Try to extract table data with multiple strategies
                    table_data, strategy_used = self._extract_table_data_multi(page, bbox_pdf)
                    
                    if not table_data:
                        logger.debug(f"No table data extracted from layout bbox on page {page_idx}")
                        continue
                    
                    # Create table item
                    table_item = self._create_table_item(
                        table_data, page_idx, [x0, y0, x1, y1],
                        ingest_data, output_dir,
                        source="layout",
                        strategy=strategy_used,
                        score=box_dict.get('score', 0.5)
                    )
                    
                    if table_item:
                        tables.append(table_item)
        
        return tables
    
    def _extract_with_table_transformer(self, pdf_path: str, ingest_data: dict, output_dir: Path) -> list:
        """
        Extract tables using Table Transformer.
        
        Table Transformer is specialized for table detection and works well
        with borderless tables that pdfplumber might miss.
        """
        if not TABLE_TRANSFORMER_AVAILABLE:
            return []
        
        tables = []
        
        try:
            from .detectors.table_transformer_detector import TableTransformerDetector
            
            logger.debug("Initializing Table Transformer detector...")
            detector = TableTransformerDetector()
            
            # Process each page
            for page_idx in range(ingest_data["num_pages"]):
                page_img_path = ingest_data["page_images"][page_idx]
                page_w, page_h = ingest_data["page_sizes"][page_idx]
                zoom = ingest_data["zoom"]
                
                # Detect tables
                detections = detector.detect_tables(page_img_path, conf_threshold=0.85)  # Increased from 0.7
                
                if not detections:
                    continue
                
                logger.debug(f"Table Transformer found {len(detections)} tables on page {page_idx}")
                
                # Process each detected table
                with pdfplumber.open(pdf_path) as pdf:
                    page = pdf.pages[page_idx]
                    
                    for det in detections:
                        bbox_rendered = det["bbox"]  # [x0, y0, x1, y1]
                        x0, y0, x1, y1 = [int(c) for c in bbox_rendered]
                        
                        # Size filter
                        if (x1 - x0) < config.MIN_TABLE_DIM or (y1 - y0) < config.MIN_TABLE_DIM:
                            continue
                        if utils.bbox_area([x0, y0, x1, y1]) < config.MIN_TABLE_AREA:
                            continue
                        
                        # Expand slightly
                        x0, y0, x1, y1 = utils.expand_bbox(
                            [x0, y0, x1, y1], config.TABLE_CROP_PAD,
                            max_w=page_w, max_h=page_h
                        )
                        
                        bbox_pdf = [c / zoom for c in [x0, y0, x1, y1]]
                        
                        # Try to extract table data
                        table_data, strategy_used = self._extract_table_data_multi(page, bbox_pdf)
                        
                        if not table_data:
                            logger.debug(f"No table data extracted from Table Transformer bbox on page {page_idx}")
                            continue
                        
                        # Create table item
                        table_item = self._create_table_item(
                            table_data, page_idx, [x0, y0, x1, y1],
                            ingest_data, output_dir,
                            source="table_transformer",
                            strategy=strategy_used,
                            score=det.get('score', 0.7)
                        )
                        
                        if table_item:
                            tables.append(table_item)
            
            logger.debug(f"Table Transformer extraction completed: {len(tables)} tables")
            
        except Exception as e:
            logger.warning(f"Table Transformer extraction failed: {e}")
            logger.debug("Traceback:", exc_info=True)
        
        return tables
    
    def _extract_with_pdfplumber_multi(self, pdf_path: str, ingest_data: dict, output_dir: Path) -> list:
        """Extract tables using pdfplumber with multiple settings."""
        tables = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                # Try each setting variant
                best_tables = []
                best_count = 0
                best_strategy = None
                
                for settings in self.table_settings_variants:
                    strategy_name = settings.get("name", "unknown")
                    # Create a copy without the "name" key for pdfplumber
                    settings_copy = {k: v for k, v in settings.items() if k != "name"}
                    
                    try:
                        found_tables = page.find_tables(table_settings=settings_copy)
                        
                        if len(found_tables) > best_count:
                            best_tables = found_tables
                            best_count = len(found_tables)
                            best_strategy = strategy_name
                    
                    except Exception as e:
                        logger.debug(f"Strategy {strategy_name} failed: {e}")
                        continue
                
                if not best_tables:
                    continue
                
                logger.debug(f"Page {page_idx}: Best strategy '{best_strategy}' found {len(best_tables)} tables")
                
                # Extract data from found tables
                zoom = ingest_data["zoom"]
                
                # Load page image for bbox shrinking
                page_img_path = ingest_data["page_images"][page_idx]
                page_img = cv2.imread(page_img_path)
                
                for t_obj in best_tables:
                    table_data = t_obj.extract()
                    
                    if not table_data or not any(any(cell for cell in row) for row in table_data):
                        continue
                    
                    bbox_pdf = list(t_obj.bbox)
                    bbox_rendered = [c * zoom for c in bbox_pdf]
                    
                    # CRITICAL: Shrink bbox BEFORE creating table item
                    # pdfplumber's bbox often includes surrounding text
                    if page_img is not None:
                        bbox_rendered = self._shrink_table_bbox(bbox_rendered, page_img)
                        logger.debug(f"Shrunk pdfplumber bbox from {[c * zoom for c in bbox_pdf]} to {bbox_rendered}")
                    
                    table_item = self._create_table_item(
                        table_data, page_idx, bbox_rendered,
                        ingest_data, output_dir,
                        source="pdfplumber",
                        strategy=best_strategy,
                        score=0.7
                    )
                    
                    if table_item:
                        tables.append(table_item)
        
        return tables
    
    def _extract_with_visual_detection(self, pdf_path: str, ingest_data: dict, output_dir: Path) -> list:
        """
        Extract tables using visual line detection.
        Useful for tables with clear borders.
        """
        tables = []
        
        for page_idx in range(ingest_data["num_pages"]):
            page_img_path = ingest_data["page_images"][page_idx]
            
            try:
                page_img = cv2.imread(page_img_path)
                if page_img is None:
                    continue
                
                # Detect table regions using lines
                table_regions = self._detect_table_regions_visual(page_img)
                
                if not table_regions:
                    continue
                
                logger.debug(f"Visual detection found {len(table_regions)} potential tables on page {page_idx}")
                
                # Try to extract data from each region
                with pdfplumber.open(pdf_path) as pdf:
                    page = pdf.pages[page_idx]
                    zoom = ingest_data["zoom"]
                    
                    for bbox_rendered in table_regions:
                        bbox_pdf = [c / zoom for c in bbox_rendered]
                        
                        table_data, strategy = self._extract_table_data_multi(page, bbox_pdf)
                        
                        if table_data:
                            table_item = self._create_table_item(
                                table_data, page_idx, bbox_rendered,
                                ingest_data, output_dir,
                                source="visual",
                                strategy=strategy,
                                score=0.6
                            )
                            
                            if table_item:
                                tables.append(table_item)
            
            except Exception as e:
                logger.debug(f"Visual detection failed on page {page_idx}: {e}")
                continue
        
        return tables
    
    def _detect_table_regions_visual(self, page_img: np.ndarray) -> List[List[float]]:
        """
        Detect table regions using line detection.
        """
        gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # Detect horizontal lines
        horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        horizontal_lines = cv2.threshold(horizontal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Detect vertical lines
        vertical = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        vertical_lines = cv2.threshold(vertical, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Combine lines
        table_mask = cv2.add(horizontal_lines, vertical_lines)
        
        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if w < 100 or h < 50:
                continue
            
            # Filter by aspect ratio (tables are usually wider than tall)
            aspect_ratio = w / h
            if aspect_ratio < 0.5 or aspect_ratio > 20:
                continue
            
            regions.append([float(x), float(y), float(x + w), float(y + h)])
        
        return regions
    
    def _extract_table_data_multi(self, page, bbox_pdf: List[float]) -> Tuple[Optional[List[List]], Optional[str]]:
        """
        Try multiple strategies to extract table data from a region.
        Returns (table_data, strategy_name) or (None, None).
        """
        cropped = page.crop(bbox_pdf)
        
        # Try each strategy
        for settings in self.table_settings_variants:
            strategy_name = settings.get("name", "unknown")
            settings_copy = {k: v for k, v in settings.items() if k != "name"}
            
            try:
                table = cropped.extract_table(table_settings=settings_copy)
                
                if table and any(any(cell for cell in row) for row in table):
                    logger.debug(f"Successfully extracted table with strategy: {strategy_name}")
                    return table, strategy_name
            
            except Exception as e:
                logger.debug(f"Strategy {strategy_name} failed: {e}")
                continue
        
        return None, None
    
    def _create_table_item(self, table_data: List[List], page_idx: int, bbox_rendered: List[float],
                          ingest_data: dict, output_dir: Path, source: str, strategy: str, score: float) -> Optional[Dict]:
        """Create a table item with all metadata."""
        try:
            # Shrink bbox to avoid including surrounding text
            page_img_path = ingest_data["page_images"][page_idx]
            page_img = cv2.imread(page_img_path)
            
            if page_img is not None:
                bbox_rendered = self._shrink_table_bbox(bbox_rendered, page_img)
            
            # Create item directory (temporary ID)
            temp_id = f"temp_{page_idx}_{int(bbox_rendered[0])}_{int(bbox_rendered[1])}"
            item_dir = utils.ensure_dir(output_dir / temp_id)
            csv_path = item_dir / "table.csv"
            preview_path = item_dir / "preview.png"
            
            # Save CSV
            df = pd.DataFrame(table_data)
            df.to_csv(csv_path, index=False, header=False)
            
            # Create preview
            try:
                if page_img is not None:
                    h, w = page_img.shape[:2]
                    x0, y0, x1, y1 = [int(c) for c in bbox_rendered]
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(w, x1), min(h, y1)
                    
                    if x1 > x0 and y1 > y0:
                        crop = page_img[y0:y1, x0:x1]
                        cv2.imwrite(str(preview_path), crop)
            except Exception as e:
                logger.debug(f"Failed to create table preview: {e}")
            
            # Create item
            zoom = ingest_data["zoom"]
            item = {
                "temp_id": temp_id,
                "type": "table",
                "subtype": "table",
                "page_index": page_idx,
                "bbox": bbox_rendered,
                "pdf_bbox": [c / zoom for c in bbox_rendered],
                "detection_score": score,
                "detection_source": source,
                "extraction_strategy": strategy,
                "row_count": len(table_data),
                "col_count": len(table_data[0]) if table_data else 0,
                "artifacts": {
                    "table_csv": f"items/{temp_id}/table.csv",
                    "preview_png": f"items/{temp_id}/preview.png"
                }
            }
            
            return item
        
        except Exception as e:
            logger.error(f"Error creating table item: {e}")
            return None
    
    def _shrink_table_bbox(self, bbox: List[float], page_img: np.ndarray, shrink_ratio: float = 0.12) -> List[float]:
        """
        AGGRESSIVELY shrink table bounding box to remove surrounding text.
        
        Strategy:
        1. First try structure-based shrinking (detect table lines)
        2. If that fails, use projection-based shrinking
        3. Apply multiple passes to ensure all surrounding text is removed
        
        Args:
            bbox: Original bounding box [x0, y0, x1, y1]
            page_img: Page image
            shrink_ratio: Ratio to shrink (default 12%, very aggressive)
            
        Returns:
            Shrunk bounding box
        """
        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0
        
        # Extract crop
        h, w = page_img.shape[:2]
        crop_x0, crop_y0 = max(0, int(x0)), max(0, int(y0))
        crop_x1, crop_y1 = min(w, int(x1)), min(h, int(y1))
        
        if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            return bbox
        
        crop = page_img[crop_y0:crop_y1, crop_x0:crop_x1]
        
        if crop.size == 0:
            return bbox
        
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        
        # PASS 1: Structure-based shrinking (detect table lines)
        edges = cv2.Canny(gray, 50, 150)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))  # Longer kernel for better line detection
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
        
        # Combine lines to get table structure
        table_structure = cv2.add(h_lines, v_lines)
        
        # Dilate to connect nearby lines
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        table_structure = cv2.dilate(table_structure, dilate_kernel, iterations=3)
        
        # Find the bounding box of the table structure
        coords = cv2.findNonZero(table_structure)
        
        if coords is not None and len(coords) > 100:  # Need sufficient structure pixels
            # Get bounding rect of table structure
            struct_x, struct_y, struct_w, struct_h = cv2.boundingRect(coords)
            
            # Add minimal padding
            padding = max(20, int(min(width, height) * shrink_ratio))
            
            new_y0 = y0 + max(0, struct_y - padding)
            new_y1 = y0 + min(crop.shape[0], struct_y + struct_h + padding)
            new_x0 = x0 + max(0, struct_x - padding)
            new_x1 = x0 + min(crop.shape[1], struct_x + struct_w + padding)
            
            # Ensure valid bbox
            if new_x1 > new_x0 and new_y1 > new_y0:
                # Check if shrinking is reasonable (keep at least 40% of original area)
                new_area = (new_x1 - new_x0) * (new_y1 - new_y0)
                old_area = width * height
                
                if new_area > old_area * 0.4:  # Reduced from 0.5
                    logger.info(f"Shrunk table bbox (structure-based) from {bbox} to [{new_x0:.0f}, {new_y0:.0f}, {new_x1:.0f}, {new_y1:.0f}] (area: {old_area:.0f} -> {new_area:.0f}, {new_area/old_area:.1%})")
                    return [new_x0, new_y0, new_x1, new_y1]
        
        # PASS 2: Projection-based shrinking (fallback)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal projection (sum along x-axis)
        h_projection = np.sum(binary, axis=1)
        # Vertical projection (sum along y-axis)
        v_projection = np.sum(binary, axis=0)
        
        # Find first and last rows/cols with significant content (VERY STRICT: 25% instead of 20%)
        threshold_h = np.max(h_projection) * 0.25  # Increased from 0.20
        threshold_v = np.max(v_projection) * 0.25  # Increased from 0.20
        
        # Find top boundary (skip empty rows at top)
        top_idx = 0
        for i in range(len(h_projection)):
            if h_projection[i] > threshold_h:
                top_idx = i
                break
        
        # Find bottom boundary (skip empty rows at bottom)
        bottom_idx = len(h_projection) - 1
        for i in range(len(h_projection) - 1, -1, -1):
            if h_projection[i] > threshold_h:
                bottom_idx = i
                break
        
        # Find left boundary
        left_idx = 0
        for i in range(len(v_projection)):
            if v_projection[i] > threshold_v:
                left_idx = i
                break
        
        # Find right boundary
        right_idx = len(v_projection) - 1
        for i in range(len(v_projection) - 1, -1, -1):
            if v_projection[i] > threshold_v:
                right_idx = i
                break
        
        # Apply shrinking with larger padding
        padding = max(20, int(min(width, height) * shrink_ratio))
        
        new_y0 = y0 + max(0, top_idx - padding)
        new_y1 = y0 + min(crop.shape[0], bottom_idx + padding)
        new_x0 = x0 + max(0, left_idx - padding)
        new_x1 = x0 + min(crop.shape[1], right_idx + padding)
        
        # Ensure valid bbox
        if new_x1 > new_x0 and new_y1 > new_y0:
            # Check if shrinking is reasonable (keep at least 40% of original area)
            new_area = (new_x1 - new_x0) * (new_y1 - new_y0)
            old_area = width * height
            
            if new_area > old_area * 0.4:  # Reduced from 0.5
                logger.info(f"Shrunk table bbox (projection-based) from {bbox} to [{new_x0:.0f}, {new_y0:.0f}, {new_x1:.0f}, {new_y1:.0f}] (area: {old_area:.0f} -> {new_area:.0f}, {new_area/old_area:.1%})")
                return [new_x0, new_y0, new_x1, new_y1]
        
        logger.warning(f"Could not shrink table bbox {bbox} - keeping original")
        return bbox
    
    def _deduplicate_tables(self, tables: List[Dict]) -> List[Dict]:
        """
        Remove duplicate table detections.
        
        STRICT deduplication to avoid merging separate tables.
        Only removes if IoU > 0.7 (very high overlap).
        """
        if not tables:
            return []
        
        # Group by page
        by_page = {}
        for table in tables:
            page_idx = table['page_index']
            if page_idx not in by_page:
                by_page[page_idx] = []
            by_page[page_idx].append(table)
        
        unique = []
        
        for page_idx, page_tables in by_page.items():
            # Sort by score (descending)
            page_tables.sort(key=lambda t: t.get('detection_score', 0), reverse=True)
            
            kept = []
            for table in page_tables:
                # Check if overlaps with any kept table
                is_duplicate = False
                for kept_table in kept:
                    iou = utils.bbox_iou(table['bbox'], kept_table['bbox'])
                    
                    # STRICT: Only consider duplicate if IoU > 0.7 (very high overlap)
                    # This prevents merging separate tables that are close together
                    if iou > 0.7:  # Increased from 0.5
                        is_duplicate = True
                        # Keep the one with more rows/cols
                        if (table.get('row_count', 0) * table.get('col_count', 0) >
                            kept_table.get('row_count', 0) * kept_table.get('col_count', 0)):
                            # Replace with better one
                            kept.remove(kept_table)
                            kept.append(table)
                            logger.debug(f"Replaced duplicate table (IoU={iou:.2f})")
                        else:
                            logger.debug(f"Skipped duplicate table (IoU={iou:.2f})")
                        break
                
                if not is_duplicate:
                    kept.append(table)
            
            unique.extend(kept)
        
        logger.info(f"Deduplication: {len(tables)} -> {len(unique)} tables")
        return unique
    
    def _filter_invalid_tables(self, tables: List[Dict], ingest_data: dict) -> List[Dict]:
        """Filter out math equations and other invalid tables."""
        valid = []
        
        for table in tables:
            page_idx = table['page_index']
            
            # Load page image
            try:
                page_img_path = ingest_data["page_images"][page_idx]
                page_img = cv2.imread(page_img_path)
            except:
                page_img = None
            
            # Get text lines
            text_lines = ingest_data.get("page_text_lines", [[]])[page_idx] if page_idx < len(ingest_data.get("page_text_lines", [])) else []
            
            # Check if valid table
            if self.classifier.is_valid_table(table, page_img, text_lines):
                valid.append(table)
            else:
                logger.info(f"Filtered invalid table on page {page_idx}: likely math equation or false positive")
        
        return valid


def extract_tables(pdf_path: str, ingest_data: dict, capabilities: dict) -> list:
    """
    Main entry point for table extraction.
    Uses enhanced extractor with multiple strategies.
    """
    extractor = EnhancedTableExtractor(capabilities)
    return extractor.extract_tables(pdf_path, ingest_data)

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
        
        # Initialize merger
        merger_config = {
            'iou_threshold': config.TABLE_MERGE_IOU,
            'overlap_threshold': 0.7,
            'distance_threshold': 30,
            'enable_semantic_merge': False,
            'enable_visual_merge': False,
            'enable_noise_filter': False,
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
        
        # Strategy 2: pdfplumber with multiple settings
        pdfplumber_tables = self._extract_with_pdfplumber_multi(pdf_path, ingest_data, output_dir)
        all_tables.extend(pdfplumber_tables)
        logger.info(f"pdfplumber found {len(pdfplumber_tables)} tables")
        
        # Strategy 3: Visual line detection
        visual_tables = self._extract_with_visual_detection(pdf_path, ingest_data, output_dir)
        all_tables.extend(visual_tables)
        logger.info(f"Visual detection found {len(visual_tables)} tables")
        
        # Deduplicate tables
        unique_tables = self._deduplicate_tables(all_tables)
        logger.info(f"After deduplication: {len(unique_tables)} tables")
        
        # Filter out math equations and invalid tables
        valid_tables = self._filter_invalid_tables(unique_tables, ingest_data)
        logger.info(f"After filtering: {len(valid_tables)} valid tables")
        
        # Assign IDs
        for i, table in enumerate(valid_tables, 1):
            table['item_id'] = f"table_{i:04d}"
        
        return valid_tables
    
    def _extract_with_layout(self, pdf_path: str, ingest_data: dict, output_dir: Path) -> list:
        """Extract tables using layout detection."""
        tables = []
        
        layout_boxes_by_page = {}
        for page_idx in range(ingest_data["num_pages"]):
            page_img_path = ingest_data["page_images"][page_idx]
            layout_blocks = layout_detect.detect_layout(page_img_path)
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
                    strategy_name = settings.pop("name")
                    
                    try:
                        found_tables = page.find_tables(table_settings=settings)
                        
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
                
                for t_obj in best_tables:
                    table_data = t_obj.extract()
                    
                    if not table_data or not any(any(cell for cell in row) for row in table_data):
                        continue
                    
                    bbox_pdf = list(t_obj.bbox)
                    bbox_rendered = [c * zoom for c in bbox_pdf]
                    
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
                page_img_path = ingest_data["page_images"][page_idx]
                page_img = cv2.imread(page_img_path)
                
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
    
    def _deduplicate_tables(self, tables: List[Dict]) -> List[Dict]:
        """Remove duplicate table detections."""
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
                    if iou > 0.5:  # Significant overlap
                        is_duplicate = True
                        # Keep the one with more rows/cols
                        if (table.get('row_count', 0) * table.get('col_count', 0) >
                            kept_table.get('row_count', 0) * kept_table.get('col_count', 0)):
                            # Replace with better one
                            kept.remove(kept_table)
                            kept.append(table)
                        break
                
                if not is_duplicate:
                    kept.append(table)
            
            unique.extend(kept)
        
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

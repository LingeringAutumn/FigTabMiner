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
        
        # Initialize merger with COMPLETELY DISABLED merging for tables
        # Tables should NEVER be merged to avoid combining separate tables
        merger_config = {
            'iou_threshold': 0.95,  # Extremely high - only merge if almost identical
            'overlap_threshold': 0.95,  # Extremely high - only merge if almost complete overlap
            'distance_threshold': 2,  # Extremely small - tables must be touching
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

    def _build_caption_candidates(self, text_lines: List[Dict]) -> List[Dict]:
        if not text_lines:
            return []
        import re
        caption_re = re.compile(r"^\s*(fig\.?|figure|table|tab\.?|scheme|chart|图|表)\b", re.IGNORECASE)
        figure_re = re.compile(r"^\s*(fig\.?|figure|scheme|chart|图)\b", re.IGNORECASE)
        table_re = re.compile(r"^\s*(table|tab\.?|表)\b", re.IGNORECASE)
        candidates = []
        for line in text_lines:
            text = line.get("text", "").strip()
            if not text or not caption_re.match(text):
                continue
            kind = None
            if figure_re.match(text):
                kind = "figure"
            elif table_re.match(text):
                kind = "table"
            if kind:
                candidates.append({"bbox": line["bbox"], "kind": kind, "text": text})
        return candidates

    def _nearest_caption_kind(self, bbox: List[float], candidates: List[Dict], prefer_kind: str) -> Tuple[Optional[str], float]:
        if not candidates:
            return None, float("inf")
        best_kind = None
        best_score = float("inf")
        best_dist = float("inf")
        for cand in candidates:
            cand_bbox = cand["bbox"]
            if cand_bbox[1] > bbox[3]:
                dist = cand_bbox[1] - bbox[3]
                direction = "below"
            elif cand_bbox[3] < bbox[1]:
                dist = bbox[1] - cand_bbox[3]
                direction = "above"
            else:
                dist = 0
                direction = "overlap"
            score = dist
            if prefer_kind == "figure" and direction == "above":
                score += config.CAPTION_DIRECTION_PENALTY
            if prefer_kind == "table" and direction == "below":
                score += config.CAPTION_DIRECTION_PENALTY
            if utils.bbox_overlap_ratio(bbox, cand_bbox) < 0.1:
                score += 50
            if score < best_score:
                best_score = score
                best_kind = cand["kind"]
                best_dist = dist
        if best_dist > config.CAPTION_SEARCH_WINDOW:
            return None, float("inf")
        return best_kind, best_dist
    
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
        
        # Strategy 2: Table Transformer
        enable_tt = config.TABLE_EXTRACTION_ENABLE_TABLE_TRANSFORMER
        if "FIGTABMINER_ENABLE_TABLE_TRANSFORMER" in os.environ:
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
        if config.TABLE_EXTRACTION_ENABLE_VISUAL_DETECTION:
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
            caption_candidates = self._build_caption_candidates(page_text_lines)
            
            layout_blocks = layout_detect.detect_layout(page_img_path, page_text)
            table_boxes = []
            for b in layout_blocks:
                if b["type"] == "table":
                    kind, dist = self._nearest_caption_kind(b["bbox"], caption_candidates, prefer_kind="table")
                    if kind == "figure" and dist < config.CAPTION_SEARCH_WINDOW * 0.7:
                        continue
                    table_boxes.append({'bbox': b["bbox"], 'type': 'table', 'score': b.get('score', 0.5)})
                elif b["type"] == "figure":
                    kind, dist = self._nearest_caption_kind(b["bbox"], caption_candidates, prefer_kind="table")
                    if kind == "table" and dist < config.CAPTION_SEARCH_WINDOW * 0.7:
                        table_boxes.append({'bbox': b["bbox"], 'type': 'table', 'score': b.get('score', 0.5)})
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
                text_lines = ingest_data.get("page_text_lines", [[]])[page_idx] if page_idx < len(ingest_data.get("page_text_lines", [])) else []
                caption_candidates = self._build_caption_candidates(text_lines)
                
                for box_dict in table_boxes:
                    bbox_rendered = box_dict['bbox']
                    x0, y0, x1, y1 = [int(c) for c in bbox_rendered]
                    
                    # Size filter
                    if (x1 - x0) < config.MIN_TABLE_DIM or (y1 - y0) < config.MIN_TABLE_DIM:
                        continue
                    if utils.bbox_area([x0, y0, x1, y1]) < config.MIN_TABLE_AREA:
                        continue
                    
                    # DUAL PATH APPROACH:
                    # Path 1: Expand bbox for data extraction (ensure completeness)
                    # Path 2: Shrink bbox for preview image (remove surrounding text)
                    
                    # Path 1: Expand for data extraction
                    bbox_expanded = utils.expand_bbox(
                        [x0, y0, x1, y1], config.TABLE_CROP_PAD,
                        max_w=page_w, max_h=page_h
                    )
                    bbox_pdf_expanded = [c / zoom for c in bbox_expanded]
                    
                    # Try to extract table data with EXPANDED bbox
                    table_data, strategy_used = self._extract_table_data_multi(page, bbox_pdf_expanded)
                    
                    if not table_data:
                        logger.debug(f"No table data extracted from layout bbox on page {page_idx}")
                        continue
                    
                    # Path 2: Shrink for preview (and validation)
                    page_img_path = ingest_data["page_images"][page_idx]
                    page_img = cv2.imread(page_img_path)
                    
                    bbox_shrunk = [x0, y0, x1, y1]  # Default to original
                    if page_img is not None:
                        try:
                            bbox_shrunk = self._shrink_table_bbox([x0, y0, x1, y1], page_img, text_lines)
                            
                            # CRITICAL: Validate shrunk bbox
                            # If shrinking reduced area by more than 70%, the detection is likely wrong
                            original_area = (x1 - x0) * (y1 - y0)
                            shrunk_area = (bbox_shrunk[2] - bbox_shrunk[0]) * (bbox_shrunk[3] - bbox_shrunk[1])
                            shrink_ratio = shrunk_area / original_area if original_area > 0 else 0
                            
                            if shrink_ratio < 0.3:
                                logger.warning(f"Shrunk bbox too much (ratio={shrink_ratio:.1%}), likely false positive, skipping")
                                continue
                            
                            logger.info(f"Shrunk layout bbox from {[x0, y0, x1, y1]} to {bbox_shrunk} (ratio={shrink_ratio:.1%})")
                        except Exception as e:
                            logger.warning(f"Failed to shrink bbox: {e}, using original")
                    
                    # Create table item with DUAL paths:
                    # - table_data from expanded bbox
                    # - preview from shrunk bbox
                    table_item = self._create_table_item_dual_path(
                        table_data, page_idx, bbox_shrunk, bbox_expanded,
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
                text_lines = ingest_data.get("page_text_lines", [[]])[page_idx] if page_idx < len(ingest_data.get("page_text_lines", [])) else []
                caption_candidates = self._build_caption_candidates(text_lines)
                
                # Detect tables
                detections = detector.detect_tables(
                    page_img_path,
                    conf_threshold=config.TABLE_EXTRACTION_TABLE_TRANSFORMER_CONFIDENCE
                )
                
                if not detections:
                    continue
                
                logger.debug(f"Table Transformer found {len(detections)} tables on page {page_idx}")
                
                # Process each detected table
                with pdfplumber.open(pdf_path) as pdf:
                    page = pdf.pages[page_idx]
                    
                    for det in detections:
                        bbox_rendered = det["bbox"]  # [x0, y0, x1, y1]
                        x0, y0, x1, y1 = [int(c) for c in bbox_rendered]
                        kind, dist = self._nearest_caption_kind(bbox_rendered, caption_candidates, prefer_kind="table")
                        if kind == "figure" and dist < config.CAPTION_SEARCH_WINDOW * 0.7:
                            continue
                        
                        # Size filter
                        if (x1 - x0) < config.MIN_TABLE_DIM or (y1 - y0) < config.MIN_TABLE_DIM:
                            continue
                        if utils.bbox_area([x0, y0, x1, y1]) < config.MIN_TABLE_AREA:
                            continue
                        
                        # DUAL PATH APPROACH:
                        # Path 1: Expand for data extraction
                        bbox_expanded = utils.expand_bbox(
                            [x0, y0, x1, y1], config.TABLE_CROP_PAD,
                            max_w=page_w, max_h=page_h
                        )
                        bbox_pdf_expanded = [c / zoom for c in bbox_expanded]
                        
                        # Extract data with expanded bbox
                        table_data, strategy_used = self._extract_table_data_multi(page, bbox_pdf_expanded)
                        
                        if not table_data:
                            logger.debug(f"No table data extracted from Table Transformer bbox on page {page_idx}")
                            continue
                        
                        # Path 2: Shrink for preview
                        page_img = cv2.imread(page_img_path)
                        bbox_shrunk = [x0, y0, x1, y1]
                        
                        if page_img is not None:
                            try:
                                bbox_shrunk = self._shrink_table_bbox([x0, y0, x1, y1], page_img, text_lines)
                                
                                # Validate shrunk bbox
                                original_area = (x1 - x0) * (y1 - y0)
                                shrunk_area = (bbox_shrunk[2] - bbox_shrunk[0]) * (bbox_shrunk[3] - bbox_shrunk[1])
                                shrink_ratio = shrunk_area / original_area if original_area > 0 else 0
                                
                                if shrink_ratio < 0.3:
                                    logger.warning(f"Table Transformer: Shrunk bbox too much (ratio={shrink_ratio:.1%}), likely false positive, skipping")
                                    continue
                                
                                logger.info(f"Shrunk Table Transformer bbox (ratio={shrink_ratio:.1%})")
                            except Exception as e:
                                logger.warning(f"Failed to shrink bbox: {e}, using original")
                        
                        # Create with dual path
                        table_item = self._create_table_item_dual_path(
                            table_data, page_idx, bbox_shrunk, bbox_expanded,
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
                text_lines = ingest_data.get("page_text_lines", [[]])[page_idx] if page_idx < len(ingest_data.get("page_text_lines", [])) else []
                
                for t_obj in best_tables:
                    bbox_pdf = list(t_obj.bbox)
                    bbox_rendered = [c * zoom for c in bbox_pdf]
                    kind, dist = self._nearest_caption_kind(bbox_rendered, caption_candidates, prefer_kind="table")
                    if kind == "figure" and dist < config.CAPTION_SEARCH_WINDOW * 0.7:
                        continue
                    
                    # DUAL PATH APPROACH:
                    # Path 1: Use original bbox for data extraction (pdfplumber already found it)
                    table_data = t_obj.extract()
                    
                    if not table_data or not any(any(cell for cell in row) for row in table_data):
                        continue
                    
                    # Path 2: Shrink bbox for preview
                    bbox_shrunk = bbox_rendered
                    if page_img is not None:
                        try:
                            bbox_shrunk = self._shrink_table_bbox(bbox_rendered, page_img, text_lines)
                            
                            # Validate shrunk bbox
                            original_area = (bbox_rendered[2] - bbox_rendered[0]) * (bbox_rendered[3] - bbox_rendered[1])
                            shrunk_area = (bbox_shrunk[2] - bbox_shrunk[0]) * (bbox_shrunk[3] - bbox_shrunk[1])
                            shrink_ratio = shrunk_area / original_area if original_area > 0 else 0
                            
                            if shrink_ratio < 0.3:
                                logger.warning(f"pdfplumber: Shrunk bbox too much (ratio={shrink_ratio:.1%}), likely false positive, skipping")
                                continue
                            
                            logger.info(f"Shrunk pdfplumber bbox (ratio={shrink_ratio:.1%})")
                        except Exception as e:
                            logger.warning(f"Failed to shrink bbox: {e}, using original")
                    
                    # Create with dual path
                    table_item = self._create_table_item_dual_path(
                        table_data, page_idx, bbox_shrunk, bbox_rendered,
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
                
                text_lines = ingest_data.get("page_text_lines", [[]])[page_idx] if page_idx < len(ingest_data.get("page_text_lines", [])) else []
                caption_candidates = self._build_caption_candidates(text_lines)

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
                        kind, dist = self._nearest_caption_kind(bbox_rendered, caption_candidates, prefer_kind="table")
                        if kind == "figure" and dist < config.CAPTION_SEARCH_WINDOW * 0.7:
                            continue
                        # DUAL PATH APPROACH:
                        # Path 1: Use detected bbox for data extraction
                        bbox_pdf = [c / zoom for c in bbox_rendered]
                        table_data, strategy = self._extract_table_data_multi(page, bbox_pdf)
                        
                        if not table_data:
                            continue
                        
                        # Path 2: Shrink for preview
                        bbox_shrunk = bbox_rendered
                        if page_img is not None:
                            try:
                                bbox_shrunk = self._shrink_table_bbox(bbox_rendered, page_img, text_lines)
                                
                                # Validate shrunk bbox
                                original_area = (bbox_rendered[2] - bbox_rendered[0]) * (bbox_rendered[3] - bbox_rendered[1])
                                shrunk_area = (bbox_shrunk[2] - bbox_shrunk[0]) * (bbox_shrunk[3] - bbox_shrunk[1])
                                shrink_ratio = shrunk_area / original_area if original_area > 0 else 0
                                
                                if shrink_ratio < 0.3:
                                    logger.warning(f"Visual: Shrunk bbox too much (ratio={shrink_ratio:.1%}), likely false positive, skipping")
                                    continue
                                
                                logger.info(f"Shrunk visual detection bbox (ratio={shrink_ratio:.1%})")
                            except Exception as e:
                                logger.warning(f"Failed to shrink bbox: {e}, using original")
                        
                        # Create with dual path
                        table_item = self._create_table_item_dual_path(
                            table_data, page_idx, bbox_shrunk, bbox_rendered,
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

        # Additional pass: detect three-line tables (horizontal lines only)
        if config.TABLE_THREE_LINE_DETECT_ENABLE:
            h_regions = self._detect_three_line_tables(gray)
            regions.extend(h_regions)

        return regions

    def _detect_three_line_tables(self, gray: np.ndarray) -> List[List[float]]:
        """
        Detect three-line (or two-line) tables using horizontal line clustering.
        """
        h, w = gray.shape[:2]
        edges = cv2.Canny(gray, 30, 100)
        min_line_length = int(w * config.TABLE_THREE_LINE_MIN_LINE_LENGTH_RATIO)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50,
            minLineLength=max(30, min_line_length), maxLineGap=10
        )
        if lines is None:
            return []

        # Collect horizontal lines
        h_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) <= 3:  # near-horizontal
                length = abs(x2 - x1)
                if length >= min_line_length:
                    y = int((y1 + y2) / 2)
                    h_lines.append((min(x1, x2), y, max(x1, x2)))

        if not h_lines:
            return []

        # Cluster lines by y
        h_lines.sort(key=lambda l: l[1])
        clusters = []
        current = [h_lines[0]]
        for line in h_lines[1:]:
            if abs(line[1] - current[-1][1]) <= 6:
                current.append(line)
            else:
                clusters.append(current)
                current = [line]
        clusters.append(current)

        # Split clusters into groups by large vertical gaps (multiple tables on one page).
        centers = [int(sum(l[1] for l in cluster) / len(cluster)) for cluster in clusters]
        groups = []
        current = [clusters[0]]
        gap_threshold = max(15, int(h * 0.08))
        for idx in range(1, len(clusters)):
            if centers[idx] - centers[idx - 1] > gap_threshold:
                groups.append(current)
                current = [clusters[idx]]
            else:
                current.append(clusters[idx])
        groups.append(current)

        regions = []
        for group in groups:
            if not (config.TABLE_THREE_LINE_MIN_LINES <= len(group) <= config.TABLE_THREE_LINE_MAX_LINES):
                continue
            ys = [int(sum(l[1] for l in cluster) / len(cluster)) for cluster in group]
            x0 = min(min(l[0] for l in cluster) for cluster in group)
            x1 = max(max(l[2] for l in cluster) for cluster in group)
            y0 = min(ys)
            y1 = max(ys)
            pad = max(10, int((y1 - y0) * 0.25))
            bbox = [
                float(max(0, x0 - pad)),
                float(max(0, y0 - pad)),
                float(min(w, x1 + pad)),
                float(min(h, y1 + pad))
            ]
            regions.append(bbox)

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
    
    def _create_table_item_dual_path(self, table_data: List[List], page_idx: int, 
                                     bbox_preview: List[float], bbox_data: List[float],
                                     ingest_data: dict, output_dir: Path, source: str, strategy: str, score: float) -> Optional[Dict]:
        """
        Create a table item with DUAL PATH approach:
        - bbox_preview: Shrunk bbox for preview image (excludes surrounding text)
        - bbox_data: Expanded bbox used for data extraction (ensures completeness)
        
        This ensures:
        1. CSV data is complete (from expanded bbox)
        2. Preview image is clean (from shrunk bbox)
        """
        try:
            # Create item directory (temporary ID)
            temp_id = f"temp_{page_idx}_{int(bbox_preview[0])}_{int(bbox_preview[1])}"
            item_dir = utils.ensure_dir(output_dir / temp_id)
            csv_path = item_dir / "table.csv"
            preview_path = item_dir / "preview.png"
            
            # Save CSV (from table_data extracted with expanded bbox)
            df = pd.DataFrame(table_data)
            df.to_csv(csv_path, index=False, header=False)
            
            # Create preview (from SHRUNK bbox)
            try:
                page_img_path = ingest_data["page_images"][page_idx]
                page_img = cv2.imread(page_img_path)
                
                if page_img is not None:
                    h, w = page_img.shape[:2]
                    x0, y0, x1, y1 = [int(c) for c in bbox_preview]
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
                "bbox": bbox_preview,  # Use shrunk bbox for display
                "pdf_bbox": [c / zoom for c in bbox_preview],
                "data_bbox": bbox_data,  # Store expanded bbox for reference
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
    
    def _create_table_item(self, table_data: List[List], page_idx: int, bbox_rendered: List[float],
                          ingest_data: dict, output_dir: Path, source: str, strategy: str, score: float) -> Optional[Dict]:
        """Create a table item with all metadata."""
        try:
            # Shrink bbox to avoid including surrounding text
            page_img_path = ingest_data["page_images"][page_idx]
            page_img = cv2.imread(page_img_path)
            
            if page_img is not None:
                bbox_rendered = self._shrink_table_bbox(bbox_rendered, page_img)
            
            return self._create_table_item_no_shrink(table_data, page_idx, bbox_rendered, ingest_data, output_dir, source, strategy, score)
        
        except Exception as e:
            logger.error(f"Error creating table item: {e}")
            return None
    
    def _create_table_item_no_shrink(self, table_data: List[List], page_idx: int, bbox_rendered: List[float],
                          ingest_data: dict, output_dir: Path, source: str, strategy: str, score: float) -> Optional[Dict]:
        """Create a table item WITHOUT shrinking bbox (bbox already shrunk)."""
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
    
    def _refine_bbox_with_text_lines(self, bbox: List[float], text_lines: List[Dict]) -> List[float]:
        """
        Refine bbox using clustered text lines to avoid caption/body text leakage.
        """
        if not text_lines:
            return bbox
        import re
        caption_re = re.compile(r"^\s*(fig\.?|figure|table|tab\.?|scheme|chart|图|表)\b", re.IGNORECASE)
        footnote_re = re.compile(r"^\s*(note|notes|footnote|注|备注)\b", re.IGNORECASE)

        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0
        if width <= 0 or height <= 0:
            return bbox

        # Collect lines inside bbox with sufficient width coverage.
        candidates = []
        for line in text_lines:
            lb = line.get("bbox")
            if not lb:
                continue
            text = line.get("text", "").strip()
            if text and (caption_re.match(text) or footnote_re.match(text)):
                continue
            if lb[2] <= x0 or lb[0] >= x1 or lb[3] <= y0 or lb[1] >= y1:
                continue
            line_width = lb[2] - lb[0]
            if line_width / max(1.0, width) < config.TABLE_TEXT_REFINE_MIN_WIDTH_RATIO:
                continue
            candidates.append(lb)

        if len(candidates) < config.TABLE_TEXT_REFINE_MIN_LINES:
            return bbox

        candidates.sort(key=lambda b: b[1])
        heights = [b[3] - b[1] for b in candidates]
        median_h = sorted(heights)[len(heights) // 2]
        gap_threshold = max(10, int(median_h * 1.6))

        # Remove isolated top/bottom lines separated by large gaps.
        def _trim_isolated(lines):
            if len(lines) <= config.TABLE_TEXT_REFINE_MIN_LINES:
                return lines
            trimmed = lines[:]
            if len(trimmed) >= 2:
                top_gap = trimmed[1][1] - trimmed[0][3]
                if top_gap > gap_threshold * 2:
                    trimmed = trimmed[1:]
            if len(trimmed) >= 2:
                bottom_gap = trimmed[-1][1] - trimmed[-2][3]
                if bottom_gap > gap_threshold * 2:
                    trimmed = trimmed[:-1]
            return trimmed

        candidates = _trim_isolated(candidates)
        if len(candidates) < config.TABLE_TEXT_REFINE_MIN_LINES:
            return bbox

        clusters = []
        current = [candidates[0]]
        for line in candidates[1:]:
            if line[1] - current[-1][3] <= gap_threshold:
                current.append(line)
            else:
                clusters.append(current)
                current = [line]
        clusters.append(current)

        # Pick the cluster with most lines.
        best = max(clusters, key=len)
        if len(best) < config.TABLE_TEXT_REFINE_MIN_LINES:
            return bbox

        nx0 = min(l[0] for l in best)
        ny0 = min(l[1] for l in best)
        nx1 = max(l[2] for l in best)
        ny1 = max(l[3] for l in best)

        pad = config.TABLE_TEXT_REFINE_PADDING
        new_bbox = [
            max(x0, nx0 - pad),
            max(y0, ny0 - pad),
            min(x1, nx1 + pad),
            min(y1, ny1 + pad),
        ]

        new_area = (new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1])
        old_area = width * height
        if old_area <= 0:
            return bbox
        if new_area >= old_area * config.TABLE_TEXT_REFINE_MIN_AREA_RATIO:
            return new_bbox
        return bbox

    def _shrink_table_bbox(self, bbox: List[float], page_img: np.ndarray,
                           text_lines: Optional[List[Dict]] = None,
                           shrink_ratio: float = 0.08) -> List[float]:
        """
        Shrink table bounding box to remove surrounding text (BALANCED approach).
        
        Strategy:
        1. Detect table structure (horizontal and vertical lines)
        2. Find the tightest bounding box around the structure
        3. Add reasonable padding
        4. Fallback to projection-based if structure detection fails
        
        Args:
            bbox: Original bounding box [x0, y0, x1, y1]
            page_img: Page image
            shrink_ratio: Ratio to shrink (default 8%, balanced)
            
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
        
        # PASS 1: Structure-based shrinking (detect table lines) - BALANCED
        edges = cv2.Canny(gray, 40, 120)  # Balanced thresholds
        
        # Detect horizontal and vertical lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))  # Balanced kernel
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel, iterations=1)
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel, iterations=1)
        
        # Combine lines to get table structure
        table_structure = cv2.add(h_lines, v_lines)
        
        # Dilate to connect nearby lines - BALANCED
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        table_structure = cv2.dilate(table_structure, dilate_kernel, iterations=2)
        
        # Find the bounding box of the table structure
        coords = cv2.findNonZero(table_structure)
        
        if coords is not None and len(coords) > 80:  # Balanced threshold
            # Get bounding rect of table structure
            struct_x, struct_y, struct_w, struct_h = cv2.boundingRect(coords)
            
            # REASONABLE padding - 15 pixels or 8% of dimension
            padding = max(15, int(min(width, height) * 0.08))
            
            new_y0 = y0 + max(0, struct_y - padding)
            new_y1 = y0 + min(crop.shape[0], struct_y + struct_h + padding)
            new_x0 = x0 + max(0, struct_x - padding)
            new_x1 = x0 + min(crop.shape[1], struct_x + struct_w + padding)
            
            # Ensure valid bbox
            if new_x1 > new_x0 and new_y1 > new_y0:
                # BALANCED: keep at least 40% of original area
                new_area = (new_x1 - new_x0) * (new_y1 - new_y0)
                old_area = width * height
                
            if new_area > old_area * 0.4:
                candidate = [new_x0, new_y0, new_x1, new_y1]
                if text_lines and config.TABLE_TEXT_REFINE_ENABLE:
                    candidate = self._refine_bbox_with_text_lines(candidate, text_lines)
                logger.info(f"Shrunk table bbox (structure-based) from {bbox} to [{candidate[0]:.0f}, {candidate[1]:.0f}, {candidate[2]:.0f}, {candidate[3]:.0f}] (area: {old_area:.0f} -> {new_area:.0f}, {new_area/old_area:.1%})")
                return candidate
        
        # PASS 2: Projection-based shrinking (fallback) - BALANCED
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Horizontal projection (sum along x-axis)
        h_projection = np.sum(binary, axis=1)
        # Vertical projection (sum along y-axis)
        v_projection = np.sum(binary, axis=0)
        
        # BALANCED: 30% threshold
        threshold_h = np.max(h_projection) * 0.30
        threshold_v = np.max(v_projection) * 0.30
        
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
        
        # REASONABLE padding
        padding = max(15, int(min(width, height) * 0.08))
        
        new_y0 = y0 + max(0, top_idx - padding)
        new_y1 = y0 + min(crop.shape[0], bottom_idx + padding)
        new_x0 = x0 + max(0, left_idx - padding)
        new_x1 = x0 + min(crop.shape[1], right_idx + padding)
        
        # Ensure valid bbox
        if new_x1 > new_x0 and new_y1 > new_y0:
            # BALANCED: keep at least 40% of original area
            new_area = (new_x1 - new_x0) * (new_y1 - new_y0)
            old_area = width * height
            
            if new_area > old_area * 0.4:
                candidate = [new_x0, new_y0, new_x1, new_y1]
                if text_lines and config.TABLE_TEXT_REFINE_ENABLE:
                    candidate = self._refine_bbox_with_text_lines(candidate, text_lines)
                logger.info(f"Shrunk table bbox (projection-based) from {bbox} to [{candidate[0]:.0f}, {candidate[1]:.0f}, {candidate[2]:.0f}, {candidate[3]:.0f}] (area: {old_area:.0f} -> {new_area:.0f}, {new_area/old_area:.1%})")
                return candidate
        
        if text_lines and config.TABLE_TEXT_REFINE_ENABLE:
            refined = self._refine_bbox_with_text_lines(bbox, text_lines)
            if refined != bbox:
                return refined
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

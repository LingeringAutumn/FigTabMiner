import pdfplumber
import pandas as pd
import os
from pathlib import Path
from typing import List, Optional
from . import config
from . import utils
from . import layout_detect
from . import bbox_merger

# Import enhanced extractor
try:
    from . import table_extract_v2
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    logger = utils.setup_logging(__name__)
    logger.warning("Enhanced table extractor not available, using basic version")

logger = utils.setup_logging(__name__)


def _extract_table_from_region(page, bbox_pdf: list) -> Optional[List[List[str]]]:
    cropped = page.crop(bbox_pdf)
    table = cropped.extract_table(table_settings=config.TABLE_SETTINGS_LINE)
    if table and any(any(cell for cell in row) for row in table):
        return table
    table = cropped.extract_table(table_settings=config.TABLE_SETTINGS_TEXT)
    if table and any(any(cell for cell in row) for row in table):
        return table
    return None


def extract_tables(pdf_path: str, ingest_data: dict, capabilities: dict) -> list:
    """
    Extract tables from PDF.
    Uses enhanced extractor if available, otherwise falls back to basic version.
    """
    # Try enhanced extractor first
    if ENHANCED_AVAILABLE:
        logger.info("Using enhanced table extractor")
        try:
            return table_extract_v2.extract_tables(pdf_path, ingest_data, capabilities)
        except Exception as e:
            logger.error(f"Enhanced extractor failed: {e}, falling back to basic")
            logger.debug("Traceback:", exc_info=True)
    
    # Fallback to basic extractor
    logger.info("Using basic table extractor")
    return _extract_tables_basic(pdf_path, ingest_data, capabilities)


def _extract_tables_basic(pdf_path: str, ingest_data: dict, capabilities: dict) -> list:
    """
    Basic table extraction (original implementation).
    Strategy:
    1. Try Camelot (Lattice -> Stream) if enabled in capabilities.
    2. Fallback to pdfplumber if Camelot fails or is disabled.
    """
    logger.info("Extracting tables...")
    
    doc_id = ingest_data["doc_id"]
    output_dir = config.OUTPUT_DIR / doc_id / "items"
    
    items = []
    
    # We will process page by page to keep track of page indices easily
    # But camelot processes file-level or page-ranges.
    
    # Note: Mixing camelot and pdfplumber logic requires care.
    # We'll try to get all tables first.
    
    table_counter = 0
    
    use_layout = bool(capabilities.get("layout"))
    
    # Initialize smart merger for tables
    merger_config = {
        'iou_threshold': config.TABLE_MERGE_IOU,
        'overlap_threshold': 0.7,
        'distance_threshold': 30,
        'enable_semantic_merge': False,  # Tables usually don't have subfigures
        'enable_visual_merge': False,    # Tables are usually standalone
        'enable_noise_filter': False,    # Don't filter tables
        'min_area_threshold': config.MIN_TABLE_AREA
    }
    merger = bbox_merger.SmartBBoxMerger(merger_config)

    # --- Strategy A: Layout-guided extraction ---
    if use_layout:
        logger.info("Using layout detection for tables...")
        layout_boxes_by_page = {}
        for page_idx in range(ingest_data["num_pages"]):
            page_img_path = ingest_data["page_images"][page_idx]
            layout_blocks = layout_detect.detect_layout(page_img_path)
            table_boxes = [{'bbox': b["bbox"], 'type': 'table', 'score': b.get('score', 0.5)} 
                          for b in layout_blocks if b["type"] == "table"]
            if not table_boxes:
                continue
            
            # Use smart merger
            table_boxes = merger._merge_by_overlap(table_boxes)
            layout_boxes_by_page[page_idx] = table_boxes

        if layout_boxes_by_page:
            with pdfplumber.open(pdf_path) as pdf:
                for page_idx, table_boxes in layout_boxes_by_page.items():
                    page = pdf.pages[page_idx]
                    page_w, page_h = ingest_data["page_sizes"][page_idx]
                    zoom = ingest_data["zoom"]

                    for box_dict in table_boxes:
                        bbox_rendered = box_dict['bbox']
                        x0, y0, x1, y1 = [int(c) for c in bbox_rendered]
                        if (x1 - x0) < config.MIN_TABLE_DIM or (y1 - y0) < config.MIN_TABLE_DIM:
                            continue
                        if utils.bbox_area([x0, y0, x1, y1]) < config.MIN_TABLE_AREA:
                            continue

                        x0, y0, x1, y1 = utils.expand_bbox(
                            [x0, y0, x1, y1],
                            config.TABLE_CROP_PAD,
                            max_w=page_w,
                            max_h=page_h,
                        )

                        bbox_pdf = [c / zoom for c in [x0, y0, x1, y1]]
                        table_data = _extract_table_from_region(page, bbox_pdf)
                        if not table_data:
                            continue

                        table_counter += 1
                        item_id = f"table_{table_counter:04d}"
                        item_dir = utils.ensure_dir(output_dir / item_id)
                        csv_path = item_dir / "table.csv"
                        preview_path = item_dir / "preview.png"

                        df = pd.DataFrame(table_data)
                        df.to_csv(csv_path, index=False, header=False)

                        try:
                            import cv2
                            page_img_path = ingest_data["page_images"][page_idx]
                            page_img = cv2.imread(page_img_path)
                            if page_img is not None:
                                h, w, _ = page_img.shape
                                bx0, by0, bx1, by1 = [int(c) for c in [x0, y0, x1, y1]]
                                bx0 = max(0, bx0); by0 = max(0, by0)
                                bx1 = min(w, bx1); by1 = min(h, by1)
                                crop = page_img[by0:by1, bx0:bx1]
                                cv2.imwrite(str(preview_path), crop)
                        except Exception as e:
                            logger.warning(f"Failed to create table preview: {e}")

                        item = {
                            "item_id": item_id,
                            "type": "table",
                            "subtype": "table",
                            "page_index": page_idx,
                            "bbox": [x0, y0, x1, y1],
                            "detection_score": box_dict.get('score', 0.5),
                            "detection_source": "layout",
                            "artifacts": {
                                "table_csv": f"items/{item_id}/table.csv",
                                "preview_png": f"items/{item_id}/preview.png"
                            }
                        }
                        items.append(item)

            if items:
                logger.info(f"Extracted {len(items)} tables using layout guidance.")
                return items

    # --- Strategy B: Camelot (Enhanced) ---
    if capabilities.get("camelot", False):
        try:
            import camelot
            logger.info("Attempting Camelot extraction...")
            # Run on all pages
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            if len(tables) == 0:
                logger.info("Camelot lattice found no tables, trying stream...")
                tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
            
            logger.info(f"Camelot found {len(tables)} tables.")
            
            for t in tables:
                table_counter += 1
                item_id = f"table_{table_counter:04d}"
                item_dir = utils.ensure_dir(output_dir / item_id)
                csv_path = item_dir / "table.csv"
                preview_path = item_dir / "preview.png"
                
                # Save CSV
                df = t.df
                df.to_csv(csv_path, index=False, header=False)
                
                # Metadata
                page_idx = t.page - 1 # camelot is 1-based
                # camelot bbox: [x0, y0, x1, y1] (PDF coords, y0 is bottom?)
                # Camelot uses PDF coordinate system where (0,0) is bottom-left usually? 
                # Need to verify. pdfplumber uses top-left.
                # Actually, let's just use what we get or approximate. 
                # Visual bbox extraction from camelot is tricky.
                # We will mark bbox as null if unsure, to avoid bad crop.
                # OR we can try to find the table area using the parsing report.
                
                # For baseline preview, we can try to crop using the reported bbox if possible.
                # Camelot '._bbox' exists? t._bbox is (x0, y0, x1, y1) in PDF coords (bottom-left origin).
                # We need to convert to top-left origin for our crop logic.
                
                bbox_pdf = None
                bbox_rendered = None
                
                if hasattr(t, '_bbox'):
                     # t._bbox is [x0, y_bottom, x1, y_top] usually in PDF
                     # We need page height to flip y
                     _, page_h = ingest_data["page_sizes"][page_idx]
                     # Wait, page_sizes is in pixels (rendered). 
                     # We need PDF page height.
                     # Let's peek pdfplumber to get page height
                     with pdfplumber.open(pdf_path) as pl_pdf:
                         pl_page = pl_pdf.pages[page_idx]
                         pdf_h = pl_page.height
                         
                     x0, y_bottom, x1, y_top = t._bbox
                     # Convert to top-left origin: y_new = H - y_old
                     # Top: H - y_top
                     # Bottom: H - y_bottom
                     new_y0 = pdf_h - y_top
                     new_y1 = pdf_h - y_bottom
                     
                     bbox_pdf = [x0, new_y0, x1, new_y1]
                     
                     zoom = ingest_data["zoom"]
                     bbox_rendered = [c * zoom for c in bbox_pdf]
                
                # Generate preview if we have bbox
                if bbox_rendered:
                    try:
                        import cv2
                        page_img_path = ingest_data["page_images"][page_idx]
                        page_img = cv2.imread(page_img_path)
                        if page_img is not None:
                            h, w, _ = page_img.shape
                            bx0, by0, bx1, by1 = [int(c) for c in bbox_rendered]
                            bx0=max(0,bx0); by0=max(0,by0); bx1=min(w,bx1); by1=min(h,by1)
                            crop = page_img[by0:by1, bx0:bx1]
                            cv2.imwrite(str(preview_path), crop)
                    except Exception as e:
                        logger.warning(f"Failed to create table preview: {e}")

                item = {
                    "item_id": item_id,
                    "type": "table",
                    "subtype": "table",
                    "page_index": page_idx,
                    "bbox": bbox_rendered,
                    "artifacts": {
                        "table_csv": f"items/{item_id}/table.csv",
                        "preview_png": f"items/{item_id}/preview.png" if bbox_rendered and os.path.exists(preview_path) else None
                    }
                }
                items.append(item)
                
            return items # If camelot runs, we use it and return
            
        except Exception as e:
            logger.error(f"Camelot extraction failed: {e}. Falling back to pdfplumber.")
            # Fall through to pdfplumber
            items = []
            table_counter = 0

    # --- Strategy B: pdfplumber (Baseline) ---
    logger.info("Using pdfplumber extraction...")
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            
            # pdfplumber find_tables() gives bboxes
            found_tables = page.find_tables()
            
            for j, t_obj in enumerate(found_tables):
                table_counter += 1
                item_id = f"table_{table_counter:04d}"
                item_dir = utils.ensure_dir(output_dir / item_id)
                csv_path = item_dir / "table.csv"
                preview_path = item_dir / "preview.png"
                
                # Get Data
                table_data = t_obj.extract()
                if not table_data:
                    continue
                    
                df = pd.DataFrame(table_data)
                df.to_csv(csv_path, index=False, header=False)
                
                # BBox
                bbox_pdf = list(t_obj.bbox) # (x0, top, x1, bottom)
                zoom = ingest_data["zoom"]
                bbox_rendered = [c * zoom for c in bbox_pdf]
                
                # Preview
                try:
                    import cv2
                    page_img_path = ingest_data["page_images"][i]
                    page_img = cv2.imread(page_img_path)
                    if page_img is not None:
                        h, w, _ = page_img.shape
                        bx0, by0, bx1, by1 = [int(c) for c in bbox_rendered]
                        bx0=max(0,bx0); by0=max(0,by0); bx1=min(w,bx1); by1=min(h,by1)
                        crop = page_img[by0:by1, bx0:bx1]
                        cv2.imwrite(str(preview_path), crop)
                except Exception as e:
                    logger.warning(f"Failed to create table preview: {e}")
                
                item = {
                    "item_id": item_id,
                    "type": "table",
                    "subtype": "table",
                    "page_index": i,
                    "bbox": bbox_rendered,
                    "artifacts": {
                        "table_csv": f"items/{item_id}/table.csv",
                        "preview_png": f"items/{item_id}/preview.png"
                    }
                }
                items.append(item)
                
    logger.info(f"Extracted {len(items)} tables.")
    return items

import pdfplumber
import pandas as pd
import os
from pathlib import Path
from . import config
from . import utils

logger = utils.setup_logging(__name__)

def extract_tables(pdf_path: str, ingest_data: dict, capabilities: dict) -> list:
    """
    Extract tables from PDF.
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
    
    # --- Strategy A: Camelot (Enhanced) ---
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

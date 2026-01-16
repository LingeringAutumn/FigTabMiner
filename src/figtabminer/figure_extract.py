import fitz
import os
from pathlib import Path
from . import config
from . import utils

logger = utils.setup_logging(__name__)

def extract_figures(ingest_data: dict) -> list:
    """
    Extract figures from PDF using PyMuPDF image blocks.
    Saves preview images for each figure.
    """
    logger.info("Extracting figures...")
    
    doc_id = ingest_data["doc_id"]
    pdf_path = ingest_data["pdf_path"]
    zoom = ingest_data["zoom"]
    output_dir = config.OUTPUT_DIR / doc_id / "items"
    
    items = []
    
    doc = fitz.open(pdf_path)
    
    fig_counter = 0
    
    for page_idx in range(ingest_data["num_pages"]):
        page = doc[page_idx]
        image_list = page.get_images(full=True)
        
        # PyMuPDF get_images returns list of xrefs, but doesn't give bbox directly easily in all cases.
        # Better approach for layout analysis: use get_text("dict") and look for image blocks.
        
        blocks = page.get_text("dict")["blocks"]
        img_blocks = [b for b in blocks if b["type"] == 1] # type 1 is image
        
        for i, block in enumerate(img_blocks):
            fig_counter += 1
            item_id = f"fig_{fig_counter:04d}"
            
            # Block bbox is in PDF unscaled coords
            bbox_pdf = block["bbox"]
            
            # Scale to rendered image coords
            bbox_rendered = [c * zoom for c in bbox_pdf]
            
            # Create item directory
            item_dir = utils.ensure_dir(output_dir / item_id)
            preview_path = item_dir / "preview.png"
            
            # Crop from the rendered page image (better quality consistency)
            # We use the page image generated in ingest
            page_img_path = ingest_data["page_images"][page_idx]
            
            # We can use fitz.Pixmap to crop from the rendered page if we want, 
            # OR we can just extract the image binary from PDF.
            # Requirement says "preview.png". Cropping from rendered page ensures 
            # we see exactly what's on the page (including if it's a vector graphic rendered to png).
            # However, `block["image"]` gives the raw image bytes if it's a raster image.
            # Let's crop from the full page render to be safe and consistent with vector graphics.
            
            try:
                import cv2
                import numpy as np
                
                # Load page image
                page_img = cv2.imread(page_img_path)
                if page_img is None:
                    logger.warning(f"Failed to load page image {page_img_path}")
                    continue
                
                h, w, _ = page_img.shape
                x0, y0, x1, y1 = [int(c) for c in bbox_rendered]
                
                # Clamp
                x0 = max(0, x0); y0 = max(0, y0)
                x1 = min(w, x1); y1 = min(h, y1)
                
                if x1 - x0 < 10 or y1 - y0 < 10:
                    continue # Too small
                
                crop = page_img[y0:y1, x0:x1]
                cv2.imwrite(str(preview_path), crop)
                
                # Create Item
                item = {
                    "item_id": item_id,
                    "type": "figure",
                    "subtype": "unknown", # To be filled by AI
                    "page_index": page_idx,
                    "bbox": bbox_rendered, # Rendered Coords
                    "pdf_bbox": bbox_pdf, # Original Coords
                    "artifacts": {
                        "preview_png": f"items/{item_id}/preview.png"
                    }
                }
                items.append(item)
                
            except Exception as e:
                logger.error(f"Error extracting figure {item_id}: {e}")

    doc.close()
    logger.info(f"Extracted {len(items)} figures.")
    return items

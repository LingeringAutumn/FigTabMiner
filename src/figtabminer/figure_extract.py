import fitz
import os
from pathlib import Path
from typing import Optional
from . import config
from . import utils
from . import layout_detect

logger = utils.setup_logging(__name__)

def _merge_candidate_boxes(bboxes: list) -> list:
    merged = []
    for box in bboxes:
        matched = False
        for i, mbox in enumerate(merged):
            if (
                utils.bbox_iou(box, mbox) >= config.FIGURE_MERGE_IOU
                or utils.bbox_overlap_ratio(box, mbox) >= 0.7
            ):
                merged[i] = utils.merge_bboxes(box, mbox)
                matched = True
                break
        if not matched:
            merged.append(box)
    changed = True
    while changed:
        changed = False
        next_pass = []
        for box in merged:
            merged_into = False
            for i, mbox in enumerate(next_pass):
                if (
                    utils.bbox_iou(box, mbox) >= config.FIGURE_MERGE_IOU
                    or utils.bbox_overlap_ratio(box, mbox) >= 0.7
                ):
                    next_pass[i] = utils.merge_bboxes(box, mbox)
                    merged_into = True
                    changed = True
                    break
            if not merged_into:
                next_pass.append(box)
        merged = next_pass
    return merged


def extract_figures(ingest_data: dict, capabilities: Optional[dict] = None) -> list:
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
    
    use_layout = bool(capabilities and capabilities.get("layout"))

    for page_idx in range(ingest_data["num_pages"]):
        page = doc[page_idx]
        page_img_path = ingest_data["page_images"][page_idx]
        page_w, page_h = ingest_data["page_sizes"][page_idx]
        
        blocks = page.get_text("dict")["blocks"]
        img_blocks = [b for b in blocks if b["type"] == 1] # type 1 is image

        candidate_boxes = []
        if use_layout:
            layout_blocks = layout_detect.detect_layout(page_img_path)
            for lb in layout_blocks:
                if lb["type"] == "figure":
                    candidate_boxes.append(lb["bbox"])
        
        for block in img_blocks:
            bbox_pdf = block["bbox"]
            bbox_rendered = [c * zoom for c in bbox_pdf]
            candidate_boxes.append(bbox_rendered)

        if not candidate_boxes:
            continue

        candidate_boxes = _merge_candidate_boxes(candidate_boxes)

        try:
            import cv2
            import numpy as np
            page_img = cv2.imread(page_img_path)
        except Exception as e:
            logger.warning(f"Failed to load page image for figures: {e}")
            continue

        if page_img is None:
            logger.warning(f"Failed to load page image {page_img_path}")
            continue

        for bbox_rendered in candidate_boxes:
            x0, y0, x1, y1 = [int(c) for c in bbox_rendered]
            if (x1 - x0) < config.MIN_FIGURE_DIM or (y1 - y0) < config.MIN_FIGURE_DIM:
                continue
            if utils.bbox_area([x0, y0, x1, y1]) < config.MIN_FIGURE_AREA:
                continue
            if (
                utils.bbox_area([x0, y0, x1, y1]) / (page_w * page_h)
                > config.MAX_FIGURE_PAGE_RATIO
                and len(ingest_data["page_text_lines"][page_idx]) >= config.MIN_TEXT_LINES_FOR_PAGE_SKIP
            ):
                continue

            x0, y0, x1, y1 = utils.expand_bbox(
                [x0, y0, x1, y1],
                config.FIGURE_CROP_PAD,
                max_w=page_w,
                max_h=page_h,
            )

            crop = page_img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, ink = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
            ink_ratio = float(np.count_nonzero(ink)) / float(ink.size)
            if ink_ratio < config.MIN_FIGURE_INK_RATIO:
                continue

            fig_counter += 1
            item_id = f"fig_{fig_counter:04d}"
            
            # Create item directory
            item_dir = utils.ensure_dir(output_dir / item_id)
            preview_path = item_dir / "preview.png"
            
            try:
                cv2.imwrite(str(preview_path), crop)
                
                # Create Item
                item = {
                    "item_id": item_id,
                    "type": "figure",
                    "subtype": "unknown", # To be filled by AI
                    "page_index": page_idx,
                    "bbox": [x0, y0, x1, y1], # Rendered Coords
                    "pdf_bbox": [c / zoom for c in [x0, y0, x1, y1]],
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

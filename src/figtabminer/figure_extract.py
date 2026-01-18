import fitz
import os
from pathlib import Path
from typing import Optional
from . import config
from . import utils
from . import layout_detect
from . import bbox_merger

logger = utils.setup_logging(__name__)


def extract_figures(ingest_data: dict, capabilities: Optional[dict] = None) -> list:
    """
    Extract figures from PDF using PyMuPDF image blocks and layout detection.
    Uses smart merging to handle subfigures and filter noise.
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
    
    # Initialize smart merger
    merger_config = {
        'iou_threshold': config.FIGURE_MERGE_IOU,
        'overlap_threshold': config.BBOX_MERGER_OVERLAP_THRESHOLD,
        'distance_threshold': config.BBOX_MERGER_DISTANCE_THRESHOLD,
        'enable_semantic_merge': config.BBOX_MERGER_ENABLE_SEMANTIC,
        'enable_visual_merge': config.BBOX_MERGER_ENABLE_VISUAL,
        'enable_noise_filter': config.BBOX_MERGER_ENABLE_NOISE_FILTER,
        'arrow_aspect_ratio_min': config.BBOX_MERGER_ARROW_ASPECT_MIN,
        'arrow_aspect_ratio_max': config.BBOX_MERGER_ARROW_ASPECT_MAX,
        'arrow_ink_ratio_max': config.BBOX_MERGER_ARROW_INK_MAX,
        'min_area_threshold': config.MIN_FIGURE_AREA
    }
    merger = bbox_merger.SmartBBoxMerger(merger_config)

    for page_idx in range(ingest_data["num_pages"]):
        page = doc[page_idx]
        page_img_path = ingest_data["page_images"][page_idx]
        page_w, page_h = ingest_data["page_sizes"][page_idx]
        
        blocks = page.get_text("dict")["blocks"]
        img_blocks = [b for b in blocks if b["type"] == 1] # type 1 is image

        candidate_boxes = []
        
        # Collect candidates from layout detection
        if use_layout:
            # Get page text for enhanced filtering
            page_text_lines = ingest_data.get("page_text_lines", [[]])[page_idx] if page_idx < len(ingest_data.get("page_text_lines", [])) else []
            page_text = " ".join([line.get("text", "") for line in page_text_lines]) if page_text_lines else None
            
            layout_blocks = layout_detect.detect_layout(page_img_path, page_text)
            for lb in layout_blocks:
                if lb["type"] == "figure":
                    candidate_boxes.append({
                        'bbox': lb["bbox"],
                        'type': 'figure',
                        'score': lb.get('score', 0.5),
                        'source': 'layout'
                    })
        
        # Collect candidates from image blocks
        for block in img_blocks:
            bbox_pdf = block["bbox"]
            bbox_rendered = [c * zoom for c in bbox_pdf]
            candidate_boxes.append({
                'bbox': bbox_rendered,
                'type': 'figure',
                'score': 0.8,
                'source': 'image_block'
            })

        if not candidate_boxes:
            continue

        # Load page image for smart merging
        try:
            import cv2
            import numpy as np
            page_img = cv2.imread(page_img_path)
        except Exception as e:
            logger.warning(f"Failed to load page image for figures: {e}")
            page_img = None

        # Use smart merger
        if page_img is not None:
            candidate_boxes = merger.merge(candidate_boxes, page_image=page_img, captions=None)
        else:
            # Fallback to simple overlap-based merge
            logger.debug("Using fallback merge (no page image)")
            candidate_boxes = merger._merge_by_overlap(candidate_boxes)
        
        logger.debug(f"Page {page_idx}: {len(candidate_boxes)} figures after merging")

        if page_img is None:
            logger.warning(f"Failed to load page image {page_img_path}")
            continue

        for box_dict in candidate_boxes:
            bbox_rendered = box_dict['bbox']
            x0, y0, x1, y1 = [int(c) for c in bbox_rendered]
            
            # Apply size filters
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

            # Expand bbox slightly
            x0, y0, x1, y1 = utils.expand_bbox(
                [x0, y0, x1, y1],
                config.FIGURE_CROP_PAD,
                max_w=page_w,
                max_h=page_h,
            )

            crop = page_img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            
            # Check ink ratio
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
                    "detection_score": box_dict.get('score', 0.5),
                    "detection_source": box_dict.get('source', 'unknown'),
                    "merged_from": box_dict.get('merged_from', 1),
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

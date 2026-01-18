import fitz
import os
from pathlib import Path
from typing import Optional
from . import config
from . import utils
from . import layout_detect
from . import bbox_merger
from . import content_classifier
from .arxiv_filter import ArxivFilter
from .models import Detection
from .text_false_positive_filter import TextFalsePositiveFilter

logger = utils.setup_logging(__name__)


def _bbox_intersection_area(bbox1: list, bbox2: list) -> float:
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(x1 - x0) * float(y1 - y0)


def _horizontal_overlap_ratio(bbox1: list, bbox2: list) -> float:
    overlap = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
    if overlap <= 0:
        return 0.0
    w1 = bbox1[2] - bbox1[0]
    w2 = bbox2[2] - bbox2[0]
    denom = max(1.0, min(w1, w2))
    return overlap / denom


def _vertical_overlap_ratio(bbox1: list, bbox2: list) -> float:
    overlap = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
    if overlap <= 0:
        return 0.0
    h1 = bbox1[3] - bbox1[1]
    h2 = bbox2[3] - bbox2[1]
    denom = max(1.0, min(h1, h2))
    return overlap / denom


def _text_line_coverage_ratio(bbox: list, text_lines: list) -> float:
    area = utils.bbox_area(bbox)
    if area <= 0:
        return 0.0
    covered = 0.0
    for line in text_lines:
        covered += _bbox_intersection_area(bbox, line["bbox"])
    return min(1.0, covered / area)


def _has_text_barrier(bbox_a: list, bbox_b: list, text_lines: list) -> bool:
    if not text_lines:
        return False
    # Check vertical separation (stacked boxes).
    if bbox_a[3] <= bbox_b[1] or bbox_b[3] <= bbox_a[1]:
        upper, lower = (bbox_a, bbox_b) if bbox_a[1] < bbox_b[1] else (bbox_b, bbox_a)
        gap_top = upper[3]
        gap_bottom = lower[1]
        if gap_bottom <= gap_top + 2:
            return False
        gap_left = max(upper[0], lower[0])
        gap_right = min(upper[2], lower[2])
        if gap_right <= gap_left:
            return False
        gap_bbox = [gap_left, gap_top, gap_right, gap_bottom]
        for line in text_lines:
            if _bbox_intersection_area(gap_bbox, line["bbox"]) <= 0:
                continue
            if (
                _horizontal_overlap_ratio(line["bbox"], gap_bbox) >= config.FIGURE_TEXT_BARRIER_MIN_OVERLAP and
                _horizontal_overlap_ratio(line["bbox"], upper) >= config.FIGURE_TEXT_BARRIER_MIN_OVERLAP and
                _horizontal_overlap_ratio(line["bbox"], lower) >= config.FIGURE_TEXT_BARRIER_MIN_OVERLAP
            ):
                return True
    # Check horizontal separation (side-by-side boxes).
    if bbox_a[2] <= bbox_b[0] or bbox_b[2] <= bbox_a[0]:
        left, right = (bbox_a, bbox_b) if bbox_a[0] < bbox_b[0] else (bbox_b, bbox_a)
        gap_left = left[2]
        gap_right = right[0]
        if gap_right <= gap_left + 2:
            return False
        gap_top = max(left[1], right[1])
        gap_bottom = min(left[3], right[3])
        if gap_bottom <= gap_top:
            return False
        gap_bbox = [gap_left, gap_top, gap_right, gap_bottom]
        for line in text_lines:
            if _bbox_intersection_area(gap_bbox, line["bbox"]) <= 0:
                continue
            if (
                _vertical_overlap_ratio(line["bbox"], gap_bbox) >= config.FIGURE_TEXT_BARRIER_MIN_OVERLAP and
                _vertical_overlap_ratio(line["bbox"], left) >= config.FIGURE_TEXT_BARRIER_MIN_OVERLAP and
                _vertical_overlap_ratio(line["bbox"], right) >= config.FIGURE_TEXT_BARRIER_MIN_OVERLAP
            ):
                return True
    return False


def _build_caption_candidates(text_lines: list) -> list:
    caption_candidates = []
    if not text_lines:
        return caption_candidates
    import re
    caption_re = re.compile(r"^\s*(fig\.?|figure|table|tab\.?|scheme|chart|图|表)\b", re.IGNORECASE)
    figure_re = re.compile(r"^\s*(fig\.?|figure|scheme|chart|图)\b", re.IGNORECASE)
    table_re = re.compile(r"^\s*(table|tab\.?|表)\b", re.IGNORECASE)
    for line in text_lines:
        text = line.get("text", "").strip()
        if not text:
            continue
        if not caption_re.match(text):
            continue
        kind = None
        if figure_re.match(text):
            kind = "figure"
        elif table_re.match(text):
            kind = "table"
        if kind:
            caption_candidates.append({"bbox": line["bbox"], "kind": kind, "text": text})
    return caption_candidates


def _nearest_caption_kind(bbox: list, candidates: list, prefer_kind: str) -> tuple:
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
        if _horizontal_overlap_ratio(bbox, cand_bbox) < 0.2:
            score += 50
        if score < best_score:
            best_score = score
            best_kind = cand["kind"]
            best_dist = dist
    if best_dist > config.CAPTION_SEARCH_WINDOW:
        return None, float("inf")
    return best_kind, best_dist


def _merge_two_boxes(box1: dict, box2: dict) -> dict:
    merged_bbox = utils.merge_bboxes(box1["bbox"], box2["bbox"])
    score = max(box1.get("score", 0.5), box2.get("score", 0.5))
    best = box1 if box1.get("score", 0.5) >= box2.get("score", 0.5) else box2
    merged_from = box1.get("merged_from", 1) + box2.get("merged_from", 1)
    return {
        "bbox": merged_bbox,
        "type": best.get("type", "figure"),
        "score": score,
        "source": best.get("source", "unknown"),
        "merged_from": merged_from,
    }


def _caption_signature(bbox: list, caption_candidates: list) -> str:
    if not caption_candidates:
        return ""
    best_text = ""
    best_score = float("inf")
    for cand in caption_candidates:
        cand_bbox = cand["bbox"]
        if cand_bbox[1] > bbox[3]:
            dist = cand_bbox[1] - bbox[3]
        elif cand_bbox[3] < bbox[1]:
            dist = bbox[1] - cand_bbox[3]
        else:
            dist = 0
        score = dist
        if _horizontal_overlap_ratio(bbox, cand_bbox) < 0.2:
            score += 50
        if score < best_score:
            best_score = score
            best_text = cand["text"]
    if best_score > config.CAPTION_SEARCH_WINDOW:
        return ""
    return best_text


def _merge_by_proximity(bboxes: list, text_lines: list, caption_candidates: list) -> list:
    if len(bboxes) <= 1:
        return bboxes
    merged = bboxes[:]
    changed = True
    signatures = {id(b): _caption_signature(b["bbox"], caption_candidates) for b in merged}
    while changed:
        changed = False
        result = []
        used = set()
        for i, box1 in enumerate(merged):
            if i in used:
                continue
            current = box1
            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                box2 = merged[j]
                sig1 = signatures.get(id(current), "")
                sig2 = signatures.get(id(box2), "")
                if sig1 and sig2 and sig1 != sig2:
                    continue
                # Check alignment and gap.
                x_overlap = _horizontal_overlap_ratio(current["bbox"], box2["bbox"])
                y_overlap = _vertical_overlap_ratio(current["bbox"], box2["bbox"])
                # Vertical merge (stacked).
                v_gap = max(0, max(box2["bbox"][1] - current["bbox"][3], current["bbox"][1] - box2["bbox"][3]))
                # Horizontal merge (side-by-side).
                h_gap = max(0, max(box2["bbox"][0] - current["bbox"][2], current["bbox"][0] - box2["bbox"][2]))
                should_merge = False
                if x_overlap >= config.FIGURE_PROXIMITY_ALIGNMENT and v_gap <= config.FIGURE_PROXIMITY_MERGE_GAP:
                    should_merge = True
                if y_overlap >= config.FIGURE_PROXIMITY_ALIGNMENT and h_gap <= config.FIGURE_PROXIMITY_MERGE_GAP:
                    should_merge = True
                if should_merge and not _has_text_barrier(current["bbox"], box2["bbox"], text_lines):
                    current = _merge_two_boxes(current, box2)
                    signatures[id(current)] = sig1 or sig2
                    used.add(j)
                    changed = True
            result.append(current)
            used.add(i)
        merged = result
    return merged


def _split_bbox_by_text_band(bbox: dict, text_lines: list, page_img, caption_candidates: list) -> list:
    if not text_lines:
        return [bbox]
    x0, y0, x1, y1 = bbox["bbox"]
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        return [bbox]
    band_lines = []
    for line in text_lines:
        lb = line["bbox"]
        if _bbox_intersection_area([x0, y0, x1, y1], lb) <= 0:
            continue
        line_width = lb[2] - lb[0]
        if line_width / max(1.0, width) < config.FIGURE_TEXT_LINE_MIN_WIDTH_RATIO:
            continue
        band_lines.append(lb)
    if len(band_lines) < config.FIGURE_TEXT_BAND_MIN_LINES:
        # Allow split if a caption line sits inside the bbox.
        for cand in caption_candidates or []:
            if _bbox_intersection_area([x0, y0, x1, y1], cand["bbox"]) > 0:
                band_lines = [cand["bbox"]]
                break
        if len(band_lines) < 1:
            return [bbox]
    band_lines.sort(key=lambda b: b[1])
    # Cluster lines by vertical gap.
    heights = [b[3] - b[1] for b in band_lines]
    median_h = sorted(heights)[len(heights) // 2]
    gap_threshold = max(12, int(median_h * 1.8))
    clusters = []
    current = [band_lines[0]]
    for line in band_lines[1:]:
        if line[1] - current[-1][3] <= gap_threshold:
            current.append(line)
        else:
            clusters.append(current)
            current = [line]
    clusters.append(current)
    # Pick the best middle cluster.
    best = None
    best_score = 0
    for cluster in clusters:
        cy0 = min(l[1] for l in cluster)
        cy1 = max(l[3] for l in cluster)
        center = (cy0 + cy1) / 2
        rel_center = (center - y0) / max(1.0, height)
        if not (config.FIGURE_TEXT_BAND_CENTER_MIN <= rel_center <= config.FIGURE_TEXT_BAND_CENTER_MAX):
            continue
        band_height = cy1 - cy0
        if band_height > height * config.FIGURE_TEXT_BAND_MAX_HEIGHT_RATIO:
            continue
        score = len(cluster) * band_height
        if score > best_score:
            best_score = score
            best = (cy0, cy1)
    if best is None:
        return [bbox]
    band_y0, band_y1 = best
    pad = 4
    top_bbox = [x0, y0, x1, max(y0, band_y0 - pad)]
    bottom_bbox = [x0, min(y1, band_y1 + pad), x1, y1]
    # Validate split regions by ink ratio.
    def _ink_ratio(region):
        import cv2
        import numpy as np
        rx0, ry0, rx1, ry1 = [int(c) for c in region]
        h, w = page_img.shape[:2]
        rx0, ry0 = max(0, rx0), max(0, ry0)
        rx1, ry1 = min(w, rx1), min(h, ry1)
        if rx1 <= rx0 or ry1 <= ry0:
            return 0.0
        crop = page_img[ry0:ry1, rx0:rx1]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, ink = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
        return float(np.count_nonzero(ink)) / float(ink.size)
    top_ok = utils.bbox_area(top_bbox) >= config.MIN_FIGURE_AREA and _ink_ratio(top_bbox) >= config.MIN_FIGURE_INK_RATIO
    bottom_ok = utils.bbox_area(bottom_bbox) >= config.MIN_FIGURE_AREA and _ink_ratio(bottom_bbox) >= config.MIN_FIGURE_INK_RATIO
    if top_ok and bottom_ok:
        top = bbox.copy()
        bottom = bbox.copy()
        top["bbox"] = top_bbox
        bottom["bbox"] = bottom_bbox
        top["merged_from"] = bbox.get("merged_from", 1)
        bottom["merged_from"] = bbox.get("merged_from", 1)
        return [top, bottom]
    return [bbox]


def _is_likely_text_region(bbox: dict, text_lines: list) -> bool:
    if not text_lines:
        return False
    coverage = _text_line_coverage_ratio(bbox["bbox"], text_lines)
    if coverage < config.FIGURE_TEXT_COVERAGE_THRESHOLD:
        return False
    wide_lines = 0
    bbox_w = bbox["bbox"][2] - bbox["bbox"][0]
    for line in text_lines:
        if _bbox_intersection_area(bbox["bbox"], line["bbox"]) <= 0:
            continue
        line_w = line["bbox"][2] - line["bbox"][0]
        if line_w / max(1.0, bbox_w) >= config.FIGURE_TEXT_LINE_MIN_WIDTH_RATIO:
            wide_lines += 1
    return wide_lines >= config.FIGURE_TEXT_BAND_MIN_LINES


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
    text_filter = TextFalsePositiveFilter(
        table_confidence_threshold=config.TEXT_FILTER_CONFIDENCE_THRESHOLD,
        enable_transformer_verification=config.TEXT_FILTER_ENABLE_TRANSFORMER,
        text_density_threshold=config.TEXT_FILTER_TEXT_DENSITY_THRESHOLD,
        min_table_structure_score=config.TEXT_FILTER_MIN_STRUCTURE_SCORE,
        enable_position_heuristics=config.TEXT_FILTER_ENABLE_POSITION_HEURISTICS,
        enable_ocr_pattern_matching=config.TEXT_FILTER_ENABLE_OCR_PATTERN,
        enable_text_line_detection=config.TEXT_FILTER_ENABLE_TEXT_LINE_DETECTION,
    )
    
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
        page_text_lines = ingest_data.get("page_text_lines", [[]])[page_idx] if page_idx < len(ingest_data.get("page_text_lines", [])) else []
        
        blocks = page.get_text("dict")["blocks"]
        img_blocks = [b for b in blocks if b["type"] == 1] # type 1 is image

        candidate_boxes = []
        caption_candidates = _build_caption_candidates(page_text_lines)
        
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
                elif lb["type"] == "table":
                    kind, dist = _nearest_caption_kind(lb["bbox"], caption_candidates, prefer_kind="figure")
                    if kind == "figure" and dist < config.CAPTION_SEARCH_WINDOW * 0.7:
                        candidate_boxes.append({
                            'bbox': lb["bbox"],
                            'type': 'figure',
                            'score': lb.get('score', 0.4),
                            'source': 'layout_reclassified'
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
            candidate_boxes = merger.split_complex_figures(candidate_boxes, page_img)
            candidate_boxes = _merge_by_proximity(candidate_boxes, page_text_lines, caption_candidates)
            if config.FIGURE_SPLIT_TEXT_BAND_ENABLE:
                split_boxes = []
                for box in candidate_boxes:
                    split_boxes.extend(_split_bbox_by_text_band(box, page_text_lines, page_img, caption_candidates))
                candidate_boxes = split_boxes
            if config.FIGURE_REFINE_BBOX_ENABLE:
                candidate_boxes = merger.refine_boundaries(candidate_boxes, page_img)
        else:
            # Fallback to simple overlap-based merge
            logger.debug("Using fallback merge (no page image)")
            candidate_boxes = merger._merge_by_overlap(candidate_boxes)
        
        logger.debug(f"Page {page_idx}: {len(candidate_boxes)} figures after merging")

        if page_img is None:
            logger.warning(f"Failed to load page image {page_img_path}")
            continue

        classifier = content_classifier.ContentClassifier()
        arxiv_filter = None
        if page_idx == 0 and config.ARXIV_FILTER_ENABLE_OCR and utils.safe_import("pytesseract"):
            arxiv_filter = ArxivFilter(
                enable_ocr=True,
                position_threshold=config.ARXIV_FILTER_POSITION_THRESHOLD,
                area_threshold=config.ARXIV_FILTER_AREA_THRESHOLD,
                aspect_ratio_range=(config.ARXIV_FILTER_ASPECT_RATIO_MIN, config.ARXIV_FILTER_ASPECT_RATIO_MAX),
                check_left_margin=config.ARXIV_FILTER_CHECK_LEFT_MARGIN,
                left_margin_threshold=config.ARXIV_FILTER_LEFT_MARGIN_THRESHOLD,
                check_rotation=config.ARXIV_FILTER_CHECK_ROTATION
            )

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

            # Filter likely text regions or equations
            if config.FIGURE_TEXT_FILTER_ENABLE:
                if _is_likely_text_region(box_dict, page_text_lines):
                    continue
                if classifier.is_math_equation({"bbox": [x0, y0, x1, y1]}, page_img, page_text_lines):
                    continue
                coverage = _text_line_coverage_ratio([x0, y0, x1, y1], page_text_lines)
                if coverage >= config.FIGURE_TEXT_COVERAGE_THRESHOLD:
                    is_text, _ = text_filter.detect_continuous_text_lines(
                        Detection(
                            bbox=[x0, y0, x1, y1],
                            type="table",
                            score=float(box_dict.get("score", 0.5)),
                            detector=str(box_dict.get("source", "unknown")),
                        ),
                        page_img,
                    )
                    if is_text:
                        continue
                kind, dist = _nearest_caption_kind([x0, y0, x1, y1], caption_candidates, prefer_kind="figure")
                if kind == "table" and dist < config.CAPTION_SEARCH_WINDOW * 0.7:
                    continue
            if arxiv_filter is not None:
                det = Detection(
                    bbox=[x0, y0, x1, y1],
                    type="figure",
                    score=float(box_dict.get("score", 0.5)),
                    detector=str(box_dict.get("source", "unknown"))
                )
                is_candidate, _ = arxiv_filter.is_arxiv_candidate(det, page_img.shape[:2])
                if is_candidate and arxiv_filter.verify_with_ocr(det, page_img):
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

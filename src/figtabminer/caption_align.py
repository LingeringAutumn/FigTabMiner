import re
from . import config
from . import utils

logger = utils.setup_logging(__name__)

def _caption_type_from_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip().lower()
    if t.startswith(("fig", "figure", "scheme", "chart", "图")):
        return "figure"
    if t.startswith(("table", "tab.", "表")):
        return "table"
    return ""


def extract_figure_number(text: str) -> int:
    """
    Extract figure/table number from caption.
    
    Examples:
        "Figure 1. Caption" -> 1
        "Fig. 2a. Caption" -> 2
        "Table 3: Results" -> 3
    """
    # Match patterns like "Figure 1", "Fig. 2", "Table 3"
    match = re.search(r'(?:fig\.?|figure|table|tab\.?)\s*(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return -1


def extract_subfigure_label(text: str) -> str:
    """
    Extract subfigure label from caption.
    
    Examples:
        "Figure 1(a). Caption" -> "a"
        "Fig. 2 (b) Caption" -> "b"
        "Figure 3a. Caption" -> "a"
    """
    # Match patterns like "(a)", "(b)", "a)", "a."
    patterns = [
        r'\(([a-z])\)',      # (a)
        r'\b([a-z])\)',      # a)
        r'(?:fig\.?|figure)\s*\d+([a-z])\b',  # Figure 1a
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
    
    return None


def align_captions(items: list, ingest_data: dict) -> list:
    """
    Align extracted items (figures/tables) with their captions and snippets.
    
    v1.4 improvements:
    - Figure number matching (Figure 1 -> first figure)
    - Subfigure label extraction (a, b, c)
    - Better multi-line caption handling
    - Direction priority (figures: below, tables: above)
    
    Modifies items in-place.
    """
    logger.info("Aligning captions with enhanced matching...")
    
    # Pre-process text lines for each page to find candidate caption lines
    # Candidate line: starts with "Figure X" or "Table Y"
    
    caption_candidates_by_page = {}
    caption_re = re.compile(r"^\s*(fig\.?|figure|table|tab\.?|scheme|chart|图|表)\b", re.IGNORECASE)
    figure_re = re.compile(r"^\s*(fig\.?|figure|scheme|chart|图)\b", re.IGNORECASE)
    table_re = re.compile(r"^\s*(table|tab\.?|表)\b", re.IGNORECASE)
    
    for page_idx, lines in enumerate(ingest_data["page_text_lines"]):
        candidates = []
        for line_obj in lines:
            text = line_obj["text"].strip()
            if caption_re.match(text):
                # Extract figure/table number
                fig_num = extract_figure_number(text)
                subfig_label = extract_subfigure_label(text)
                
                line_obj["figure_number"] = fig_num
                line_obj["subfigure_label"] = subfig_label
                candidates.append(line_obj)
        
        caption_candidates_by_page[page_idx] = candidates
    
    # Group items by type and page for number matching
    figures_by_page = {}
    tables_by_page = {}
    
    for item in items:
        page_idx = item["page_index"]
        if item["type"] == "figure":
            if page_idx not in figures_by_page:
                figures_by_page[page_idx] = []
            figures_by_page[page_idx].append(item)
        elif item["type"] == "table":
            if page_idx not in tables_by_page:
                tables_by_page[page_idx] = []
            tables_by_page[page_idx].append(item)
    
    # Sort items by position (top to bottom, left to right)
    for page_idx in figures_by_page:
        figures_by_page[page_idx].sort(key=lambda x: (x["bbox"][1], x["bbox"][0]) if x["bbox"] else (0, 0))
    for page_idx in tables_by_page:
        tables_by_page[page_idx].sort(key=lambda x: (x["bbox"][1], x["bbox"][0]) if x["bbox"] else (0, 0))

    # Align
    for item in items:
        page_idx = item["page_index"]
        item_bbox = item["bbox"] # Rendered coords
        
        if not item_bbox:
            # If table has no bbox, we can't align spatially easily.
            # Maybe just take the first table caption on the page?
            # For now, skip spatial alignment if no bbox.
            candidates = caption_candidates_by_page.get(page_idx, [])
            if candidates:
                best_cand = candidates[0]
                item["caption"] = best_cand["text"]
                item["caption_bbox"] = best_cand["bbox"]
                item["evidence_snippet"] = best_cand["text"]
                item["subfigure_label"] = best_cand.get("subfigure_label")
            continue
            
        candidates = caption_candidates_by_page.get(page_idx, [])
        if not candidates:
            item["caption"] = "No caption found."
            item["evidence_snippet"] = ""
            item["subfigure_label"] = None
            continue
        
        # Determine item index on page (for number matching)
        if item["type"] == "figure":
            page_items = figures_by_page.get(page_idx, [])
        else:
            page_items = tables_by_page.get(page_idx, [])
        
        try:
            item_index = page_items.index(item)
        except ValueError:
            item_index = 0
        
        # Find best matching caption
        best_cand = None
        min_score = float("inf")
        
        for cand in candidates:
            cand_bbox = cand["bbox"]
            
            # 1. Spatial distance
            if cand_bbox[1] > item_bbox[3]: # Candidate is below
                spatial_dist = cand_bbox[1] - item_bbox[3]
                direction = "below"
            elif cand_bbox[3] < item_bbox[1]: # Candidate is above
                spatial_dist = item_bbox[1] - cand_bbox[3]
                direction = "above"
            else:
                # Overlap
                spatial_dist = 0
                direction = "overlap"
            
            # 2. Direction penalty
            direction_penalty = 0
            if item["type"] == "figure" and direction == "above":
                direction_penalty = config.CAPTION_DIRECTION_PENALTY
            if item["type"] == "table" and direction == "below":
                direction_penalty = config.CAPTION_DIRECTION_PENALTY
            
            # 3. Type matching bonus
            type_bonus = 0
            if item["type"] == "figure" and figure_re.match(cand["text"]):
                type_bonus = 30
            if item["type"] == "table" and table_re.match(cand["text"]):
                type_bonus = 30
            
            # 4. Number matching bonus (NEW!)
            number_bonus = 0
            fig_num = cand.get("figure_number", -1)
            if fig_num >= 0:
                # Check if figure number matches item index
                # Figure 1 should match first figure (index 0)
                expected_index = fig_num - 1
                if expected_index == item_index:
                    number_bonus = 50  # Strong bonus for number match
                elif abs(expected_index - item_index) <= 1:
                    number_bonus = 20  # Weak bonus for close match
            
            # 5. Calculate final score (lower is better)
            score = spatial_dist + direction_penalty - type_bonus - number_bonus
            
            # Only consider captions within search window
            if spatial_dist < config.CAPTION_SEARCH_WINDOW and score < min_score:
                min_score = score
                best_cand = cand
        
        if best_cand:
            item["caption"] = best_cand["text"]
            item["caption_bbox"] = best_cand["bbox"]
            item["subfigure_label"] = best_cand.get("subfigure_label")
            caption_type = _caption_type_from_text(best_cand["text"])
            if caption_type:
                item["caption_type"] = caption_type
                if config.CAPTION_FORCE_TYPE and item.get("type") != caption_type:
                    item["type"] = caption_type
                    item["type_override_reason"] = "caption_force_type"
            
            # Generate snippet: Get lines surrounding the caption (multi-line support)
            all_lines = ingest_data["page_text_lines"][page_idx]
            try:
                # Find caption line index
                c_idx = -1
                for i, l in enumerate(all_lines):
                    if l["bbox"] == best_cand["bbox"] and l["text"] == best_cand["text"]:
                        c_idx = i
                        break
                
                if c_idx != -1:
                    # Collect multi-line caption
                    caption_lines = [all_lines[c_idx]]
                    caption_bbox = list(all_lines[c_idx]["bbox"])
                    prev_bbox = all_lines[c_idx]["bbox"]
                    
                    # Look for continuation lines
                    for j in range(c_idx + 1, len(all_lines)):
                        line = all_lines[j]
                        
                        # Stop if we hit another caption
                        if caption_re.match(line["text"]) and j != c_idx:
                            break
                        
                        # Check if line is continuation (small gap)
                        gap = line["bbox"][1] - prev_bbox[3]
                        if gap <= config.CAPTION_CONTINUATION_GAP:
                            caption_lines.append(line)
                            caption_bbox = utils.merge_bboxes(caption_bbox, line["bbox"])
                            prev_bbox = line["bbox"]
                        else:
                            break

                    item["caption"] = " ".join([l["text"] for l in caption_lines]).strip()
                    item["caption_bbox"] = caption_bbox

                    caption_type = _caption_type_from_text(item["caption"])
                    if caption_type:
                        item["caption_type"] = caption_type
                        if config.CAPTION_FORCE_TYPE and item.get("type") != caption_type:
                            item["type"] = caption_type
                            item["type_override_reason"] = "caption_force_type"

                    # Extract snippet (context around caption)
                    start = max(0, c_idx - 2)
                    end = min(len(all_lines), c_idx + len(caption_lines) + 3)
                    snippet_lines = all_lines[start:end]
                    item["evidence_snippet"] = "\n".join([l["text"] for l in snippet_lines])
                else:
                    item["evidence_snippet"] = best_cand["text"]
            except Exception as e:
                logger.warning(f"Error extracting snippet: {e}")
                item["evidence_snippet"] = best_cand["text"]
        else:
            item["caption"] = "No matching caption found."
            item["evidence_snippet"] = ""
            item["subfigure_label"] = None
    
    # Log statistics
    with_caption = len([i for i in items if i.get("caption") and "No" not in i["caption"]])
    with_subfig = len([i for i in items if i.get("subfigure_label")])
    logger.info(f"Caption alignment: {with_caption}/{len(items)} items matched, {with_subfig} with subfigure labels")

    return items

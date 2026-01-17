import re
from . import config
from . import utils

logger = utils.setup_logging(__name__)

def align_captions(items: list, ingest_data: dict) -> list:
    """
    Align extracted items (figures/tables) with their captions and snippets.
    Modifies items in-place.
    """
    logger.info("Aligning captions...")
    
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
                candidates.append(line_obj)
        
        caption_candidates_by_page[page_idx] = candidates

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
            continue
            
        candidates = caption_candidates_by_page.get(page_idx, [])
        if not candidates:
            item["caption"] = "No caption found."
            item["evidence_snippet"] = ""
            continue
            
        # Find nearest candidate below the item (for Figures usually) or above (for Tables usually)
        # But simple distance is often enough if we restrict search window.
        # Let's compute distance: 
        #   If candidate is below item: dist = candidate_y0 - item_y1
        #   If candidate is above item: dist = item_y0 - candidate_y1
        
        best_cand = None
        min_dist = float("inf")
        
        for cand in candidates:
            cand_bbox = cand["bbox"]
            
            # Check vertical overlap or proximity
            # Simple metric: center distance or gap distance
            
            # Gap distance
            if cand_bbox[1] > item_bbox[3]: # Candidate is below
                dist = cand_bbox[1] - item_bbox[3]
                direction = "below"
            elif cand_bbox[3] < item_bbox[1]: # Candidate is above
                dist = item_bbox[1] - cand_bbox[3]
                direction = "above"
            else:
                # Overlap?
                dist = 0
                direction = "overlap"
            penalty = 0
            if item["type"] == "figure" and direction == "above":
                penalty = config.CAPTION_DIRECTION_PENALTY
            if item["type"] == "table" and direction == "below":
                penalty = config.CAPTION_DIRECTION_PENALTY

            bonus = 0
            if item["type"] == "figure" and figure_re.match(cand["text"]):
                bonus = 30
            if item["type"] == "table" and table_re.match(cand["text"]):
                bonus = 30

            score = dist + penalty - bonus
            if dist < config.CAPTION_SEARCH_WINDOW and score < min_dist:
                min_dist = score
                best_cand = cand
        
        if best_cand:
            item["caption"] = best_cand["text"]
            item["caption_bbox"] = best_cand["bbox"]
            
            # Generate snippet: Get lines surrounding the caption
            # We iterate through all lines on page to find index of best_cand
            all_lines = ingest_data["page_text_lines"][page_idx]
            try:
                # Identify index by reference equality or content/bbox match
                # JSON serialization in ingest breaks ref equality if we reloaded.
                # Let's match by bbox center
                
                c_idx = -1
                for i, l in enumerate(all_lines):
                    if l["bbox"] == best_cand["bbox"] and l["text"] == best_cand["text"]:
                        c_idx = i
                        break
                
                if c_idx != -1:
                    caption_lines = [all_lines[c_idx]]
                    caption_bbox = list(all_lines[c_idx]["bbox"])
                    prev_bbox = all_lines[c_idx]["bbox"]
                    for j in range(c_idx + 1, len(all_lines)):
                        line = all_lines[j]
                        if caption_re.match(line["text"]) and j != c_idx:
                            break
                        gap = line["bbox"][1] - prev_bbox[3]
                        if gap <= config.CAPTION_CONTINUATION_GAP:
                            caption_lines.append(line)
                            caption_bbox = utils.merge_bboxes(caption_bbox, line["bbox"])
                            prev_bbox = line["bbox"]
                        else:
                            break

                    item["caption"] = " ".join([l["text"] for l in caption_lines]).strip()
                    item["caption_bbox"] = caption_bbox

                    start = max(0, c_idx - 2)
                    end = min(len(all_lines), c_idx + 5)
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

    return items

import re
import os
from . import config
from . import layout_detect
from . import utils

# v1.3: Import enhanced classifiers
try:
    from . import chart_classifier
    CHART_CLASSIFIER_AVAILABLE = True
except ImportError:
    CHART_CLASSIFIER_AVAILABLE = False

try:
    from . import enhanced_chart_classifier
    ENHANCED_CHART_CLASSIFIER_AVAILABLE = True
except ImportError:
    ENHANCED_CHART_CLASSIFIER_AVAILABLE = False

try:
    from . import enhanced_ai_analyzer
    ENHANCED_AI_ANALYZER_AVAILABLE = True
except ImportError:
    ENHANCED_AI_ANALYZER_AVAILABLE = False

try:
    from . import bar_chart_digitizer
    BAR_DIGITIZER_AVAILABLE = True
except ImportError:
    BAR_DIGITIZER_AVAILABLE = False

logger = utils.setup_logging(__name__)


def _analyze_table_structure(preview_path: str) -> dict:
    try:
        import cv2
        import numpy as np
        img = cv2.imread(preview_path)
        if img is None:
            return {}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
        h_count = int(np.count_nonzero(h_lines))
        v_count = int(np.count_nonzero(v_lines))
        is_three_line = h_count > 100 and v_count < 60
        return {
            "horizontal_line_pixels": h_count,
            "vertical_line_pixels": v_count,
            "has_grid": h_count > 100 and v_count > 100,
            "is_three_line_table": is_three_line,
        }
    except Exception:
        return {}

def detect_capabilities() -> dict:
    """Detect available optional dependencies."""
    caps = {
        "ocr": utils.safe_import("easyocr"),
        "camelot": utils.safe_import("camelot"),
        "layout": layout_detect.layout_available(),
    }
    logger.info(f"Capabilities detected: {caps}")
    return caps

def classify_figure_subtype(preview_path: str, caption: str, snippet: str, ocr_text: str = "") -> tuple:
    """
    Classify figure subtype (spectrum, line_plot, microscopy, etc.)
    Returns: (subtype, confidence, keywords, debug_info)
    """
    text_combined = (caption + " " + snippet + " " + ocr_text).lower()
    
    # 1. Keyword matching
    scores = {k: 0.0 for k in config.SUBTYPE_KEYWORDS.keys()}
    matched_kws = []
    
    for subtype, kws in config.SUBTYPE_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in text_combined:
                scores[subtype] += 1.0
                matched_kws.append(kw)
    
    # 2. Image Heuristics (Simple)
    # Load image and check properties
    import cv2
    import numpy as np
    
    img = cv2.imread(preview_path)
    white_ratio = 0.0
    if img is not None:
        # Check if it looks like a microscopy image (high texture, fills frame)
        # vs Line plot (lots of whitespace/background)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate white pixel ratio (assuming white background)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        white_ratio = np.sum(thresh == 255) / thresh.size
        
        if white_ratio < 0.3:
            # Dark or filled image -> likely microscopy
            scores["microscopy"] += 0.5
        elif white_ratio > 0.7:
            # Mostly white -> likely plot
            scores["line_plot"] += 0.5
            scores["spectrum"] += 0.3 # Spectrum is a type of line plot
            
    # Determine winner
    best_subtype = "unknown"
    max_score = 0.0
    
    for st, score in scores.items():
        if score > max_score:
            max_score = score
            best_subtype = st
            
    # Normalize confidence roughly
    confidence = min(1.0, max_score / 3.0)
    
    # If no keywords found but image looks like plot, default to line_plot
    if best_subtype == "unknown" and white_ratio > 0.7:
        best_subtype = "line_plot"
        confidence = 0.3
        
    return best_subtype, confidence, list(set(matched_kws)), {"scores": scores, "white_ratio": white_ratio}

def extract_scientific_conditions(text: str) -> dict:
    """Extract conditions using regex."""
    conditions = []
    
    for cond_type, pattern in config.CONDITION_PATTERNS.items():
        matches = re.finditer(pattern, text)
        for m in matches:
            conditions.append({
                "name": cond_type,
                "value": m.group(1),
                "source": "regex_heuristic",
                "confidence": 0.8
            })
            
    # Materials
    materials = []
    for pattern in config.MATERIAL_PATTERNS:
        # Avoid matching common words, strict filtering needed
        # This is a very basic heuristic
        pass
        
    # Simple keyword search for common materials mentioned in context could be better
    # For now, let's skip complex material regex to avoid noise
    
    return {
        "conditions": conditions,
        "material_candidates": materials,
        "keywords": []
    }

def enrich_items_with_ai(items: list, ingest_data: dict, capabilities: dict) -> list:
    """
    Enrich items with AI analysis (OCR, Classification, Extraction).
    """
    logger.info("Enriching items with AI...")
    
    # Initialize OCR reader if needed
    reader = None
    if capabilities["ocr"]:
        try:
            import easyocr
            # Initialize once
            gpu_flag = False
            if config.OCR_GPU in ("1", "true", "yes", "y", "on"):
                gpu_flag = True
            elif config.OCR_GPU in ("0", "false", "no", "n", "off"):
                gpu_flag = False
            else:
                try:
                    import torch
                    gpu_flag = torch.cuda.is_available()
                except Exception:
                    gpu_flag = False
            logger.info(f"EasyOCR GPU enabled: {gpu_flag} (FIGTABMINER_OCR_GPU={config.OCR_GPU})")
            reader = easyocr.Reader(['en'], gpu=gpu_flag)
        except Exception as e:
            logger.error(f"Failed to init EasyOCR: {e}")
            capabilities["ocr"] = False

    enhanced_classifier = None
    if ENHANCED_CHART_CLASSIFIER_AVAILABLE and config.CHART_CLASSIFICATION_USE_ENHANCED:
        chart_config = {
            'enable_visual_analysis': config.CHART_CLASSIFICATION_VISUAL,
            'visual_weight': config.CHART_CLASSIFICATION_VISUAL_WEIGHT,
            'keyword_weight': config.CHART_CLASSIFICATION_KEYWORD_WEIGHT,
            'context_weight': 0.1
        }
        enhanced_classifier = enhanced_chart_classifier.EnhancedChartClassifier(chart_config)

    enhanced_analyzer = None
    if ENHANCED_AI_ANALYZER_AVAILABLE:
        enhanced_analyzer = enhanced_ai_analyzer.EnhancedAIAnalyzer()

    for item in items:
        ai_data = {
            "subtype": "unknown",
            "subtype_confidence": 0.0,
            "conditions": [],
            "material_candidates": [],
            "keywords": [],
            "method": "heuristic",
            "debug": {}
        }
        
        caption_text = item.get("caption", "")
        snippet_text = item.get("evidence_snippet", "")
        full_text = f"{caption_text} {snippet_text}"
        
        ocr_text = ""
        
        # 1. OCR (Figures only)
        if item["type"] == "figure" and capabilities["ocr"] and reader:
            try:
                preview_path = item["artifacts"]["preview_png"]
                # Resolve full path
                # item["artifacts"]["preview_png"] is relative to outputs/{doc_id}/
                # We need absolute path.
                # Actually figure_extract saved it. We need to reconstruct full path.
                
                doc_id = ingest_data["doc_id"]
                # artifact path is "items/fig_xxxx/preview.png"
                # full path is config.OUTPUT_DIR / doc_id / artifact_path
                full_img_path = config.OUTPUT_DIR / doc_id / item["artifacts"]["preview_png"]
                
                if os.path.exists(full_img_path):
                    results = reader.readtext(str(full_img_path), detail=0)
                    ocr_text = " ".join(results)
                    ai_data["debug"]["ocr_text"] = ocr_text
                    ai_data["method"] = "ocr_heuristic"
            except Exception as e:
                logger.warning(f"OCR failed for item {item['item_id']}: {e}")
        
        # 2. Classification
        if item["type"] == "figure":
            full_img_path = config.OUTPUT_DIR / ingest_data["doc_id"] / item["artifacts"]["preview_png"]
            
            # v1.7: Use enhanced classifier if available
            if enhanced_classifier is not None:
                try:
                    subtype, conf, kws, debug = enhanced_classifier.classify(
                        str(full_img_path),
                        caption_text,
                        snippet_text,
                        ocr_text
                    )
                    
                    ai_data["subtype"] = subtype
                    ai_data["subtype_confidence"] = conf
                    ai_data["keywords"] = kws
                    ai_data["debug"].update(debug)
                    ai_data["method"] = "enhanced_classifier_v1.7"
                    
                except Exception as e:
                    logger.warning(f"Enhanced classifier failed, using fallback: {e}")
                    # Fallback to old method
                    subtype, conf, kws, debug = classify_figure_subtype(
                        str(full_img_path), caption_text, snippet_text, ocr_text
                    )
                    ai_data["subtype"] = subtype
                    ai_data["subtype_confidence"] = conf
                    ai_data["keywords"] = kws
                    ai_data["debug"].update(debug)
            elif CHART_CLASSIFIER_AVAILABLE and config.CHART_CLASSIFICATION_USE_ENHANCED:
                try:
                    chart_config = {
                        'enable_visual_analysis': config.CHART_CLASSIFICATION_VISUAL,
                        'enable_ocr_assist': config.CHART_CLASSIFICATION_OCR,
                        'visual_weight': config.CHART_CLASSIFICATION_VISUAL_WEIGHT,
                        'keyword_weight': config.CHART_CLASSIFICATION_KEYWORD_WEIGHT
                    }

                    subtype, conf, kws, debug = chart_classifier.classify_chart_type(
                        str(full_img_path),
                        caption_text,
                        snippet_text,
                        ocr_text,
                        config=chart_config,
                        fallback_classifier=classify_figure_subtype
                    )

                    ai_data["subtype"] = subtype
                    ai_data["subtype_confidence"] = conf
                    ai_data["keywords"] = kws
                    ai_data["debug"].update(debug)
                    ai_data["method"] = "enhanced_classifier_v1.3"
                except Exception as e:
                    logger.warning(f"Enhanced classifier failed, using fallback: {e}")
                    subtype, conf, kws, debug = classify_figure_subtype(
                        str(full_img_path), caption_text, snippet_text, ocr_text
                    )
                    ai_data["subtype"] = subtype
                    ai_data["subtype_confidence"] = conf
                    ai_data["keywords"] = kws
                    ai_data["debug"].update(debug)
            else:
                # Use old method
                subtype, conf, kws, debug = classify_figure_subtype(
                    str(full_img_path), caption_text, snippet_text, ocr_text
                )
                ai_data["subtype"] = subtype
                ai_data["subtype_confidence"] = conf
                ai_data["keywords"] = kws
                ai_data["debug"].update(debug)
            
            # v1.3: Auto-digitize bar charts if enabled
            if (
                ai_data["subtype"] == "bar_chart" and
                BAR_DIGITIZER_AVAILABLE and
                config.BAR_CHART_AUTO_DIGITIZE
            ):
                try:
                    bar_config = {
                        'min_bar_width': config.BAR_CHART_MIN_WIDTH,
                        'min_bar_height': config.BAR_CHART_MIN_HEIGHT,
                        'axis_detection_threshold': config.BAR_CHART_AXIS_THRESHOLD,
                        'enable_ocr': config.BAR_CHART_OCR_LABELS
                    }
                    
                    df = bar_chart_digitizer.digitize_bar_chart(
                        str(full_img_path),
                        orientation='auto',
                        config=bar_config
                    )
                    
                    if df is not None and len(df) > 0:
                        # Save bar data
                        item_dir = config.OUTPUT_DIR / ingest_data["doc_id"] / "items" / item["item_id"]
                        bar_data_path = item_dir / "bar_data.csv"
                        df.to_csv(bar_data_path, index=False)
                        
                        # Add to artifacts
                        item["artifacts"]["bar_data_csv"] = f"items/{item['item_id']}/bar_data.csv"
                        
                        ai_data["bar_extraction"] = {
                            "success": True,
                            "num_bars": len(df),
                            "method": "auto_digitize_v1.3"
                        }
                        
                        logger.info(f"Bar chart data extracted: {len(df)} bars")
                    else:
                        ai_data["bar_extraction"] = {
                            "success": False,
                            "reason": "No bars detected"
                        }
                
                except Exception as e:
                    logger.warning(f"Bar chart digitization failed: {e}")
                    ai_data["bar_extraction"] = {
                        "success": False,
                        "reason": str(e)
                    }
        
        elif item["type"] == "table":
            ai_data["subtype"] = "table"
            ai_data["subtype_confidence"] = 1.0
            preview_path = item["artifacts"].get("preview_png")
            if preview_path:
                full_preview = config.OUTPUT_DIR / ingest_data["doc_id"] / preview_path
                if os.path.exists(full_preview):
                    ai_data["table_structure"] = _analyze_table_structure(str(full_preview))
        
        # 3. Enhanced AI analysis (optional)
        analysis_result = None
        if enhanced_analyzer is not None and item["type"] == "figure":
            try:
                analysis_result = enhanced_analyzer.analyze_robust(
                    str(full_img_path),
                    item["type"],
                    caption_text,
                    snippet_text,
                    ocr_text
                )
                ai_data["debug"]["enhanced_analysis"] = analysis_result.debug
            except Exception as e:
                logger.warning(f"Enhanced analyzer failed: {e}")

        # 4. Condition Extraction
        extraction = extract_scientific_conditions(full_text + " " + ocr_text)
        ai_data["conditions"] = extraction["conditions"]
        if analysis_result is not None:
            if ai_data["subtype"] == "unknown" or ai_data["subtype_confidence"] < 0.4:
                ai_data["subtype"] = analysis_result.subtype
                ai_data["subtype_confidence"] = analysis_result.subtype_confidence
            ai_data["conditions"].extend(analysis_result.conditions)
            ai_data["material_candidates"] = analysis_result.materials
            ai_data["keywords"] = list(set(ai_data["keywords"] + analysis_result.keywords))
        
        # Save to item
        item["ai_annotations"] = ai_data
        
        # Write ai.json
        item_dir = config.OUTPUT_DIR / ingest_data["doc_id"] / "items" / item["item_id"]
        utils.write_json(ai_data, item_dir / "ai.json")
        
        # Add to artifacts
        item["artifacts"]["ai_json"] = f"items/{item['item_id']}/ai.json"

    return items

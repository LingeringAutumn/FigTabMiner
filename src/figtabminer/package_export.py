import datetime
from . import config
from . import utils

logger = utils.setup_logging(__name__)

def export_package(ingest_data: dict, items: list, capabilities: dict) -> str:
    """
    Finalize the data package.
    Writes manifest.json, evidence.json, params.json.
    Returns the absolute path to the output directory.
    """
    doc_id = ingest_data["doc_id"]
    output_dir = config.OUTPUT_DIR / doc_id
    
    # 1. Prepare Manifest
    manifest = {
        "meta": {
            "version": "figtabminer-baseline-v1",
            "doc_id": doc_id,
            "source_pdf": os.path.basename(ingest_data["pdf_path"]),
            "created_at": datetime.datetime.now().isoformat(),
            "note": "baseline with heuristic alignment"
        },
        "capabilities": capabilities,
        "items": []
    }
    
    for item in items:
        # Simplify item for manifest (remove heavy fields if any, keep structure)
        manifest_item = {
            "item_id": item["item_id"],
            "type": item["type"],
            "subtype": item["ai_annotations"].get("subtype", "unknown"),
            "page_index": item["page_index"],
            "bbox": item["bbox"],
            "caption": item.get("caption", ""),
            "evidence_snippet": item.get("evidence_snippet", ""),
            "ai_annotations": item["ai_annotations"],
            "artifacts": item["artifacts"]
        }
        manifest["items"].append(manifest_item)
        
        # 2. Write evidence.json per item
        evidence = {
            "page_index": item["page_index"],
            "bbox": item["bbox"],
            "caption_text": item.get("caption", ""),
            "caption_bbox": item.get("caption_bbox", None),
            "snippet_text": item.get("evidence_snippet", ""),
            "extraction_trace": ["Extracted from PDF", "Aligned Caption", "Enriched with AI"]
        }
        
        item_dir = output_dir / "items" / item["item_id"]
        utils.write_json(evidence, item_dir / "evidence.json")
        
        # 3. Write params.json (Empty placeholder or basic info)
        params = {
            "extraction_config": {
                "zoom": ingest_data["zoom"],
                "engine": "baseline"
            }
        }
        utils.write_json(params, item_dir / "params.json")

    # Write Manifest
    utils.write_json(manifest, output_dir / "manifest.json")
    
    logger.info(f"Package exported to: {output_dir}")
    return str(output_dir)

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from figtabminer import pdf_ingest, figure_extract, table_extract, caption_align, ai_enrich, package_export, utils

def run_pipeline(pdf_path):
    logger = utils.setup_logging("Pipeline")
    logger.info(f"Starting pipeline for {pdf_path}")
    
    try:
        # 1. Ingest
        ingest_data = pdf_ingest.ingest_pdf(pdf_path)
        
        # 2. Detect Capabilities
        caps = ai_enrich.detect_capabilities()
        
        # 3. Extract
        figs = figure_extract.extract_figures(ingest_data, caps)
        tabs = table_extract.extract_tables(pdf_path, ingest_data, caps)
        items = figs + tabs
        
        # 4. Align Evidence
        items = caption_align.align_captions(items, ingest_data)
        
        # 5. Enrich
        items = ai_enrich.enrich_items_with_ai(items, ingest_data, caps)
        
        # 6. Export
        out_dir = package_export.export_package(ingest_data, items, caps)
        
        print(f"SUCCESS: Output generated at {out_dir}")
        return out_dir
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FigTabMiner Pipeline")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    args = parser.parse_args()
    
    run_pipeline(args.pdf)

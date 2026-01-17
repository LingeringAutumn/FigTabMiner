#!/usr/bin/env python3
"""
Test script to verify improved figure and table extraction.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_on_sample_pdf():
    """Test extraction on a sample PDF"""
    from figtabminer import pdf_ingest, figure_extract, table_extract, ai_enrich
    
    # Find a sample PDF (adjust path since we're in tests/)
    sample_dir = Path(__file__).parent.parent / "data" / "samples"
    if not sample_dir.exists():
        print("❌ Sample directory not found")
        return False
    
    pdfs = list(sample_dir.glob("*.pdf"))
    if not pdfs:
        print("❌ No sample PDFs found")
        return False
    
    pdf_path = str(pdfs[0])
    print(f"\n{'='*60}")
    print(f"Testing on: {Path(pdf_path).name}")
    print(f"{'='*60}\n")
    
    # Ingest PDF
    print("Step 1: Ingesting PDF...")
    try:
        ingest_data = pdf_ingest.ingest_pdf(pdf_path)
        print(f"✓ Ingested {ingest_data['num_pages']} pages")
    except Exception as e:
        print(f"❌ Ingest failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Detect capabilities
    print("\nStep 2: Detecting capabilities...")
    capabilities = ai_enrich.detect_capabilities()
    print(f"Capabilities:")
    for key, value in capabilities.items():
        print(f"  {key}: {value}")
    
    # Extract figures
    print("\nStep 3: Extracting figures...")
    try:
        figures = figure_extract.extract_figures(ingest_data, capabilities)
        print(f"✓ Extracted {len(figures)} figures")
        
        # Show details
        for fig in figures[:5]:  # Show first 5
            print(f"  - {fig['item_id']}: page {fig['page_index']}, "
                  f"score={fig.get('detection_score', 0):.2f}, "
                  f"source={fig.get('detection_source', 'unknown')}, "
                  f"merged_from={fig.get('merged_from', 1)}")
        
        if len(figures) > 5:
            print(f"  ... and {len(figures) - 5} more")
            
    except Exception as e:
        print(f"❌ Figure extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Extract tables
    print("\nStep 4: Extracting tables...")
    try:
        tables = table_extract.extract_tables(pdf_path, ingest_data, capabilities)
        print(f"✓ Extracted {len(tables)} tables")
        
        # Show details
        for table in tables[:5]:  # Show first 5
            print(f"  - {table['item_id']}: page {table['page_index']}, "
                  f"score={table.get('detection_score', 0):.2f}")
        
        if len(tables) > 5:
            print(f"  ... and {len(tables) - 5} more")
            
    except Exception as e:
        print(f"❌ Table extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total items extracted: {len(figures) + len(tables)}")
    print(f"  Figures: {len(figures)}")
    print(f"  Tables: {len(tables)}")
    
    # Check for merged items
    merged_figures = [f for f in figures if f.get('merged_from', 1) > 1]
    if merged_figures:
        print(f"\nMerged figures: {len(merged_figures)}")
        for fig in merged_figures:
            print(f"  - {fig['item_id']}: merged from {fig['merged_from']} boxes")
    
    return True

def main():
    """Run tests"""
    print("\n" + "="*60)
    print("FigTabMiner Improved Extraction Test")
    print("="*60)
    
    success = test_on_sample_pdf()
    
    if success:
        print("\n✓ Test completed successfully!")
        return 0
    else:
        print("\n❌ Test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

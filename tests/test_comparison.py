#!/usr/bin/env python3
"""
Comparison test to show improvements in figure/table extraction.
"""

import sys
import logging
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_all_samples():
    """Test extraction on all sample PDFs"""
    from figtabminer import pdf_ingest, figure_extract, table_extract, ai_enrich
    
    # Find sample PDFs
    sample_dir = Path(__file__).parent.parent / "data" / "samples"
    if not sample_dir.exists():
        print("❌ Sample directory not found")
        return False
    
    pdfs = sorted(sample_dir.glob("*.pdf"))
    if not pdfs:
        print("❌ No sample PDFs found")
        return False
    
    print(f"\n{'='*70}")
    print(f"Testing on {len(pdfs)} sample PDFs")
    print(f"{'='*70}\n")
    
    # Detect capabilities once
    capabilities = ai_enrich.detect_capabilities()
    print(f"Capabilities: {', '.join(f'{k}={v}' for k, v in capabilities.items())}\n")
    
    results = []
    
    for pdf_path in pdfs:
        pdf_name = pdf_path.name
        print(f"{'='*70}")
        print(f"Processing: {pdf_name}")
        print(f"{'='*70}")
        
        try:
            # Ingest
            ingest_data = pdf_ingest.ingest_pdf(str(pdf_path))
            num_pages = ingest_data['num_pages']
            print(f"  Pages: {num_pages}")
            
            # Extract figures
            figures = figure_extract.extract_figures(ingest_data, capabilities)
            print(f"  Figures: {len(figures)}")
            
            # Show merged figures
            merged_figures = [f for f in figures if f.get('merged_from', 1) > 1]
            if merged_figures:
                print(f"    Merged: {len(merged_figures)} figures")
                for fig in merged_figures:
                    print(f"      - {fig['item_id']}: {fig['merged_from']} boxes merged")
            
            # Extract tables
            tables = table_extract.extract_tables(str(pdf_path), ingest_data, capabilities)
            print(f"  Tables: {len(tables)}")
            
            # Calculate metrics
            total_items = len(figures) + len(tables)
            items_per_page = total_items / num_pages if num_pages > 0 else 0
            
            result = {
                'pdf': pdf_name,
                'pages': num_pages,
                'figures': len(figures),
                'tables': len(tables),
                'total_items': total_items,
                'items_per_page': items_per_page,
                'merged_figures': len(merged_figures),
                'merge_ratio': len(merged_figures) / len(figures) if figures else 0
            }
            results.append(result)
            
            print(f"  Total items: {total_items} ({items_per_page:.2f} per page)")
            print()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}\n")
    
    if results:
        total_pdfs = len(results)
        total_pages = sum(r['pages'] for r in results)
        total_figures = sum(r['figures'] for r in results)
        total_tables = sum(r['tables'] for r in results)
        total_items = sum(r['total_items'] for r in results)
        total_merged = sum(r['merged_figures'] for r in results)
        
        print(f"PDFs processed: {total_pdfs}")
        print(f"Total pages: {total_pages}")
        print(f"Total figures: {total_figures}")
        print(f"Total tables: {total_tables}")
        print(f"Total items: {total_items}")
        print(f"Average items per page: {total_items / total_pages:.2f}")
        print(f"\nMerging statistics:")
        print(f"  Figures merged: {total_merged}")
        print(f"  Merge rate: {total_merged / total_figures * 100:.1f}%" if total_figures > 0 else "  Merge rate: N/A")
        
        # Per-PDF breakdown
        print(f"\n{'='*70}")
        print("Per-PDF Breakdown")
        print(f"{'='*70}\n")
        print(f"{'PDF':<30} {'Pages':>6} {'Figs':>6} {'Tables':>6} {'Total':>6} {'Merged':>6}")
        print(f"{'-'*70}")
        for r in results:
            print(f"{r['pdf']:<30} {r['pages']:>6} {r['figures']:>6} {r['tables']:>6} "
                  f"{r['total_items']:>6} {r['merged_figures']:>6}")
        
        # Save results
        output_file = Path(__file__).parent.parent / "test_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")
        
        return True
    else:
        print("❌ No results to summarize")
        return False

def main():
    """Run comparison test"""
    print("\n" + "="*70)
    print("FigTabMiner Extraction Comparison Test")
    print("="*70)
    
    success = test_all_samples()
    
    if success:
        print("\n✓ Test completed successfully!")
        return 0
    else:
        print("\n❌ Test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

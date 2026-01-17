#!/usr/bin/env python3
"""
Visualization tool to inspect extraction results.
Creates an HTML report showing all extracted figures and tables.
"""

import sys
import json
from pathlib import Path
import base64

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_html_report(output_dir: Path, report_path: Path):
    """Create an HTML report showing all extracted items."""
    
    # Find all doc_id directories
    doc_dirs = [d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith('_')]
    
    if not doc_dirs:
        print("No extraction results found")
        return
    
    html_parts = ["""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>FigTabMiner Extraction Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
        }
        .document {
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .item {
            border: 1px solid #ddd;
            margin: 15px 0;
            padding: 15px;
            border-radius: 5px;
            background: #fafafa;
        }
        .item-header {
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .item-type {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
        }
        .type-figure {
            background: #e3f2fd;
            color: #1976d2;
        }
        .type-table {
            background: #f3e5f5;
            color: #7b1fa2;
        }
        .item-meta {
            color: #666;
            font-size: 14px;
            margin: 5px 0;
        }
        .badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            margin-left: 5px;
        }
        .badge-merged {
            background: #fff3cd;
            color: #856404;
        }
        .badge-score {
            background: #d4edda;
            color: #155724;
        }
        .preview {
            margin-top: 10px;
            text-align: center;
        }
        .preview img {
            max-width: 100%;
            max-height: 400px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .caption {
            margin-top: 10px;
            padding: 10px;
            background: #e8f5e9;
            border-left: 3px solid #4caf50;
            font-style: italic;
        }
        .stats {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }
        .stat-item {
            background: white;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #1976d2;
        }
        .stat-label {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <h1>📊 FigTabMiner Extraction Results</h1>
"""]
    
    total_figures = 0
    total_tables = 0
    total_merged = 0
    
    for doc_dir in sorted(doc_dirs):
        manifest_path = doc_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        doc_id = manifest['meta']['doc_id']
        source_pdf = Path(manifest['meta']['source_pdf']).name
        items = manifest.get('items', [])
        
        figures = [item for item in items if item['type'] == 'figure']
        tables = [item for item in items if item['type'] == 'table']
        merged_items = [item for item in items if item.get('merged_from', 1) > 1]
        
        total_figures += len(figures)
        total_tables += len(tables)
        total_merged += len(merged_items)
        
        html_parts.append(f"""
    <div class="document">
        <h2>📄 {source_pdf}</h2>
        <div class="stats">
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{len(items)}</div>
                    <div class="stat-label">Total Items</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{len(figures)}</div>
                    <div class="stat-label">Figures</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{len(tables)}</div>
                    <div class="stat-label">Tables</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{len(merged_items)}</div>
                    <div class="stat-label">Merged Items</div>
                </div>
            </div>
        </div>
""")
        
        # Show items
        for item in items:
            item_id = item['item_id']
            item_type = item['type']
            page_idx = item['page_index']
            score = item.get('detection_score', 0)
            merged_from = item.get('merged_from', 1)
            source = item.get('detection_source', 'unknown')
            
            type_class = f"type-{item_type}"
            
            html_parts.append(f"""
        <div class="item">
            <div class="item-header">
                <div>
                    <span class="item-type {type_class}">{item_type.upper()}</span>
                    <strong>{item_id}</strong>
                </div>
                <div>
""")
            
            if merged_from > 1:
                html_parts.append(f'                    <span class="badge badge-merged">Merged from {merged_from}</span>\n')
            
            html_parts.append(f'                    <span class="badge badge-score">Score: {score:.2f}</span>\n')
            html_parts.append(f"""
                </div>
            </div>
            <div class="item-meta">
                📄 Page {page_idx + 1} | 🔍 Source: {source}
            </div>
""")
            
            # Show preview image
            preview_path = doc_dir / item['artifacts'].get('preview_png', '')
            if preview_path.exists():
                try:
                    with open(preview_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                    html_parts.append(f"""
            <div class="preview">
                <img src="data:image/png;base64,{img_data}" alt="{item_id}">
            </div>
""")
                except Exception as e:
                    html_parts.append(f'            <div class="preview">⚠️ Could not load preview: {e}</div>\n')
            
            # Show caption if available
            if item.get('caption'):
                html_parts.append(f"""
            <div class="caption">
                <strong>Caption:</strong> {item['caption']}
            </div>
""")
            
            html_parts.append('        </div>\n')
        
        html_parts.append('    </div>\n')
    
    # Add summary
    html_parts.append(f"""
    <div class="document">
        <h2>📈 Overall Summary</h2>
        <div class="stats">
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{len(doc_dirs)}</div>
                    <div class="stat-label">Documents Processed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{total_figures}</div>
                    <div class="stat-label">Total Figures</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{total_tables}</div>
                    <div class="stat-label">Total Tables</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{total_merged}</div>
                    <div class="stat-label">Merged Items</div>
                </div>
            </div>
        </div>
    </div>
""")
    
    html_parts.append("""
</body>
</html>
""")
    
    # Write HTML file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(''.join(html_parts))
    
    print(f"✓ HTML report created: {report_path}")
    print(f"  Open it in a browser to view results")


def main():
    """Generate visualization report."""
    output_dir = Path("data/outputs")
    report_path = Path("extraction_report.html")
    
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return 1
    
    print("Generating HTML report...")
    create_html_report(output_dir, report_path)
    
    print(f"\n✓ Report generated successfully!")
    print(f"  Open {report_path.absolute()} in your browser")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

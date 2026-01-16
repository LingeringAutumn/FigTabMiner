import fitz  # PyMuPDF
import os
from pathlib import Path
from typing import Dict, Any, List
from . import config
from . import utils

logger = utils.setup_logging(__name__)

def ingest_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Ingest a PDF file: calculate doc_id, render pages, extract text lines.
    
    Returns:
        Dict containing ingest data.
    """
    logger.info(f"Ingesting PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Read PDF bytes for ID
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    
    doc_id = utils.get_doc_id(pdf_bytes)
    output_dir = config.OUTPUT_DIR / doc_id
    pages_dir = utils.ensure_dir(output_dir / "_pages")
    
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    
    page_images = []
    page_text_lines = []
    page_sizes = []
    
    # Zoom matrix for rendering
    mat = fitz.Matrix(config.RENDER_ZOOM, config.RENDER_ZOOM)
    
    for i in range(num_pages):
        page = doc[i]
        
        # 1. Render Page
        pix = page.get_pixmap(matrix=mat)
        img_path = pages_dir / f"page_{i:04d}.png"
        pix.save(str(img_path))
        page_images.append(str(img_path))
        
        # 2. Extract Text Lines with BBox
        # text_dict = page.get_text("dict") # "dict" gives blocks -> lines -> spans
        # We want lines directly for simpler processing or we flatten structure
        blocks = page.get_text("dict")["blocks"]
        lines_data = []
        
        for b in blocks:
            if b["type"] == 0: # text block
                for line in b["lines"]:
                    # line bbox in PDF coords
                    l_bbox = line["bbox"]
                    
                    # combine spans for text
                    text = "".join([span["text"] for span in line["spans"]])
                    
                    # Scale bbox to rendered image coords
                    scaled_bbox = [
                        l_bbox[0] * config.RENDER_ZOOM,
                        l_bbox[1] * config.RENDER_ZOOM,
                        l_bbox[2] * config.RENDER_ZOOM,
                        l_bbox[3] * config.RENDER_ZOOM
                    ]
                    
                    lines_data.append({
                        "text": text,
                        "bbox": scaled_bbox, # Rendered Coords
                        "pdf_bbox": l_bbox   # Original PDF Coords
                    })
        
        page_text_lines.append(lines_data)
        page_sizes.append((pix.width, pix.height))
        
        if i % 5 == 0:
            logger.info(f"Processed page {i+1}/{num_pages}")
            
    doc.close()
    
    ingest_data = {
        "doc_id": doc_id,
        "pdf_path": str(pdf_path),
        "num_pages": num_pages,
        "page_images": page_images,
        "page_text_lines": page_text_lines,
        "page_sizes": page_sizes,
        "zoom": config.RENDER_ZOOM
    }
    
    logger.info(f"Ingest complete. Doc ID: {doc_id}")
    return ingest_data

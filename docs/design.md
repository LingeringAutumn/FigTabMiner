# Design Document

## Goal
To build a robust, demonstrable baseline for extracting structured data (figures and tables) from scientific PDFs, serving as an ingestion module for "AI for Science" databases.

## Architecture
The system follows a linear pipeline with optional enhancement branches:

1.  **Ingest**: PyMuPDF renders pages to images and extracts text with bounding boxes. Coordinates are normalized to the rendered image size.
2.  **Extraction**:
    *   **Figures**: Image block detection from PDF structure.
    *   **Tables**: Priority queue: Camelot (Lattice -> Stream) -> pdfplumber (fallback).
3.  **Alignment**: Spatial heuristics link extracted items to "Figure X" or "Table Y" captions and surrounding text snippets.
4.  **Enrichment (AI)**:
    *   **Capabilities**: Runtime detection of OCR/Camelot libraries.
    *   **Classification**: Hybrid approach using keywords (caption/snippet) and image statistics (white ratio).
    *   **Conditions**: Regex-based extraction of physical parameters (Temperature, Wavelength, etc.).
5.  **Digitization**: Computer Vision (OpenCV) based curve extraction on line plots.

## Capability Boundary & Fallback
- **OCR**: If `easyocr` fails to import or initialize, the pipeline skips text-in-image analysis but continues with metadata heuristics.
- **Tables**: If `camelot` is missing (common due to Ghostscript dependency), the system transparently falls back to `pdfplumber`.

## Validation
- Verified on arXiv papers.
- Validated fallback by running in environments without Ghostscript.

## Future Evolution
- **Deep Learning Detection**: Replace heuristic `figure_extract` with YOLO/LayoutLM.
- **Vector Graphics Parsing**: Parse PDF drawing instructions directly for perfect data recovery.
- **Multi-curve Digitization**: Support legend matching and color separation.

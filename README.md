# FigTabMiner 🧪

An AI for Science demo project for extracting Figures and Tables from PDF research papers.
Generates structured datasets with evidence alignment, ready for downstream AI tasks.

## Features
- **End-to-End Extraction**: From PDF to JSON/CSV/PNG.
- **Evidence Alignment**: Links extracted items to captions and text snippets.
- **AI Enrichment**: Heuristic & OCR-based classification and condition extraction.
- **Table Editing**: Interactive table correction.
- **Plot Digitization**: Semi-automatic extraction of data points from line plots.
- **Dual Mode**:
  - **Baseline**: Runs on any machine with minimal dependencies.
  - **Enhanced**: Auto-enables EasyOCR and Camelot if available.

## Installation

### Baseline (Required)
```bash
pip install -r requirements.txt
```

### Enhanced (Optional)
For OCR and advanced table extraction:
```bash
# Ubuntu/Debian
bash scripts/install_extra_ubuntu.sh
pip install -r requirements-extra.txt

# Windows
# Install Ghostscript manually for Camelot
pip install -r requirements-extra.txt
```

## Usage

### 1. Web UI (Streamlit)
The best way to explore.
```bash
streamlit run src/app_streamlit.py
```
- Upload a PDF.
- Click "Run Extraction".
- View results, edit tables, digitize plots.
- Download ZIP.

### 2. CLI (Batch Processing)
```bash
python scripts/run_pipeline.py --pdf data/samples/sample.pdf
```

## Demo Walkthrough
1. Run `python scripts/download_samples.py` to get a test PDF.
2. Run `streamlit run src/app_streamlit.py`.
3. Upload the downloaded PDF.
4. Click **Run Extraction**.
5. Select a Figure item:
   - Check the detected "subtype" (e.g., line_plot).
   - See extracted "conditions" (e.g., Temperature).
   - Enter axis bounds (e.g., 0, 100, 0, 1) and click **Digitize**.
6. Select a Table item:
   - Edit a cell value and click **Save**.

## Checklist
- [x] End-to-End Pipeline
- [x] Streamlit UI
- [x] Figure Extraction & Digitization
- [x] Table Extraction & Editing
- [x] AI Capabilities (Fallback mechanism)

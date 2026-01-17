# FigTabMiner Manual Test Guide (Ubuntu + GPU)

This guide is a step-by-step checklist you can follow line by line.

## 0) Prerequisites

- You already set up the environment (Python venv + deps installed).
- You have at least one PDF to test.

## 1) Start a clean test run

Run from project root:

```bash
cd /home/lingeringautumn/AI4Science/Code/FigTabMiner
source .venv/bin/activate
rm -rf data/outputs/*
```

## 2) Quick GPU sanity check

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import detectron2; print('detectron2 ok')"
```

Expected:
- `torch.cuda.is_available()` returns `True`
- `detectron2 ok` prints with no error

## 3) Edit config file (no env vars needed)

Open the config file and set the values you want:

```bash
sed -n '1,120p' config/figtabminer.json
```

Recommended values for GPU + layout:
- `"ocr_gpu": "true"`
- `"layout_enable": "true"`
- `"layout_score": 0.6`

After edits, just save the file. The app reads it on startup.

## 4) CLI pipeline test (recommended first)

Pick a PDF file:

```bash
export PDF_PATH="data/samples/2110.14774v1 (1).pdf"
python scripts/run_pipeline.py --pdf "$PDF_PATH"
```

Watch for these log lines:
- `Capabilities detected: {'ocr': True, 'camelot': ..., 'layout': True}`
- `EasyOCR GPU enabled: True`
- `Extracted X figures.`
- `Extracted Y tables.`

Output location is printed at the end:
`data/outputs/{doc_id}/`

## 5) Check output files

Replace `{doc_id}` with the value from the logs:

```bash
ls data/outputs/{doc_id}/
ls data/outputs/{doc_id}/items
```

You should see:
- `manifest.json`
- `items/fig_*/preview.png`
- `items/table_*/table.csv`
- `items/*/ai.json`

## 6) Manual quality inspection checklist

Open a few previews and check:
- Bounding box is tight around the figure/table (not huge or clipped).
- Multi-panel figures are not merged into one giant box (unless truly one panel).
- Tables are detected (not missing).

Quick table sanity check:

```bash
python - <<'PY'
import pandas as pd, glob
paths = glob.glob("data/outputs/{doc_id}/items/table_*/table.csv")
print("tables:", len(paths))
for p in paths[:3]:
    df = pd.read_csv(p, header=None)
    print(p, df.shape)
PY
```

## 7) UI test (Streamlit)

```bash
streamlit run src/app_streamlit.py
```

Steps:
1. Upload a PDF.
2. Click "Run Extraction".
3. Left list: verify item count and types.
4. Right panel: verify preview image matches the selected item.
5. Evidence: caption should match the figure/table nearby.
6. Table: edit a cell, click "Save Table".
7. Figure: input x/y bounds, click "Digitize Curve".

## 8) Compare with baseline (no layout)

Disable layout detection in `config/figtabminer.json`:

```json
"layout_enable": "false"
```

Then re-run:

```bash
rm -rf data/outputs/*
python scripts/run_pipeline.py --pdf "$PDF_PATH"
```

Compare:
- Total figures/tables
- Preview bbox quality
- Table extraction success rate

## 9) Common issues and fixes

- Layout not enabled:
  - Check `Capabilities detected` shows `layout: True`.
  - If not, re-check detectron2 install.
- OCR still on CPU:
  - Ensure `config/figtabminer.json` has `"ocr_gpu": "true"`.
  - Check log: `EasyOCR GPU enabled: True`.
- Table CSV looks empty:
  - Many PDFs store tables as images; OCR is needed for those.

## 10) Cleanup between runs

```bash
rm -rf data/outputs/*
```

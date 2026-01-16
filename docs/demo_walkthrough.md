# Demo Walkthrough Script (10 Minutes)

1.  **Setup (1 min)**
    *   Show terminal.
    *   Run `streamlit run src/app_streamlit.py`.
    *   Open browser to `localhost:8501`.

2.  **Ingestion (2 min)**
    *   Drag & Drop a PDF (e.g., a material science paper).
    *   Point out the "Capabilities" status in the sidebar (showing if OCR/Camelot is active).
    *   Click **Run Extraction**. Show the logs/spinner.

3.  **Review Figures (3 min)**
    *   Click a "Figure" item in the sidebar.
    *   Show the **Preview Image**.
    *   Show the **AI Analysis**: "Look, it detected 'XRD' from the caption and classified it as 'spectrum'."
    *   Show the **Evidence**: "Here is the caption and the text paragraph referring to it."

4.  **Digitization (2 min)**
    *   Find a line plot.
    *   Enter dummy axis bounds (e.g., 0-100).
    *   Click **Digitize**.
    *   Show the extracted curve overlay or chart. "We have recovered the raw data."

5.  **Review Tables (1 min)**
    *   Select a Table item.
    *   Show the editable dataframe.
    *   Change a value and save.

6.  **Export (1 min)**
    *   Click **Download Result Package**.
    *   Open the ZIP and show the JSON structure.

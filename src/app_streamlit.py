import streamlit as st
import sys
import os
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'figtabminer'))
from figtabminer import pdf_ingest, figure_extract, table_extract, caption_align, ai_enrich, package_export, utils, \
    config, plot_digitize

st.set_page_config(page_title="FigTabMiner Demo", layout="wide")


def main():
    st.title("🔬 FigTabMiner: AI for Science Extraction")

    # Sidebar: Config & Capabilities
    st.sidebar.header("Configuration")
    caps = ai_enrich.detect_capabilities()
    st.sidebar.write("Capabilities detected:")
    st.sidebar.checkbox("OCR (EasyOCR)", value=caps["ocr"], disabled=True)
    st.sidebar.checkbox("Table (Camelot)", value=caps["camelot"], disabled=True)
    st.sidebar.checkbox("Layout (Detectron2)", value=caps.get("layout", False), disabled=True)

    # File Upload
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

    if "ingest_data" not in st.session_state:
        st.session_state["ingest_data"] = None
    if "items" not in st.session_state:
        st.session_state["items"] = []
    if "doc_id" not in st.session_state:
        st.session_state["doc_id"] = None

    if uploaded_file:
        # Save temp
        temp_dir = Path("temp_upload")
        temp_dir.mkdir(exist_ok=True)
        pdf_path = temp_dir / uploaded_file.name
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.sidebar.button("Run Extraction"):
            with st.spinner("Running pipeline..."):
                try:
                    # Pipeline Steps
                    ingest_data = pdf_ingest.ingest_pdf(str(pdf_path))
                    st.session_state["ingest_data"] = ingest_data
                    st.session_state["doc_id"] = ingest_data["doc_id"]

                    figs = figure_extract.extract_figures(ingest_data, caps)
                    tabs = table_extract.extract_tables(str(pdf_path), ingest_data, caps)
                    items = figs + tabs

                    items = caption_align.align_captions(items, ingest_data)
                    items = ai_enrich.enrich_items_with_ai(items, ingest_data, caps)

                    st.session_state["items"] = items

                    # Initial Export
                    package_export.export_package(ingest_data, items, caps)

                    st.success(f"Extracted {len(items)} items!")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.exception(e)

    # Main Area
    if st.session_state["items"]:
        items = st.session_state["items"]
        doc_id = st.session_state["doc_id"]

        # Layout
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Items List")
            selected_idx = st.radio(
                "Select Item",
                range(len(items)),
                format_func=lambda
                    i: f"{items[i]['item_id']} ({items[i]['type']}) - {items[i]['ai_annotations'].get('subtype', '')}"
            )

            st.divider()

            # Export Zip
            output_dir = config.OUTPUT_DIR / doc_id
            shutil.make_archive(str(output_dir), 'zip', str(output_dir))
            zip_path = str(output_dir) + ".zip"

            with open(zip_path, "rb") as fp:
                st.download_button(
                    label="Download Result Package (.zip)",
                    data=fp,
                    file_name=f"{doc_id}.zip",
                    mime="application/zip"
                )

            # Manual Add
            st.divider()
            st.subheader("Manual Add Figure")
            with st.form("manual_add_form"):
                m_page = st.number_input("Page Index (0-based)", min_value=0, step=1)
                c1, c2, c3, c4 = st.columns(4)
                m_x0 = c1.number_input("x0", value=0.0)
                m_y0 = c2.number_input("y0", value=0.0)
                m_x1 = c3.number_input("x1", value=100.0)
                m_y1 = c4.number_input("y1", value=100.0)

                if st.form_submit_button("Add Item"):
                    if st.session_state["ingest_data"]:
                        try:
                            # Create new item
                            new_id = f"fig_manual_{len(st.session_state['items']) + 1}"
                            bbox = [m_x0, m_y0, m_x1, m_y1]

                            # Crop Image
                            import cv2
                            ingest = st.session_state["ingest_data"]
                            if m_page < ingest["num_pages"]:
                                page_img_path = ingest["page_images"][m_page]
                                page_img = cv2.imread(page_img_path)
                                if page_img is not None:
                                    h, w, _ = page_img.shape
                                    # Clamp
                                    bx0 = int(max(0, m_x0));
                                    by0 = int(max(0, m_y0))
                                    bx1 = int(min(w, m_x1));
                                    by1 = int(min(h, m_y1))

                                    crop = page_img[by0:by1, bx0:bx1]

                                    item_dir = config.OUTPUT_DIR / st.session_state["doc_id"] / "items" / new_id
                                    item_dir.mkdir(parents=True, exist_ok=True)
                                    preview_path = item_dir / "preview.png"
                                    cv2.imwrite(str(preview_path), crop)

                                    new_item = {
                                        "item_id": new_id,
                                        "type": "figure",
                                        "subtype": "unknown",
                                        "page_index": m_page,
                                        "bbox": [bx0, by0, bx1, by1],
                                        "caption": "Manual Entry",
                                        "evidence_snippet": "",
                                        "artifacts": {
                                            "preview_png": f"items/{new_id}/preview.png"
                                        }
                                    }

                                    # Run AI Enrich
                                    caps = ai_enrich.detect_capabilities()
                                    enriched_items = ai_enrich.enrich_items_with_ai([new_item], ingest, caps)

                                    st.session_state["items"].append(enriched_items[0])
                                    st.success(f"Added {new_id}")
                                    st.rerun()
                                else:
                                    st.error("Could not load page image")
                            else:
                                st.error("Invalid page index")
                        except Exception as e:
                            st.error(f"Failed to add: {e}")
                    else:
                        st.warning("Please process a PDF first.")

        with col2:
            if selected_idx is not None:
                item = items[selected_idx]
                st.header(f"Item: {item['item_id']}")

                # Preview
                preview_rel = item["artifacts"]["preview_png"]
                if preview_rel:
                    full_preview_path = config.OUTPUT_DIR / doc_id / preview_rel
                    if os.path.exists(full_preview_path):
                        st.image(str(full_preview_path), caption="Preview")
                    else:
                        st.warning("Preview image missing")

                # Evidence
                st.subheader("Evidence")
                st.text_area("Caption", item.get("caption", ""), height=100)
                st.text_area("Snippet", item.get("evidence_snippet", ""), height=100)

                # AI Info
                st.subheader("AI Analysis")
                st.json(item["ai_annotations"])

                # Interaction
                if item["type"] == "table":
                    st.subheader("Table Data")
                    csv_rel = item["artifacts"].get("table_csv")
                    if csv_rel:
                        csv_path = config.OUTPUT_DIR / doc_id / csv_rel
                        if os.path.exists(csv_path):
                            df = pd.read_csv(csv_path, header=None)
                            edited_df = st.data_editor(df)
                            if st.button("Save Table"):
                                edited_df.to_csv(csv_path, index=False, header=False)
                                st.success("Saved!")

                elif item["type"] == "figure":
                    st.subheader("Plot Digitization")
                    c1, c2, c3, c4 = st.columns(4)
                    x_min = c1.number_input("X Min", value=0.0)
                    x_max = c2.number_input("X Max", value=100.0)
                    y_min = c3.number_input("Y Min", value=0.0)
                    y_max = c4.number_input("Y Max", value=100.0)

                    if st.button("Digitize Curve"):
                        if preview_rel and os.path.exists(full_preview_path):
                            df_curve = plot_digitize.digitize_line_plot(
                                str(full_preview_path), x_min, x_max, y_min, y_max
                            )
                            st.line_chart(df_curve, x="x", y="y")

                            # Save
                            csv_path = config.OUTPUT_DIR / doc_id / "items" / item["item_id"] / "data.csv"
                            df_curve.to_csv(csv_path, index=False)
                            item["artifacts"]["data_csv"] = f"items/{item['item_id']}/data.csv"
                            st.success(f"Curve saved! {len(df_curve)} points.")


if __name__ == "__main__":
    main()

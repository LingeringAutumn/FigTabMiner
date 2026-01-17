# FigTabMiner 🧪

> **🎉 v1.1 改进版** - 识别准确率显著提升！查看 [改进总结](IMPROVEMENTS_SUMMARY.md) 和 [快速开始](QUICK_START.md)

An AI for Science demo project for extracting Figures and Tables from PDF research papers.
Generates structured datasets with evidence alignment, ready for downstream AI tasks.

## 🆕 最新改进 (v1.1)

- ✅ **模型加载问题修复** - 100% 成功率（之前约 50%）
- ✅ **智能边界框合并** - 70.3% 合并率，显著减少过度分割
- ✅ **箭头过滤** - 基本消除箭头误识别问题
- ✅ **质量评估系统** - 5 维度评分和过滤
- ✅ **详细日志** - 更好的调试和监控
- ✅ **测试套件** - 完整的测试和可视化工具

**快速验证改进**：
```bash
bash run_tests.sh                    # 运行所有测试
python tools/visualize_results.py   # 生成 HTML 报告
```

详细信息请查看：
- 📖 [改进总结](IMPROVEMENTS_SUMMARY.md) - 完整的改进说明
- 🚀 [快速开始](QUICK_START.md) - 如何使用和验证
- 📊 [状态报告](STATUS_REPORT.md) - 项目状态和下一步计划

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

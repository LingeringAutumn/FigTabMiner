# FigTabMiner 🧪

> **🎉 v1.3 最新版** - 精确图表分类 + 柱状图数据提取！查看 [v1.3 改进说明](IMPROVEMENTS_V1.3.md)

An AI for Science demo project for extracting Figures and Tables from PDF research papers.
Generates structured datasets with evidence alignment, ready for downstream AI tasks.

## 🆕 最新改进 (v1.3)

**新增功能**：

1. ✅ **精确图表类型识别** - 9 种图表类型
   - bar_chart（柱状图）、pie_chart（饼图）、line_plot（折线图）
   - scatter_plot（散点图）、heatmap（热力图）、box_plot（箱线图）
   - microscopy（显微镜）、diagram（流程图）
   - 关键词 + 视觉特征双重识别

2. ✅ **柱状图数据自动提取** - 结构化数据输出
   - 自动检测坐标轴和柱子
   - 提取数值数据到 CSV
   - 支持垂直/水平柱状图
   - **预期成功率：60-70%**（简单柱状图）

3. ✅ **保留降级策略** - 增强功能失败时优雅回退
   - 新分类器 → 旧分类器 → unknown
   - 柱状图提取失败仍保存预览图

**快速验证**：
```bash
streamlit run src/app_streamlit.py      # 启动 UI 测试
python tests/test_v1.3_improvements.py  # 运行测试
```

详细信息：
- 📖 [v1.3 改进说明](IMPROVEMENTS_V1.3.md) - 图表分类和数据提取
- 📖 [v1.2 改进说明](IMPROVEMENTS_V1.2.md) - 表格提取优化
- 📖 [v1.1 改进总结](IMPROVEMENTS_SUMMARY.md) - 基础优化
- 🚀 [快速开始](QUICK_START.md) - 使用指南

## 🆕 v1.2 改进 (表格优化)

**针对用户反馈的三大问题**：

1. ✅ **数学公式过滤** - 不再将公式误识别为表格
2. ✅ **合并验证增强** - 减少图表错误合并
3. ✅ **表格数据提取增强** - 成功率提升到 85-90%

## 🆕 v1.1 改进 (基础优化)

- ✅ **模型加载问题修复** - 100% 成功率
- ✅ **智能边界框合并** - 70.3% 合并率
- ✅ **箭头过滤** - 基本消除误识别
- ✅ **质量评估系统** - 5 维度评分

## Features
- **End-to-End Extraction**: From PDF to JSON/CSV/PNG.
- **Evidence Alignment**: Links extracted items to captions and text snippets.
- **AI Enrichment**: Enhanced chart classification and data extraction (v1.3).
- **Table Editing**: Interactive table correction.
- **Bar Chart Digitization**: Automatic data extraction from bar charts (v1.3).
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

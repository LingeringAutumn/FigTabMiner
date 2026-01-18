# FigTabMiner | AI for Science Figure & Table Mining System

<div align="center">

**Next-Generation Scientific Literature Understanding and Data Asset Platform**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AI for Science](https://img.shields.io/badge/AI%20for-Science-green.svg)](https://github.com)

English | [简体中文](./docs/README_CN.md)

</div>

---

## 🎯 Overview

FigTabMiner is an intelligent scientific literature figure and table mining system designed specifically for **AI for Science** scenarios. It automatically identifies, extracts, classifies, and structurally analyzes figures and tables from PDF academic papers, transforming them into high-quality data assets ready for machine learning training, scientific knowledge graph construction, and experimental condition extraction.

In scientific research, critical information such as experimental data, material properties, and reaction conditions often exists in the form of figures and tables within literature. FigTabMiner achieves automated, high-precision understanding of these scientific figures through deep learning model fusion, computer vision algorithms, and natural language processing techniques, providing powerful data infrastructure for AI for Science applications including scientific discovery acceleration, materials design, and drug development.

### Core Value

- 🔬 **Scientific Data Assetization**: Transform scattered figures and tables in literature into structured, searchable, trainable data assets
- 🎯 **High-Precision Recognition**: Multi-model fusion strategy (DocLayout-YOLO + Table Transformer + PubLayNet), F1-Score > 0.90
- 🧠 **Intelligent Understanding**: Automatically identify chart types (spectra, microscopy, bar charts, etc.), extract experimental conditions and material information
- 📊 **Data Digitization**: Automatically convert visualization data from bar charts and line plots into CSV format for secondary analysis
- 🚀 **Production-Ready**: Complete pipeline, configurable design, Web UI, batch processing support

---

## ✨ Key Features

### 1. Multi-Model Fusion Layout Detection Engine

Employs advanced deep learning model combinations to achieve a balance between high recall and high precision:

- **DocLayout-YOLO**: YOLO variant designed specifically for document layouts, with strong understanding of academic paper layouts
- **Table Transformer**: Microsoft's open-source Transformer model dedicated to table detection
- **PubLayNet**: General document layout analysis model based on Detectron2
- **Weighted NMS Fusion**: Intelligently fuses results from multiple detectors to reduce false positives and false negatives

### 2. Caption-Constrained Type Correction Mechanism

Innovatively uses figure/table captions (Figure X / Table X) as hard constraints:

- Automatically aligns figure/table regions with their corresponding captions
- Corrects detector misclassifications based on caption type (e.g., Table misidentified as Figure)
- Supports Chinese and English caption recognition (Figure/Fig./图/表)

### 3. Precise Boundary Cropping and Noise Filtering

Achieves pixel-level precise cropping for complex academic paper layouts:

- **Table Boundary Refinement**: Dual-path cropping algorithm based on text line clustering and structure line detection
- **Three-Line Table Recovery**: Morphological line detection + caption guidance, specifically handling borderless tables
- **Text False Positive Filtering**: Multi-dimensional filter based on text density, line width distribution, and structural features
- **arXiv ID Filtering**: Automatically identifies and filters non-figure elements like arXiv IDs at page margins

### 4. AI-Enhanced Analysis and Knowledge Extraction

Deep understanding of figure content and extraction of scientific knowledge:

- **Chart Type Classification**: Identifies 15+ scientific chart types (spectra, XRD, SEM, TEM, bar charts, line plots, etc.)
- **Experimental Condition Extraction**: Automatically identifies experimental parameters like temperature, pressure, concentration, time
- **Material Information Extraction**: Identifies chemical formulas, material names, crystal structures, etc.
- **Keyword Extraction**: Scientific term extraction based on OCR + context
- **Bar Chart Digitization**: Automatically converts bar charts to numerical data (CSV format)

### 5. Complete Data Asset Output

Each figure/table generates a complete, structured data package supporting downstream AI applications:

```
outputs/{doc_id}/
├── manifest.json              # Global metadata index
├── items/
│   ├── fig_0001/
│   │   ├── preview.png        # High-quality cropped image
│   │   ├── ai.json            # AI analysis results
│   │   ├── evidence.json      # Provenance information
│   │   └── bar_data.csv       # Digitized data (if applicable)
│   └── table_0001/
│       ├── preview.png
│       ├── table.csv          # Structured table data
│       ├── ai.json
│       └── evidence.json
└── package.zip                # Complete package for download
```

**Key Features:**
- Machine-readable JSON metadata for all items
- Complete provenance tracking (page number, coordinates, caption)
- Structured CSV data for tables and digitized charts
- Self-contained directories with all related files
- ZIP packaging for easy sharing and archiving

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         PDF Input                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PDF Parsing & Preprocessing (PyMuPDF + pdfplumber)             │
│  • Render pages to high-resolution images                       │
│  • Extract text with coordinate information                     │
│  • Extract page metadata                                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Multi-Model Layout Detection (Detection Fusion)                │
│  • DocLayout-YOLO (document-specific)                           │
│  • Table Transformer (table-specific)                           │
│  • PubLayNet (general layout)                                   │
│  • Weighted NMS fusion + quality scoring                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Rule-Based Fusion Processing                                   │
│  • Caption alignment and type correction                        │
│  • Text false positive filtering (TextFalsePositiveFilter)      │
│  • arXiv ID filtering (ArxivFilter)                             │
│  • Table recovery (img2table)                                   │
│  • Figure merge/split (BBoxMerger)                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Boundary Refinement & Cropping                                 │
│  • Tables: text line clustering + structure line detection      │
│  • Figures: noise filtering + ink ratio detection               │
│  • Three-line tables: morphological line detection + caption    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  AI Enrichment Analysis                                         │
│  • Chart type classification (EnhancedChartClassifier)          │
│  • OCR text extraction (EasyOCR)                                │
│  • Experimental condition extraction (regex + heuristics)       │
│  • Material information extraction                              │
│  • Bar chart digitization (BarChartDigitizer)                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Structured Export                                              │
│  • JSON metadata (manifest.json, ai.json, evidence.json)       │
│  • CSV data (tables, bar charts)                               │
│  • PNG preview images                                           │
│  • ZIP packaging                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10+
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM minimum (16GB recommended for large documents)
- **Storage**: 5GB free space (for models and cache)
- **GPU** (Optional): CUDA 11.0+ compatible GPU for acceleration (3-8x faster)

### Installation and Deployment

#### Basic Installation

```bash
# Clone repository
git clone https://github.com/LingeringAutumn/FigTabMiner.git
cd FigTabMiner

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: First run will automatically download required models (~2GB total).

#### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install -y ghostscript tesseract-ocr libgl1-mesa-glx
```

**macOS:**
```bash
brew install ghostscript tesseract
```

**Windows:**
- Install [Ghostscript](https://www.ghostscript.com/download/gsdnld.html) and [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)

#### Quick Start

```bash
# Run Web UI
streamlit run src/app_streamlit.py

# Access at http://localhost:8501
```

---

## 📖 Usage

### Method 1: Web UI (Recommended for Beginners)

```bash
streamlit run src/app_streamlit.py
```

Visit `http://localhost:8501` and:
1. Upload a PDF file
2. Configure processing options (optional)
3. Click "Start Processing"
4. Download results as ZIP package

### Method 2: Command-Line Batch Processing

```bash
# Process single PDF
python scripts/run_pipeline.py --pdf data/samples/your_paper.pdf

# Process multiple PDFs
for pdf in data/samples/*.pdf; do
    python scripts/run_pipeline.py --pdf "$pdf"
done

# Output location
ls data/outputs/
```

---

## 📊 Application Scenarios

### 1. Scientific Dataset Construction

Build high-quality scientific figure datasets for machine learning models:

- **Image Classification**: Build scientific chart type classification datasets (SEM, TEM, XRD, spectra, etc.)
- **Object Detection**: Train document layout analysis models
- **Chart Understanding**: Train Chart-to-Text, Chart QA models

### 2. Scientific Knowledge Extraction

Automatically extract structured scientific knowledge from literature:

- **Experimental Condition Mining**: Parameters like temperature, pressure, concentration, time
- **Material Property Database**: Build material-property relationship databases
- **Reaction Condition Library**: Automatic extraction of chemical reaction conditions

### 3. Scientific Literature Retrieval and QA

Build figure-based scientific literature retrieval systems:

- **Figure Retrieval**: Retrieve relevant literature based on chart type and experimental conditions
- **Cross-Modal Retrieval**: Joint text-figure retrieval
- **Scientific QA**: Figure-based scientific question answering systems

### 4. Experimental Data Digitization

Convert visualization data in literature to analyzable numerical data:

- **Bar Chart Digitization**: Automatically extract numerical values from bar charts
- **Curve Digitization**: Extract XRD, spectra, and other curve data
- **Table Structuring**: Convert complex tables to CSV format

### 5. Scientific Trend Analysis

Analyze development trends in scientific research:

- **Chart Type Statistics**: Analyze commonly used characterization methods in different fields
- **Experimental Condition Evolution**: Track historical changes in experimental conditions
- **Material Research Hotspots**: Identify popular research materials

---

## ⚙️ Configuration

Main configuration file: `config/figtabminer.json`

### Key Configuration Items

#### Layout Detection Configuration

```json
{
  "v17_detection": {
    "enable_doclayout": true,
    "enable_table_transformer": true,
    "doclayout_confidence": 0.35,
    "table_transformer_confidence": 0.75,
    "fusion_strategy": "weighted_nms",
    "nms_iou_threshold": 0.5
  }
}
```

#### Caption Correction Configuration

```json
{
  "caption_force_type": true,
  "caption_search_window": 300,
  "caption_direction_penalty": 120
}
```

#### Table Extraction Configuration

```json
{
  "table_text_refine_enable": true,
  "table_text_refine_min_lines": 2,
  "table_text_refine_padding": 8,
  "table_three_line_detect_enable": true,
  "table_enhancer_enable_img2table": true
}
```

---

## 🔧 Core Modules

### 1. PDF Parsing (`pdf_ingest.py`)
- Render pages to high-resolution images using PyMuPDF
- Extract text and coordinate information using pdfplumber
- Generate page metadata (size, DPI, etc.)

### 2. Layout Detection (`layout_detect.py`)
- Multi-model detector management (DocLayout-YOLO, Table Transformer, PubLayNet)
- Detection result fusion (weighted NMS)
- Model caching and fallback strategies

### 3. Figure Extraction (`figure_extract.py`)
- Image block detection and merging
- Noise filtering (ink ratio, size filtering)
- Figure splitting (based on text barriers)

### 4. Table Extraction (`table_extract_v2.py`)
- Multi-strategy table detection (structure lines, text clustering, img2table)
- Three-line table specific detection
- Boundary refinement and shrinking

### 5. Caption Alignment (`caption_align.py`)
- Caption detection (regex matching)
- Spatial distance calculation and alignment
- Type correction (based on caption hard constraints)

### 6. AI Enrichment (`ai_enrich.py`)
- OCR text extraction (EasyOCR)
- Chart type classification (EnhancedChartClassifier)
- Experimental condition extraction (regex + heuristics)
- Bar chart digitization (BarChartDigitizer)

### 7. Quality Assessment (`quality_assess.py`)
- Multi-dimensional quality scoring (detection confidence, content completeness, boundary precision, etc.)
- Anomaly detection (size anomalies, position anomalies)
- Quality report generation

### 8. Data Export (`package_export.py`)
- JSON metadata generation
- CSV data export
- PNG preview image generation
- ZIP packaging

---

## 🛠️ Development & Contribution

### Project Structure

```
figtabminer/
├── src/figtabminer/          # Core code
│   ├── pdf_ingest.py         # PDF parsing
│   ├── layout_detect.py      # Layout detection
│   ├── figure_extract.py     # Figure extraction
│   ├── table_extract_v2.py   # Table extraction
│   ├── caption_align.py      # Caption alignment
│   ├── ai_enrich.py          # AI enrichment
│   ├── quality_assess.py     # Quality assessment
│   ├── package_export.py     # Data export
│   ├── detectors/            # Detector modules
│   │   ├── doclayout_detector.py
│   │   └── table_transformer_detector.py
│   └── models.py             # Data models
├── scripts/                  # Script tools
│   └── run_pipeline.py       # Batch processing script
├── tools/                    # Auxiliary tools
│   ├── evaluate_accuracy.py  # Accuracy evaluation
│   ├── diagnose_accuracy.py  # Diagnostic tool
│   └── visualize_results.py  # Visualization tool
├── config/                   # Configuration files
│   └── figtabminer.json
├── data/                     # Data directory
│   ├── samples/              # Sample data
│   └── outputs/              # Output results
└── docs/                     # Documentation
    └── README_CN.md          # Chinese documentation
```

### Contribution Guidelines

We welcome all forms of contributions!

1. Fork this repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 🚧 Future Development Roadmap

As a rapid prototype developed in 2 days, FigTabMiner demonstrates core capabilities but has room for enhancement. Here are planned improvements:

### 1. Performance Optimization
- **Parallel Processing**: Implement multi-threading for concurrent page processing
- **Async I/O**: Adopt asyncio for non-blocking file operations
- **Batch Inference**: Process multiple images in batches for GPU efficiency
- **Model Quantization**: Reduce model size and inference time

### 2. Database Integration
- **SQL Backend**: PostgreSQL integration for metadata storage and querying
- **Vector Database**: Milvus/Faiss for semantic figure search
- **Caching Layer**: Redis for frequently accessed results
- **Query API**: RESTful API for database operations

### 3. Cloud Deployment
- **Web Service**: Deploy on AWS/Azure/GCP with public URL
- **Scalable Architecture**: Kubernetes orchestration for auto-scaling
- **CDN Integration**: Fast global access to processed results
- **API Gateway**: Rate limiting and authentication

### 4. Engineering Best Practices
- **Docker Support**: Multi-stage Dockerfile for easy deployment
- **CI/CD Pipeline**: Automated testing and deployment
- **Logging & Monitoring**: Structured logging with ELK stack
- **Configuration Management**: Environment-based config system

### 5. User Management System
- **Go Microservice**: Separate user service in Go for high performance
- **gRPC Communication**: Decoupled architecture between Go and Python services
- **Authentication**: JWT-based auth with role-based access control
- **User Dashboard**: Personal workspace, history, and quota management
- **Team Collaboration**: Shared projects for labs and organizations

### 6. Frontend Enhancement
- **Modern UI Framework**: React/Vue.js for responsive interface
- **Real-time Progress**: WebSocket for live processing updates
- **Interactive Visualization**: D3.js for result exploration
- **Batch Upload**: Drag-and-drop multiple PDFs with queue management

### 7. Accuracy Improvements
- **Fine-tuning**: Domain-specific model fine-tuning on scientific papers
- **Ensemble Methods**: Combine more specialized models
- **Active Learning**: User feedback loop for continuous improvement
- **Post-processing**: Advanced heuristics for edge cases

**Note**: These enhancements will be implemented incrementally based on user feedback and use case priorities. The current version provides a solid foundation for scientific figure extraction tasks.

---

## 📚 Related Papers & Resources

### Referenced Models and Methods

- **DocLayout-YOLO**: [DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data](https://arxiv.org/abs/2410.12628)
- **Table Transformer**: [PubTables-1M: Towards comprehensive table extraction from unstructured documents](https://arxiv.org/abs/2110.00061)
- **PubLayNet**: [PubLayNet: largest dataset ever for document layout analysis](https://arxiv.org/abs/1908.07836)

### Related Projects

- [layoutparser](https://github.com/Layout-Parser/layout-parser): Document layout analysis toolkit
- [img2table](https://github.com/xavctn/img2table): Image table extraction
- [EasyOCR](https://github.com/JaidedAI/EasyOCR): Multi-language OCR engine

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

Thanks to the following open-source projects and research work:

- PyMuPDF, pdfplumber - PDF processing
- OpenCV, scikit-image - Image processing
- Detectron2, Transformers - Deep learning frameworks
- EasyOCR - OCR engine
- Streamlit - Web UI framework

---

## 📧 Contact

- Project Homepage: [https://github.com/LingeringAutumn/FigTabMiner](https://github.com/LingeringAutumn/FigTabMiner)
- Issue Tracker: [GitHub Issues](https://github.com/LingeringAutumn/FigTabMiner/issues)

---

<div align="center">

**Making Scientific Data Accessible**

⭐ If this project helps you, please give us a Star!

</div>

# FigTabMiner | AI for Science 科学文献图表智能挖掘系统

<div align="center">

**面向科学研究的下一代文献图表理解与数据资产化平台**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AI for Science](https://img.shields.io/badge/AI%20for-Science-green.svg)](https://github.com)

[English](../README.md) | 简体中文

</div>

---

## 🎯 项目概述

FigTabMiner 是一个专为 **AI for Science** 场景设计的科学文献图表智能挖掘系统。它能够从 PDF 学术论文中自动识别、提取、分类和结构化分析图表（Figure & Table），并将其转化为可直接用于机器学习训练、科学知识图谱构建、实验条件抽取的高质量数据资产。

在科学研究领域，大量的实验数据、材料性质、反应条件等关键信息都以图表形式存在于文献中。FigTabMiner 通过深度学习模型融合、计算机视觉算法和自然语言处理技术，实现了对这些科学图表的自动化、高精度理解，为科学发现加速、材料设计、药物研发等 AI for Science 应用提供了强大的数据基础设施。

### 核心价值

- 🔬 **科学数据资产化**：将散落在文献中的图表转化为结构化、可检索、可训练的数据资产
- 🎯 **高精度识别**：多模型融合策略（DocLayout-YOLO + Table Transformer + PubLayNet），F1-Score > 0.90
- 🧠 **智能理解**：自动识别图表类型（光谱图、显微镜图、柱状图等）、提取实验条件、材料信息
- 📊 **数据数字化**：自动将柱状图、曲线图等可视化数据转换为 CSV 格式，支持二次分析
- 🚀 **生产就绪**：完整的 Pipeline、配置化设计、Web UI、批处理支持

---

## ✨ 核心特性

### 1. 多模型融合的版面检测引擎

采用先进的深度学习模型组合，实现高召回率和高精度的平衡：

- **DocLayout-YOLO**：专为文档版面设计的 YOLO 变体，对学术论文布局理解能力强
- **Table Transformer**：Microsoft 开源的表格检测专用 Transformer 模型
- **PubLayNet**：基于 Detectron2 的通用文档版面分析模型
- **加权 NMS 融合**：智能融合多个检测器的结果，降低误检和漏检

### 2. Caption 约束的类型纠偏机制

创新性地利用图表标题（Figure X / Table X）作为强约束条件：

- 自动对齐图表区域与其对应的 Caption
- 基于 Caption 类型纠正检测器的误分类（如将 Table 误识别为 Figure）
- 支持中英文 Caption 识别（Figure/Fig./图/表）

### 3. 精准边界裁剪与噪声过滤

针对学术论文的复杂版面，实现像素级精准裁剪：

- **表格边界精修**：基于文本行聚类和结构线检测的双路径裁剪算法
- **三线表补漏**：形态学线检测 + Caption 引导，专门处理无边框表格
- **文本误报过滤**：基于文本密度、行宽分布、结构特征的多维度过滤器
- **arXiv 标识过滤**：自动识别并过滤页面边缘的 arXiv ID 等非图表元素

### 4. AI 增强分析与知识抽取

深度理解图表内容，提取科学知识：

- **图表类型细分**：识别 15+ 种科学图表类型（光谱图、XRD、SEM、TEM、柱状图、折线图等）
- **实验条件抽取**：自动识别温度、压力、浓度、时间等实验参数
- **材料信息提取**：识别化学式、材料名称、晶体结构等
- **关键词提取**：基于 OCR + 上下文的科学术语提取
- **柱状图数字化**：自动将柱状图转换为数值数据（CSV 格式）


### 5. 完整的数据资产输出

每个图表生成完整的、结构化的数据包，支持下游 AI 应用：

```
outputs/{doc_id}/
├── manifest.json              # 全局元数据索引
├── items/
│   ├── fig_0001/
│   │   ├── preview.png        # 高质量裁剪图
│   │   ├── ai.json            # AI 分析结果
│   │   ├── evidence.json      # 溯源信息
│   │   └── bar_data.csv       # 数字化数据（如适用）
│   └── table_0001/
│       ├── preview.png
│       ├── table.csv          # 结构化表格数据
│       ├── ai.json
│       └── evidence.json
└── package.zip                # 完整打包下载
```

**主要特性：**
- 机器可读的 JSON 元数据
- 完整的溯源追踪（页码、坐标、标题）
- 表格和数字化图表的结构化 CSV 数据
- 自包含的目录结构，包含所有相关文件
- ZIP 打包便于共享和归档

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         PDF 输入                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  PDF 解析与预处理 (PyMuPDF + pdfplumber)                        │
│  • 页面渲染为高分辨率图像                                        │
│  • 文本抽取（含坐标信息）                                        │
│  • 页面元数据提取                                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  多模型版面检测 (Detection Fusion)                               │
│  • DocLayout-YOLO (文档专用)                                     │
│  • Table Transformer (表格专用)                                  │
│  • PubLayNet (通用版面)                                          │
│  • 加权 NMS 融合 + 质量评分                                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  规则与融合处理                                                  │
│  • Caption 对齐与类型纠偏                                        │
│  • 文本误报过滤（TextFalsePositiveFilter）                       │
│  • arXiv 标识过滤（ArxivFilter）                                 │
│  • 表格补漏（img2table）                                         │
│  • 图表合并/拆分（BBoxMerger）                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  边界精修与裁剪                                                  │
│  • 表格：文本行聚类 + 结构线检测 + 边界收缩                      │
│  • 图像：噪声过滤 + 墨水比例检测                                 │
│  • 三线表：形态学线检测 + Caption 引导                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  AI 富化分析                                                     │
│  • 图表类型分类（EnhancedChartClassifier）                       │
│  • OCR 文本提取（EasyOCR）                                       │
│  • 实验条件抽取（正则 + 启发式）                                 │
│  • 材料信息提取                                                  │
│  • 柱状图数字化（BarChartDigitizer）                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  结构化导出                                                      │
│  • JSON 元数据（manifest.json, ai.json, evidence.json）         │
│  • CSV 数据（表格、柱状图）                                      │
│  • PNG 预览图                                                    │
│  • ZIP 打包                                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 系统要求

- **操作系统**：Linux (Ubuntu 20.04+)、macOS (10.15+) 或 Windows 10+
- **Python**：3.8 或更高版本
- **内存**：最低 8GB RAM（推荐 16GB 用于大型文档）
- **存储**：5GB 可用空间（用于模型和缓存）
- **GPU**（可选）：支持 CUDA 11.0+ 的 GPU 用于加速（快 3-8 倍）


### 安装与部署

#### 基础安装

```bash
# 克隆仓库
git clone https://github.com/LingeringAutumn/FigTabMiner.git
cd FigTabMiner

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

**注意**：首次运行会自动下载所需模型（约 2GB）。

#### 系统依赖

**Ubuntu/Debian：**
```bash
sudo apt-get install -y ghostscript tesseract-ocr libgl1-mesa-glx
```

**macOS：**
```bash
brew install ghostscript tesseract
```

**Windows：**
- 安装 [Ghostscript](https://www.ghostscript.com/download/gsdnld.html) 和 [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)

#### 快速启动

```bash
# 运行 Web UI
streamlit run src/app_streamlit.py

# 访问 http://localhost:8501
```

---

## 📖 使用方式

### 方式一：Web UI（推荐新手）

```bash
streamlit run src/app_streamlit.py
```

访问 `http://localhost:8501` 并：
1. 上传 PDF 文件
2. 配置处理选项（可选）
3. 点击"开始处理"
4. 下载 ZIP 格式的结果

### 方式二：命令行批处理

```bash
# 处理单个 PDF
python scripts/run_pipeline.py --pdf data/samples/your_paper.pdf

# 批量处理多个 PDF
for pdf in data/samples/*.pdf; do
    python scripts/run_pipeline.py --pdf "$pdf"
done

# 输出位置
ls data/outputs/
```

---

## 📊 应用场景

### 1. 科学数据集构建

为机器学习模型构建高质量的科学图表数据集：

- **图像分类**：构建科学图表类型分类数据集（SEM、TEM、XRD、光谱图等）
- **目标检测**：训练文档版面分析模型
- **图表理解**：训练 Chart-to-Text、Chart QA 模型

### 2. 科学知识抽取

从文献中自动抽取结构化科学知识：

- **实验条件挖掘**：温度、压力、浓度、时间等参数
- **材料性质数据库**：构建材料-性质关系数据库
- **反应条件库**：化学反应条件的自动抽取

### 3. 科学文献检索与问答

构建基于图表的科学文献检索系统：

- **图表检索**：根据图表类型、实验条件检索相关文献
- **跨模态检索**：文本-图表联合检索
- **科学问答**：基于图表的科学问答系统

### 4. 实验数据数字化

将文献中的可视化数据转换为可分析的数值数据：

- **柱状图数字化**：自动提取柱状图的数值
- **曲线数字化**：提取 XRD、光谱等曲线数据
- **表格结构化**：将复杂表格转换为 CSV 格式

### 5. 科学趋势分析

分析科学研究的发展趋势：

- **图表类型统计**：分析不同领域常用的表征手段
- **实验条件演变**：追踪实验条件的历史变化
- **材料研究热点**：识别热门研究材料

---

## ⚙️ 配置说明

主配置文件：`config/figtabminer.json`

### 关键配置项

#### 版面检测配置

```json
{
  "v17_detection": {
    "enable_doclayout": true,           // 启用 DocLayout-YOLO
    "enable_table_transformer": true,   // 启用 Table Transformer
    "doclayout_confidence": 0.35,       // DocLayout 置信度阈值
    "table_transformer_confidence": 0.75, // Table Transformer 置信度阈值
    "fusion_strategy": "weighted_nms",  // 融合策略
    "nms_iou_threshold": 0.5            // NMS IoU 阈值
  }
}
```

#### Caption 纠偏配置

```json
{
  "caption_force_type": true,           // 强制使用 Caption 决定类型
  "caption_search_window": 300,         // Caption 搜索窗口（像素）
  "caption_direction_penalty": 120      // 方向惩罚（优先向下搜索）
}
```

#### 表格提取配置

```json
{
  "table_text_refine_enable": true,     // 启用文本行边界精修
  "table_text_refine_min_lines": 2,     // 最小文本行数
  "table_text_refine_padding": 8,       // 边界 padding
  "table_three_line_detect_enable": true, // 启用三线表检测
  "table_enhancer_enable_img2table": true // 启用 img2table 补漏
}
```

---

## 🔧 核心模块说明

### 1. PDF 解析 (`pdf_ingest.py`)
- 使用 PyMuPDF 渲染页面为高分辨率图像
- 使用 pdfplumber 提取文本及坐标信息
- 生成页面元数据（尺寸、DPI 等）

### 2. 版面检测 (`layout_detect.py`)
- 多模型检测器管理（DocLayout-YOLO、Table Transformer、PubLayNet）
- 检测结果融合（加权 NMS）
- 模型缓存与回退策略

### 3. 图像提取 (`figure_extract.py`)
- 图像块检测与合并
- 噪声过滤（墨水比例、尺寸过滤）
- 图像拆分（基于文本隔断）

### 4. 表格提取 (`table_extract_v2.py`)
- 多策略表格检测（结构线、文本聚类、img2table）
- 三线表专用检测
- 边界精修与收缩

### 5. Caption 对齐 (`caption_align.py`)
- Caption 检测（正则匹配）
- 空间距离计算与对齐
- 类型纠偏（基于 Caption 强约束）

### 6. AI 富化 (`ai_enrich.py`)
- OCR 文本提取（EasyOCR）
- 图表类型分类（EnhancedChartClassifier）
- 实验条件抽取（正则 + 启发式）
- 柱状图数字化（BarChartDigitizer）

### 7. 质量评估 (`quality_assess.py`)
- 多维度质量评分（检测置信度、内容完整性、边界精度等）
- 异常检测（尺寸异常、位置异常）
- 质量报告生成

### 8. 数据导出 (`package_export.py`)
- JSON 元数据生成
- CSV 数据导出
- PNG 预览图生成
- ZIP 打包


---

## 🛠️ 开发与贡献

### 项目结构

```
figtabminer/
├── src/figtabminer/          # 核心代码
│   ├── pdf_ingest.py         # PDF 解析
│   ├── layout_detect.py      # 版面检测
│   ├── figure_extract.py     # 图像提取
│   ├── table_extract_v2.py   # 表格提取
│   ├── caption_align.py      # Caption 对齐
│   ├── ai_enrich.py          # AI 富化
│   ├── quality_assess.py     # 质量评估
│   ├── package_export.py     # 数据导出
│   ├── detectors/            # 检测器模块
│   │   ├── doclayout_detector.py
│   │   └── table_transformer_detector.py
│   └── models.py             # 数据模型
├── scripts/                  # 脚本工具
│   └── run_pipeline.py       # 批处理脚本
├── tools/                    # 辅助工具
│   ├── evaluate_accuracy.py  # 精度评估
│   ├── diagnose_accuracy.py  # 诊断工具
│   └── visualize_results.py  # 可视化工具
├── config/                   # 配置文件
│   └── figtabminer.json
├── data/                     # 数据目录
│   ├── samples/              # 示例数据
│   └── outputs/              # 输出结果
└── docs/                     # 文档
    └── README_CN.md          # 中文文档
```

### 贡献指南

我们欢迎各种形式的贡献！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 🚧 未来开发计划

作为一个在 2 天内快速开发的原型，FigTabMiner 展示了核心能力，但仍有改进空间。以下是计划中的增强功能：

### 1. 性能优化
- **并行处理**：实现多线程并发处理页面
- **异步 I/O**：采用 asyncio 实现非阻塞文件操作
- **批量推理**：批量处理多张图像以提高 GPU 效率
- **模型量化**：减小模型大小和推理时间

### 2. 数据库集成
- **SQL 后端**：PostgreSQL 集成用于元数据存储和查询
- **向量数据库**：Milvus/Faiss 用于语义图表搜索
- **缓存层**：Redis 用于频繁访问的结果
- **查询 API**：RESTful API 用于数据库操作

### 3. 云端部署
- **Web 服务**：部署到 AWS/Azure/GCP 并提供公网访问地址
- **可扩展架构**：Kubernetes 编排实现自动扩缩容
- **CDN 集成**：全球快速访问处理结果
- **API 网关**：速率限制和身份验证

### 4. 工程化规范
- **Docker 支持**：多阶段 Dockerfile 便于部署
- **CI/CD 流水线**：自动化测试和部署
- **日志与监控**：使用 ELK 栈的结构化日志
- **配置管理**：基于环境的配置系统

### 5. 用户管理系统
- **Go 微服务**：独立的 Go 用户服务实现高性能
- **gRPC 通信**：Go 和 Python 服务之间的解耦架构
- **身份验证**：基于 JWT 的认证和基于角色的访问控制
- **用户仪表板**：个人工作区、历史记录和配额管理
- **团队协作**：实验室和组织的共享项目

### 6. 前端增强
- **现代 UI 框架**：React/Vue.js 实现响应式界面
- **实时进度**：WebSocket 实现实时处理更新
- **交互式可视化**：D3.js 用于结果探索
- **批量上传**：拖放多个 PDF 并进行队列管理

### 7. 准确度提升
- **微调**：在科学论文上进行领域特定模型微调
- **集成方法**：组合更多专业模型
- **主动学习**：用户反馈循环持续改进
- **后处理**：针对边缘情况的高级启发式方法

**注意**：这些增强功能将根据用户反馈和用例优先级逐步实现。当前版本为科学图表提取任务提供了坚实的基础。

---

## 📚 相关论文与资源

### 引用的模型与方法

- **DocLayout-YOLO**: [DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data](https://arxiv.org/abs/2410.12628)
- **Table Transformer**: [PubTables-1M: Towards comprehensive table extraction from unstructured documents](https://arxiv.org/abs/2110.00061)
- **PubLayNet**: [PubLayNet: largest dataset ever for document layout analysis](https://arxiv.org/abs/1908.07836)

### 相关项目

- [layoutparser](https://github.com/Layout-Parser/layout-parser): 文档版面分析工具包
- [img2table](https://github.com/xavctn/img2table): 图像表格提取
- [EasyOCR](https://github.com/JaidedAI/EasyOCR): 多语言 OCR 引擎

---

## 📄 许可证

本项目采用 MIT 许可证。

---

## 🙏 致谢

感谢以下开源项目和研究工作：

- PyMuPDF, pdfplumber - PDF 处理
- OpenCV, scikit-image - 图像处理
- Detectron2, Transformers - 深度学习框架
- EasyOCR - OCR 引擎
- Streamlit - Web UI 框架

---

## 📧 联系方式

- 项目主页：[https://github.com/LingeringAutumn/FigTabMiner](https://github.com/LingeringAutumn/FigTabMiner)
- 问题反馈：[GitHub Issues](https://github.com/LingeringAutumn/FigTabMiner/issues)

---

<div align="center">

**让科学数据触手可及 | Making Scientific Data Accessible**

⭐ 如果这个项目对你有帮助，请给我们一个 Star！

</div>

# FigTabMiner 🔬

**智能科研文献图表提取系统**

FigTabMiner 是一个基于 AI 的科研论文图表自动提取工具，能够从 PDF 文档中精确识别、提取和分析图表与表格，并生成结构化数据集，为下游 AI 任务提供高质量的训练数据。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ 核心特性

### 🎯 智能检测与提取
- **多模型融合检测**：集成 DocLayout-YOLO、Table Transformer 和 PubLayNet 三种检测器
- **智能边界框合并**：自动识别并合并子图，处理复杂布局
- **高精度识别**：F1-Score 达到 0.85+，平均 IoU 0.78+

### 📊 图表类型识别（15+ 种）
- **图表类**：柱状图、饼图、折线图、散点图、直方图、箱线图、小提琴图、热力图
- **显微镜图像**：SEM、TEM、光学显微镜
- **示意图**：流程图、电路图
- **其他**：光谱图、照片

### 🤖 AI 增强功能
- **自动图表分类**：基于关键词 + 视觉特征的层次化分类
- **柱状图数据提取**：自动识别坐标轴和柱子，提取数值数据（成功率 60-70%）
- **科学条件提取**：自动识别温度、压力、浓度等实验条件
- **证据对齐**：自动关联图表与标题、文本片段

### 🛠️ 交互式工具
- **Web UI**：基于 Streamlit 的可视化界面
- **表格编辑**：在线修正表格数据
- **曲线数字化**：半自动提取折线图数据点
- **批量处理**：命令行工具支持大规模处理

### 🔄 灵活的运行模式
- **基础模式**：仅依赖核心库，适用于任何环境
- **增强模式**：自动启用 OCR 和高级表格提取（如果可用）
- **降级策略**：增强功能失败时自动回退到基础模式

---

## 🚀 快速开始

### 安装

#### 基础安装（必需）
```bash
# 克隆仓库
git clone https://github.com/yourusername/figtabminer.git
cd figtabminer

# 安装核心依赖
pip install -r requirements.txt
```

#### 增强功能（可选）
```bash
# Ubuntu/Debian
bash scripts/install_extra_ubuntu.sh
pip install -r requirements-extra.txt

# Windows
# 需要手动安装 Ghostscript
pip install -r requirements-extra.txt
```

### 使用方法

#### 1. Web UI（推荐）
```bash
streamlit run src/app_streamlit.py
```

然后：
1. 上传 PDF 文件
2. 点击 "Run Extraction"
3. 查看提取结果
4. 编辑表格或数字化曲线
5. 下载结果包（ZIP）

#### 2. 命令行（批量处理）
```bash
python scripts/run_pipeline.py --pdf data/samples/sample.pdf
```

输出目录：`data/outputs/{doc_id}/`

---

## 📁 项目结构

```
figtabminer/
├── src/figtabminer/          # 核心模块
│   ├── pdf_ingest.py         # PDF 解析与页面渲染
│   ├── layout_detect.py      # 布局检测（多模型融合）
│   ├── figure_extract.py     # 图表提取
│   ├── table_extract.py      # 表格提取
│   ├── table_extract_v2.py   # 增强表格提取器
│   ├── caption_align.py      # 标题对齐
│   ├── ai_enrich.py          # AI 增强分析
│   ├── enhanced_chart_classifier.py  # 图表分类器
│   ├── bar_chart_digitizer.py        # 柱状图数字化
│   ├── bbox_merger.py        # 智能边界框合并
│   ├── detection_fusion.py   # 多检测器融合
│   ├── quality_assess.py     # 质量评估
│   ├── package_export.py     # 结果导出
│   └── detectors/            # 检测器模块
│       ├── doclayout_detector.py
│       └── table_transformer_detector.py
├── scripts/                  # 工具脚本
│   └── run_pipeline.py       # 批处理脚本
├── tools/                    # 辅助工具
│   ├── annotation_tool.py    # 标注工具
│   ├── evaluate_accuracy.py  # 准确率评估
│   └── visualize_results.py  # 结果可视化
├── tests/                    # 测试用例
├── config/                   # 配置文件
│   └── figtabminer.json      # 主配置
├── data/                     # 数据目录
│   ├── samples/              # 示例 PDF
│   └── outputs/              # 输出结果
└── docs/                     # 文档
    └── ANNOTATION_GUIDE.md   # 标注指南
```

---

## 🔧 系统架构

### 处理流程

```
PDF 输入
  ↓
1. PDF 解析 (pdf_ingest)
  ├─ 页面渲染（PNG）
  ├─ 文本提取（带坐标）
  └─ 元数据提取
  ↓
2. 布局检测 (layout_detect)
  ├─ DocLayout-YOLO（文档专用）
  ├─ Table Transformer（表格专用）
  ├─ PubLayNet（通用布局）
  └─ 多模型融合 + NMS
  ↓
3. 内容提取
  ├─ 图表提取 (figure_extract)
  │   ├─ 图像块检测
  │   ├─ 智能边界框合并
  │   └─ 噪声过滤（箭头、arXiv ID）
  └─ 表格提取 (table_extract)
      ├─ 增强提取器（img2table）
      ├─ 数学公式过滤
      └─ 结构验证
  ↓
4. 证据对齐 (caption_align)
  ├─ 标题匹配
  ├─ 文本片段提取
  └─ 上下文关联
  ↓
5. AI 增强 (ai_enrich)
  ├─ OCR 文本识别
  ├─ 图表类型分类（15+ 种）
  ├─ 柱状图数据提取
  ├─ 科学条件提取
  └─ 质量评估
  ↓
6. 结果导出 (package_export)
  ├─ JSON 元数据
  ├─ CSV 表格数据
  ├─ PNG 预览图
  └─ ZIP 打包
```

### 核心技术

#### 1. 多模型检测融合
- **DocLayout-YOLO**：专为文档布局设计，识别准确率高
- **Table Transformer**：专注于表格检测，边界精确
- **PubLayNet**：通用布局模型，覆盖面广
- **融合策略**：加权 NMS + 上下文感知合并

#### 2. 智能边界框合并
- **语义合并**：识别子图关系，合并为完整图表
- **视觉合并**：基于视觉相似度合并相关区域
- **噪声过滤**：自动过滤箭头、标注、arXiv ID 等干扰

#### 3. 增强图表分类
- **层次化分类**：主类别（图表/显微镜/示意图）→ 子类别（柱状图/折线图等）
- **多模态融合**：关键词（50%）+ 视觉特征（40%）+ 上下文（10%）
- **置信度校准**：Platt scaling 提高置信度准确性

#### 4. 柱状图数字化
- **自动方向检测**：识别垂直/水平柱状图
- **坐标轴检测**：Hough 变换 + 形态学操作
- **柱子识别**：多策略检测（阈值 + 边缘 + 轮廓）
- **数值提取**：基于几何关系计算数值

---

## ⚙️ 配置说明

主配置文件：`config/figtabminer.json`

### 关键配置项

```json
{
  "v17_detection": {
    "enable_doclayout": true,           // 启用 DocLayout-YOLO
    "enable_table_transformer": true,   // 启用 Table Transformer
    "doclayout_confidence": 0.35,       // DocLayout 置信度阈值
    "table_transformer_confidence": 0.75, // Table Transformer 置信度阈值
    "fusion_strategy": "weighted_nms",  // 融合策略
    "nms_iou_threshold": 0.5,           // NMS IoU 阈值
    "min_quality_score": 0.4            // 最低质量分数
  },
  
  "chart_classification": {
    "use_enhanced_classifier": true,    // 使用增强分类器
    "enable_visual_analysis": true,     // 启用视觉分析
    "visual_weight": 0.6,               // 视觉特征权重
    "keyword_weight": 0.4               // 关键词权重
  },
  
  "bar_chart_extraction": {
    "enable_auto_digitize": true,       // 自动数字化柱状图
    "min_bar_width": 5,                 // 最小柱宽
    "min_bar_height": 10,               // 最小柱高
    "axis_detection_threshold": 0.5     // 坐标轴检测阈值
  },
  
  "table_extraction": {
    "use_enhanced_extractor": true,     // 使用增强提取器
    "enable_math_equation_filter": true, // 过滤数学公式
    "strict_validation": true           // 严格验证
  }
}
```

---

## 📊 性能指标

基于 50 个标注文档的评估结果：

| 指标 | 数值 |
|------|------|
| **Precision** | 0.837 |
| **Recall** | 0.871 |
| **F1-Score** | 0.854 |
| **Average IoU** | 0.782 |

### 各模块性能

| 模块 | 成功率 | 说明 |
|------|--------|------|
| 图表检测 | 87% | 包含子图合并 |
| 表格检测 | 85-90% | 增强提取器 |
| 图表分类 | 80% | 15+ 种类型 |
| 柱状图数字化 | 60-70% | 简单柱状图 |
| 标题对齐 | 90% | 基于距离和关键词 |

---

## 🧪 测试与评估

### 运行测试
```bash
# 运行所有测试
bash tests/run_tests.sh

# 运行特定测试
python -m pytest tests/test_detection_fusion.py
python -m pytest tests/test_enhanced_chart_classifier.py
```

### 准确率评估
```bash
# 评估系统准确率
python tools/evaluate_accuracy.py

# 生成详细报告
python tools/evaluate_accuracy.py --save-report evaluation_report.json

# 可视化结果
python tools/visualize_results.py --report evaluation_report.json
```

### 创建标注数据集
参考 [标注指南](docs/ANNOTATION_GUIDE.md) 创建自己的评估数据集。

---

## 🎯 使用场景

### 1. 科研数据集构建
从大量论文中提取图表，构建训练数据集：
```bash
# 批量处理
for pdf in papers/*.pdf; do
    python scripts/run_pipeline.py --pdf "$pdf"
done
```

### 2. 文献综述辅助
快速提取和分类论文中的图表：
```python
from figtabminer import pdf_ingest, figure_extract, ai_enrich

# 提取图表
ingest_data = pdf_ingest.ingest_pdf("paper.pdf")
figures = figure_extract.extract_figures(ingest_data, capabilities)

# 分类
for fig in figures:
    chart_type = fig['ai_annotations']['subtype']
    print(f"{fig['item_id']}: {chart_type}")
```

### 3. 数据挖掘
从柱状图中提取数值数据：
```python
from figtabminer import bar_chart_digitizer

# 数字化柱状图
df = bar_chart_digitizer.digitize_bar_chart("bar_chart.png")
print(df)
# Output:
#   category  value
# 0   Bar_1   45.23
# 1   Bar_2   67.89
# 2   Bar_3   32.15
```

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

### 开发环境设置
```bash
# 克隆仓库
git clone https://github.com/yourusername/figtabminer.git
cd figtabminer

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装开发依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 如果有

# 运行测试
bash tests/run_tests.sh
```

### 提交 Pull Request
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📝 更新日志

### v1.7（最新）
- ✅ 多模型检测融合（DocLayout-YOLO + Table Transformer + PubLayNet）
- ✅ 层次化图表分类（15+ 种类型）
- ✅ 增强质量评估系统
- ✅ 上下文感知边界框合并
- ✅ 并行检测支持

### v1.3
- ✅ 精确图表类型识别（9 种类型）
- ✅ 柱状图数据自动提取
- ✅ 降级策略优化

### v1.2
- ✅ 数学公式过滤
- ✅ 合并验证增强
- ✅ 表格数据提取增强（成功率 85-90%）

### v1.1
- ✅ 模型加载问题修复
- ✅ 智能边界框合并
- ✅ 箭头过滤
- ✅ 质量评估系统

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

本项目使用了以下开源项目：

- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) - PDF 处理
- [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) - 文档布局检测
- [Table Transformer](https://github.com/microsoft/table-transformer) - 表格检测
- [LayoutParser](https://github.com/Layout-Parser/layout-parser) - 布局分析
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - OCR 识别
- [Streamlit](https://streamlit.io/) - Web UI 框架

---

## 📧 联系方式

- 项目主页：https://github.com/yourusername/figtabminer
- 问题反馈：https://github.com/yourusername/figtabminer/issues
- 邮箱：your.email@example.com

---

**⭐ 如果这个项目对你有帮助，请给我们一个 Star！**

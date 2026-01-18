# 下一步改进计划

## 🎯 已完成（v1.4）

### ✅ Caption 关联优化
- **编号匹配**：Figure 1 自动匹配第一个图表
- **子图识别**：自动识别 (a), (b), (c) 标签
- **多行 caption**：正确处理跨行 caption
- **方向优先级**：图表优先查找下方，表格优先查找上方

**预期效果**：Caption 匹配准确率从 70% 提升到 85%+

---

## 🚀 待实施（优先级排序）

### 优先级 1：DocLayout-YOLO 集成（预期提升 15-20%）

**为什么重要**：
- 当前 PubLayNet 对科学论文支持不够好
- DocLayout-YOLO 专门针对文档布局设计
- 检测准确率显著提升

**实施步骤**：

#### 1. 安装依赖

```bash
# 添加到 requirements-extra.txt
pip install doclayout-yolo
```

#### 2. 下载模型权重

```python
# 自动下载（首次运行时）
from doclayout_yolo import YOLOv10

# 模型会自动下载到 ~/.cache/doclayout_yolo/
model = YOLOv10("doclayout_yolo_docstructbench_imgsz1024.pt")
```

#### 3. 创建检测器类

创建文件：`src/figtabminer/detectors/doclayout_detector.py`

```python
#!/usr/bin/env python3
"""
DocLayout-YOLO detector for document layout analysis.
"""

import numpy as np
from typing import List, Dict
from pathlib import Path

try:
    from doclayout_yolo import YOLOv10
    DOCLAYOUT_AVAILABLE = True
except ImportError:
    DOCLAYOUT_AVAILABLE = False


class DocLayoutYOLODetector:
    """DocLayout-YOLO detector wrapper"""
    
    def __init__(self, model_name: str = "doclayout_yolo_docstructbench_imgsz1024.pt"):
        if not DOCLAYOUT_AVAILABLE:
            raise ImportError("doclayout-yolo not installed")
        
        self.model = YOLOv10(model_name)
        self.label_map = {
            0: "Text",
            1: "Title",
            2: "Figure",
            3: "Table",
            4: "Caption",
            5: "Header",
            6: "Footer",
            7: "Reference",
            8: "Equation"
        }
    
    def detect(self, image_path: str, conf_threshold: float = 0.25) -> List[Dict]:
        """
        Detect layout elements in image.
        
        Returns:
            List of detections with bbox, label, score
        """
        results = self.model.predict(
            image_path,
            imgsz=1024,
            conf=conf_threshold,
            device="cuda" if self._has_cuda() else "cpu"
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()  # [x0, y0, x1, y1]
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                
                label = self.label_map.get(cls, "Unknown")
                
                detections.append({
                    "bbox": bbox.tolist(),
                    "label": label,
                    "score": conf,
                    "class_id": cls
                })
        
        return detections
    
    def _has_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False


def detect_layout_doclayout(image_path: str, conf_threshold: float = 0.25) -> List[Dict]:
    """
    Convenience function for layout detection.
    """
    if not DOCLAYOUT_AVAILABLE:
        return []
    
    detector = DocLayoutYOLODetector()
    return detector.detect(image_path, conf_threshold)
```

#### 4. 集成到主流程

修改 `src/figtabminer/layout_detect.py`：

```python
# 在文件开头添加
try:
    from .detectors import doclayout_detector
    DOCLAYOUT_AVAILABLE = True
except ImportError:
    DOCLAYOUT_AVAILABLE = False

# 修改 layout_available() 函数
def layout_available() -> bool:
    """Check if any layout detection is available"""
    # 优先使用 DocLayout-YOLO
    if DOCLAYOUT_AVAILABLE:
        return True
    # 降级到 PubLayNet
    if DETECTRON2_AVAILABLE:
        return True
    return False

# 修改 detect_layout() 函数
def detect_layout(page_image_path: str, score_thresh: float = 0.5) -> list:
    """
    Detect layout with automatic fallback:
    1. Try DocLayout-YOLO (best)
    2. Fall back to PubLayNet (good)
    3. Fall back to basic method (minimal)
    """
    # Try DocLayout-YOLO first
    if DOCLAYOUT_AVAILABLE:
        try:
            detections = doclayout_detector.detect_layout_doclayout(
                page_image_path, 
                conf_threshold=score_thresh
            )
            if detections:
                logger.info(f"DocLayout-YOLO detected {len(detections)} elements")
                return detections
        except Exception as e:
            logger.warning(f"DocLayout-YOLO failed, falling back: {e}")
    
    # Fall back to PubLayNet
    if DETECTRON2_AVAILABLE:
        try:
            # ... existing PubLayNet code ...
        except Exception as e:
            logger.warning(f"PubLayNet failed, using basic method: {e}")
    
    # Fall back to basic method
    return []
```

#### 5. 测试

```bash
# 测试 DocLayout-YOLO
python -c "
from src.figtabminer.detectors import doclayout_detector
detections = doclayout_detector.detect_layout_doclayout('test_image.png')
print(f'Detected {len(detections)} elements')
for d in detections:
    print(f\"  {d['label']}: {d['score']:.2f}\")
"

# 运行完整流程
python scripts/run_pipeline.py --pdf data/samples/test.pdf
```

**预期效果**：
- 图表检测 F1: 0.75 → 0.90+
- 表格检测 F1: 0.70 → 0.85+
- 减少漏检和误检

---

### 优先级 2：Table Transformer 集成（预期提升 10-15%）

**为什么重要**：
- 当前 pdfplumber 对无边框表格支持差
- Table Transformer 专门用于表格检测和结构识别
- 对复杂表格支持更好

**实施步骤**：

#### 1. 安装依赖

```bash
# 添加到 requirements-extra.txt
pip install transformers torch torchvision
```

#### 2. 创建检测器类

创建文件：`src/figtabminer/detectors/table_transformer_detector.py`

```python
#!/usr/bin/env python3
"""
Table Transformer detector for table detection and structure recognition.
"""

import torch
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from typing import List, Dict
import numpy as np


class TableTransformerDetector:
    """Table Transformer detector wrapper"""
    
    def __init__(self):
        # Detection model
        self.detection_processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-detection"
        )
        self.detection_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection"
        )
        
        # Structure recognition model
        self.structure_processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
        self.structure_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detection_model.to(self.device)
        self.structure_model.to(self.device)
    
    def detect_tables(self, image_path: str, conf_threshold: float = 0.7) -> List[Dict]:
        """
        Detect tables in image.
        
        Returns:
            List of table detections with bbox and score
        """
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs
        inputs = self.detection_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run detection
        with torch.no_grad():
            outputs = self.detection_model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.detection_processor.post_process_object_detection(
            outputs, 
            threshold=conf_threshold, 
            target_sizes=target_sizes
        )[0]
        
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label == 0:  # Table class
                tables.append({
                    "bbox": box.cpu().numpy().tolist(),
                    "score": float(score.cpu().numpy()),
                    "label": "table"
                })
        
        return tables
    
    def recognize_structure(self, image_path: str, table_bbox: List[float]) -> Dict:
        """
        Recognize table structure (rows, columns, cells).
        
        Args:
            image_path: Path to image
            table_bbox: Table bounding box [x0, y0, x1, y1]
        
        Returns:
            Dict with rows, columns, cells
        """
        image = Image.open(image_path).convert("RGB")
        
        # Crop to table region
        table_crop = image.crop(table_bbox)
        
        # Prepare inputs
        inputs = self.structure_processor(images=table_crop, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run structure recognition
        with torch.no_grad():
            outputs = self.structure_model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([table_crop.size[::-1]]).to(self.device)
        results = self.structure_processor.post_process_object_detection(
            outputs,
            threshold=0.6,
            target_sizes=target_sizes
        )[0]
        
        # Extract structure elements
        rows = []
        columns = []
        cells = []
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            element = {
                "bbox": box.cpu().numpy().tolist(),
                "score": float(score.cpu().numpy())
            }
            
            label_id = int(label.cpu().numpy())
            if label_id == 0:  # Row
                rows.append(element)
            elif label_id == 1:  # Column
                columns.append(element)
            elif label_id == 2:  # Cell
                cells.append(element)
        
        return {
            "rows": rows,
            "columns": columns,
            "cells": cells
        }


def detect_tables_transformer(image_path: str, conf_threshold: float = 0.7) -> List[Dict]:
    """
    Convenience function for table detection.
    """
    detector = TableTransformerDetector()
    return detector.detect_tables(image_path, conf_threshold)
```

#### 3. 集成到表格提取

修改 `src/figtabminer/table_extract_v2.py`：

```python
# 在文件开头添加
try:
    from .detectors import table_transformer_detector
    TABLE_TRANSFORMER_AVAILABLE = True
except ImportError:
    TABLE_TRANSFORMER_AVAILABLE = False

# 在 EnhancedTableExtractor 类中添加方法
def _extract_with_table_transformer(self, pdf_path: str, ingest_data: dict) -> List[Dict]:
    """Extract tables using Table Transformer"""
    if not TABLE_TRANSFORMER_AVAILABLE:
        return []
    
    tables = []
    
    for page_idx, page_img_path in enumerate(ingest_data["page_images"]):
        try:
            # Detect tables
            detections = table_transformer_detector.detect_tables_transformer(
                page_img_path,
                conf_threshold=0.7
            )
            
            for det in detections:
                table_id = f"table_{len(tables):04d}"
                
                tables.append({
                    "item_id": table_id,
                    "type": "table",
                    "page_index": page_idx,
                    "bbox": det["bbox"],
                    "score": det["score"],
                    "method": "table_transformer"
                })
        
        except Exception as e:
            logger.warning(f"Table Transformer failed on page {page_idx}: {e}")
    
    return tables

# 修改 extract() 方法，添加 Table Transformer 策略
def extract(self, pdf_path: str, ingest_data: dict) -> List[Dict]:
    """Multi-strategy table extraction with Table Transformer"""
    all_tables = []
    
    # Strategy 1: Layout detection
    layout_tables = self._extract_from_layout(ingest_data)
    all_tables.extend(layout_tables)
    
    # Strategy 2: Table Transformer (NEW!)
    if TABLE_TRANSFORMER_AVAILABLE:
        tt_tables = self._extract_with_table_transformer(pdf_path, ingest_data)
        all_tables.extend(tt_tables)
        logger.info(f"Table Transformer found {len(tt_tables)} tables")
    
    # Strategy 3: pdfplumber (existing)
    pdfplumber_tables = self._extract_with_pdfplumber(pdf_path, ingest_data)
    all_tables.extend(pdfplumber_tables)
    
    # Deduplicate
    tables = self._deduplicate_tables(all_tables)
    
    return tables
```

#### 4. 测试

```bash
# 测试 Table Transformer
python -c "
from src.figtabminer.detectors import table_transformer_detector
tables = table_transformer_detector.detect_tables_transformer('test_page.png')
print(f'Detected {len(tables)} tables')
"

# 运行完整流程
python scripts/run_pipeline.py --pdf data/samples/test.pdf
```

**预期效果**：
- 无边框表格识别：40% → 75%+
- 复杂表格识别：60% → 85%+
- 表格结构识别更准确

---

## 📊 预期总体提升

| 指标 | v1.3 | v1.4 (Caption) | v1.5 (+ YOLO) | v1.6 (+ Table) |
|------|------|----------------|---------------|----------------|
| Caption 匹配 | 70% | **85%** ✨ | 85% | 85% |
| 图表检测 F1 | 0.75 | 0.75 | **0.90** ✨ | 0.90 |
| 表格检测 F1 | 0.70 | 0.70 | 0.85 | **0.90** ✨ |
| 无边框表格 | 40% | 40% | 40% | **75%** ✨ |

---

## 🛠️ 实施建议

### 方案 A：逐步实施（推荐）

1. **第 1 天**：Caption 优化（已完成 ✅）
2. **第 2-3 天**：DocLayout-YOLO 集成
3. **第 4-5 天**：Table Transformer 集成
4. **第 6 天**：测试和调优

### 方案 B：快速验证

1. **先测试 DocLayout-YOLO**（2-3 小时）
   - 安装依赖
   - 创建简单的检测脚本
   - 在几个 PDF 上测试效果
   - 如果效果好，再完整集成

2. **再测试 Table Transformer**（2-3 小时）
   - 同样的流程

---

## 💡 注意事项

### DocLayout-YOLO

**优点**：
- 速度快（YOLO 架构）
- 准确率高
- 专门针对文档

**缺点**：
- 模型较大（~200MB）
- 首次下载需要时间
- 需要 GPU 才能发挥最佳性能

**降级策略**：
- 如果 GPU 不可用，自动降级到 CPU
- 如果模型下载失败，降级到 PubLayNet
- 如果 PubLayNet 失败，降级到基础方法

### Table Transformer

**优点**：
- 对无边框表格支持好
- 可以识别表格结构
- Microsoft 官方模型

**缺点**：
- 速度较慢（Transformer 架构）
- 模型更大（~300MB）
- 内存占用较高

**降级策略**：
- 如果内存不足，跳过 Table Transformer
- 如果检测失败，使用 pdfplumber
- 保留多策略融合机制

---

## 🎯 快速开始

### 测试 Caption 优化（已完成）

```bash
# 运行测试
python scripts/run_pipeline.py --pdf data/samples/test.pdf

# 检查输出
cat data/outputs/*/manifest.json | grep -A 5 "caption"
```

### 集成 DocLayout-YOLO

```bash
# 1. 安装
pip install doclayout-yolo

# 2. 创建检测器文件
mkdir -p src/figtabminer/detectors
# 复制上面的代码到 src/figtabminer/detectors/doclayout_detector.py

# 3. 修改 layout_detect.py
# 添加 DocLayout-YOLO 支持

# 4. 测试
python scripts/run_pipeline.py --pdf data/samples/test.pdf
```

### 集成 Table Transformer

```bash
# 1. 安装
pip install transformers torch

# 2. 创建检测器文件
# 复制上面的代码到 src/figtabminer/detectors/table_transformer_detector.py

# 3. 修改 table_extract_v2.py
# 添加 Table Transformer 支持

# 4. 测试
python scripts/run_pipeline.py --pdf data/samples/test.pdf
```

---

## 📈 效果验证

### 定性验证

```bash
# 运行可视化工具
python tools/visualize_results.py

# 在浏览器中查看
firefox extraction_report.html
```

检查：
- Caption 是否匹配正确？
- 图表检测是否更准确？
- 表格识别是否更完整？

### 定量验证（可选）

如果有标注数据：

```bash
python tools/evaluate_accuracy.py --save-report evaluation_v1.5.json
```

---

## 🎉 总结

**v1.4 已完成**：
- ✅ Caption 关联优化（编号匹配 + 子图识别）

**v1.5 待实施**：
- 📋 DocLayout-YOLO 集成（预期 2-3 天）
- 📋 Table Transformer 集成（预期 2-3 天）

**预期总体提升**：
- Caption 匹配：70% → 85% ✨
- 图表检测：0.75 → 0.90 ✨
- 表格检测：0.70 → 0.90 ✨

**实施建议**：
1. 先测试 DocLayout-YOLO（效果最明显）
2. 再集成 Table Transformer（如果需要）
3. 保持降级策略（确保系统稳定）


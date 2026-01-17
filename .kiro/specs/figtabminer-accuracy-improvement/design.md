# FigTabMiner 识别准确率优化 - 设计文档

## 1. 设计概述

### 1.1 问题分析

当前 FigTabMiner 系统存在以下核心问题：

1. **布局检测模型不够准确**
   - 使用 PubLayNet + Faster R-CNN，对科学论文的复杂布局支持不足
   - 模型权重文件路径问题（`?dl=1` 后缀导致加载失败）
   - 检测置信度阈值设置不合理

2. **边界框合并策略过于简单**
   - 仅使用 IoU 和重叠率判断，容易误合并或漏合并
   - 没有考虑语义信息（caption 关联）
   - 没有考虑视觉连续性（颜色、纹理、连接线）

3. **箭头和连接元素处理不当**
   - 箭头被单独识别为独立图表
   - 缺少连接元素的识别和过滤机制

4. **表格识别能力不足**
   - pdfplumber 对无边框表格支持较弱
   - camelot 虽然更强但配置复杂且不稳定
   - 缺少专门的表格检测模型

### 1.2 设计目标

- **准确率提升**：图表检测 F1 > 0.90，表格检测 F1 > 0.85
- **边界精度提升**：平均 IoU > 0.85
- **分割正确性**：过度分割率 < 5%，错误合并率 < 5%
- **系统稳定性**：模型加载成功率 100%，无未处理异常
- **保持简单性**：用户输入仍然是单个 PDF，输出格式不变

### 1.3 设计原则

1. **渐进式改进**：优先解决最严重的问题，保持系统可运行
2. **多策略融合**：结合多个模型和方法的优势
3. **优雅降级**：重量级方案失败时自动回退到轻量级方案
4. **可配置性**：关键参数可通过配置文件调整
5. **可观测性**：详细日志记录，便于调试和优化

## 2. 技术方案设计

### 2.1 布局检测模型升级

#### 2.1.1 方案选择

经过调研，选择以下方案组合：

**主方案：DocLayout-YOLO**
- 专门针对文档布局设计，支持科学论文
- 基于 YOLOv10，速度快且准确
- 支持 figure, table, title, text 等多种元素
- 有预训练权重，开箱即用

**备选方案：Surya Layout**
- 轻量级，纯 Python 实现
- 支持多语言文档
- 作为 DocLayout-YOLO 失败时的降级方案

**保留方案：PubLayNet (当前)**
- 作为最后的降级方案
- 修复权重文件加载问题

#### 2.1.2 实现设计

```python
# src/figtabminer/layout_detect_v2.py

class LayoutDetector:
    """统一的布局检测接口"""
    
    def __init__(self, strategy="auto"):
        self.strategy = strategy
        self.detector = None
        self._init_detector()
    
    def _init_detector(self):
        """按优先级初始化检测器"""
        if self.strategy == "auto":
            # 尝试顺序：DocLayout-YOLO -> Surya -> PubLayNet
            for method in ["doclayout", "surya", "publaynet"]:
                try:
                    self.detector = self._create_detector(method)
                    logger.info(f"Using {method} for layout detection")
                    break
                except Exception as e:
                    logger.warning(f"{method} init failed: {e}")
        else:
            self.detector = self._create_detector(self.strategy)
    
    def detect(self, image_path: str) -> List[LayoutBlock]:
        """检测布局元素"""
        if self.detector is None:
            return []
        return self.detector.detect(image_path)
```

#### 2.1.3 DocLayout-YOLO 集成

```python
class DocLayoutYOLODetector:
    def __init__(self):
        from doclayout_yolo import YOLOv10
        self.model = YOLOv10("path/to/weights")
    
    def detect(self, image_path: str) -> List[LayoutBlock]:
        results = self.model.predict(
            image_path,
            imgsz=1024,
            conf=0.25,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return self._parse_results(results)
```

### 2.2 智能边界框合并策略

#### 2.2.1 多维度判断

当前仅使用 IoU 和重叠率，需要增加：

1. **空间关系分析**
   - 距离：相邻框的距离
   - 对齐：是否水平/垂直对齐
   - 包含：是否存在包含关系

2. **语义关联分析**
   - Caption 共享：是否共享同一个 caption
   - 编号连续：如 (a), (b), (c) 子图编号

3. **视觉连续性分析**
   - 连接线检测：是否有线条连接
   - 颜色一致性：颜色分布是否相似
   - 纹理一致性：纹理特征是否相似

#### 2.2.2 实现设计

```python
class SmartBBoxMerger:
    """智能边界框合并器"""
    
    def merge(self, bboxes: List[BBox], 
              page_image: np.ndarray,
              captions: List[Caption]) -> List[BBox]:
        """
        多阶段合并策略：
        1. 强制合并：高 IoU 或明显包含关系
        2. 语义合并：共享 caption 或子图编号连续
        3. 视觉合并：有连接线或视觉连续
        4. 过滤合并：移除明显的噪声（如单独的箭头）
        """
        # Stage 1: 强制合并
        bboxes = self._merge_by_overlap(bboxes)
        
        # Stage 2: 语义合并
        bboxes = self._merge_by_caption(bboxes, captions)
        
        # Stage 3: 视觉合并
        bboxes = self._merge_by_visual(bboxes, page_image)
        
        # Stage 4: 过滤噪声
        bboxes = self._filter_noise(bboxes, page_image)
        
        return bboxes
    
    def _merge_by_caption(self, bboxes, captions):
        """基于 caption 关联合并"""
        # 为每个 bbox 找到最近的 caption
        bbox_caption_map = {}
        for bbox in bboxes:
            nearest_caption = self._find_nearest_caption(bbox, captions)
            if nearest_caption:
                bbox_caption_map[bbox.id] = nearest_caption.id
        
        # 合并共享同一 caption 的 bbox
        groups = self._group_by_caption(bbox_caption_map)
        merged = []
        for group in groups:
            if len(group) > 1:
                # 检查是否是子图（如 Figure 1(a), (b), (c)）
                if self._is_subfigure_group(group, captions):
                    merged.append(self._merge_group(group))
                else:
                    merged.extend(group)
            else:
                merged.extend(group)
        return merged
    
    def _merge_by_visual(self, bboxes, page_image):
        """基于视觉连续性合并"""
        # 检测连接线
        connections = self._detect_connections(bboxes, page_image)
        
        # 合并有连接的 bbox
        graph = self._build_connection_graph(bboxes, connections)
        components = self._find_connected_components(graph)
        
        merged = []
        for component in components:
            if len(component) > 1:
                # 验证是否应该合并（避免误合并）
                if self._should_merge_component(component, page_image):
                    merged.append(self._merge_group(component))
                else:
                    merged.extend(component)
            else:
                merged.extend(component)
        return merged
    
    def _filter_noise(self, bboxes, page_image):
        """过滤噪声检测（如单独的箭头）"""
        filtered = []
        for bbox in bboxes:
            # 检查是否是箭头
            if self._is_arrow(bbox, page_image):
                logger.debug(f"Filtered arrow bbox: {bbox}")
                continue
            
            # 检查是否是其他噪声
            if self._is_noise(bbox, page_image):
                logger.debug(f"Filtered noise bbox: {bbox}")
                continue
            
            filtered.append(bbox)
        return filtered
    
    def _is_arrow(self, bbox, page_image):
        """判断是否是箭头"""
        crop = self._crop_bbox(page_image, bbox)
        
        # 特征1：长宽比极端（箭头通常很细长）
        aspect_ratio = bbox.width / bbox.height
        if aspect_ratio > 5 or aspect_ratio < 0.2:
            # 特征2：内容稀疏（箭头像素少）
            ink_ratio = self._compute_ink_ratio(crop)
            if ink_ratio < 0.05:
                return True
        
        # 特征3：使用简单的形状检测
        # 检测是否有三角形（箭头头部）
        if self._has_triangle_shape(crop):
            return True
        
        return False
```

### 2.3 表格识别增强

#### 2.3.1 方案选择

**主方案：Table Transformer**
- Microsoft 开源，专门用于表格检测和结构识别
- 两阶段：检测表格位置 + 识别表格结构
- 准确率高，特别是对无边框表格

**辅助方案：PaddleOCR 表格识别**
- 轻量级，速度快
- 支持中英文表格
- 作为 Table Transformer 的补充

**保留方案：pdfplumber + camelot**
- 作为降级方案

#### 2.3.2 实现设计

```python
class TableExtractor:
    """统一的表格提取接口"""
    
    def __init__(self, strategy="auto"):
        self.detectors = []
        self._init_detectors(strategy)
    
    def extract(self, pdf_path, ingest_data, capabilities):
        """多策略融合提取"""
        all_tables = []
        
        # 策略1：使用布局检测找到表格区域
        layout_tables = self._extract_from_layout(ingest_data)
        all_tables.extend(layout_tables)
        
        # 策略2：使用 Table Transformer
        if capabilities.get("table_transformer"):
            tt_tables = self._extract_with_table_transformer(pdf_path, ingest_data)
            all_tables.extend(tt_tables)
        
        # 策略3：使用 pdfplumber
        pdfplumber_tables = self._extract_with_pdfplumber(pdf_path, ingest_data)
        all_tables.extend(pdfplumber_tables)
        
        # 去重和合并
        tables = self._deduplicate_tables(all_tables)
        
        return tables
    
    def _deduplicate_tables(self, tables):
        """去重：移除重复检测的表格"""
        unique = []
        for table in tables:
            is_duplicate = False
            for existing in unique:
                if (table.page_index == existing.page_index and
                    self._bbox_iou(table.bbox, existing.bbox) > 0.7):
                    # 保留置信度更高的
                    if table.confidence > existing.confidence:
                        unique.remove(existing)
                        unique.append(table)
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(table)
        return unique
```

### 2.4 Caption 关联优化

#### 2.4.1 改进策略

当前 caption 关联仅基于距离，需要改进：

1. **方向优先级**：优先查找 bbox 下方的 caption
2. **编号匹配**：匹配 "Figure 1" 与图表的对应关系
3. **多行 caption**：正确识别跨多行的 caption
4. **子图编号**：识别 (a), (b), (c) 等子图标记

#### 2.4.2 实现设计

```python
class CaptionAligner:
    """Caption 对齐器"""
    
    def align(self, items, ingest_data):
        """为每个 item 找到对应的 caption"""
        for item in items:
            page_idx = item["page_index"]
            bbox = item["bbox"]
            text_lines = ingest_data["page_text_lines"][page_idx]
            
            # 查找候选 caption
            candidates = self._find_caption_candidates(
                bbox, text_lines, item["type"]
            )
            
            # 评分和选择
            best_caption = self._select_best_caption(
                bbox, candidates, item["type"]
            )
            
            if best_caption:
                item["caption"] = best_caption["text"]
                item["caption_bbox"] = best_caption["bbox"]
                
                # 提取 snippet（caption 上下文）
                item["evidence_snippet"] = self._extract_snippet(
                    best_caption, text_lines
                )
                
                # 提取子图编号（如果有）
                item["subfigure_label"] = self._extract_subfigure_label(
                    best_caption["text"]
                )
        
        return items
    
    def _find_caption_candidates(self, bbox, text_lines, item_type):
        """查找候选 caption"""
        candidates = []
        keywords = ["Figure", "Fig.", "Table", "Tab."] if item_type == "figure" else ["Table", "Tab."]
        
        for line in text_lines:
            # 检查是否包含关键词
            if not any(kw in line["text"] for kw in keywords):
                continue
            
            # 计算距离和方向
            distance = self._compute_distance(bbox, line["bbox"])
            direction = self._compute_direction(bbox, line["bbox"])
            
            # 方向惩罚：caption 通常在图表下方
            penalty = 0
            if direction == "above":
                penalty = 100
            elif direction == "side":
                penalty = 50
            
            score = distance + penalty
            
            candidates.append({
                "text": line["text"],
                "bbox": line["bbox"],
                "score": score,
                "direction": direction
            })
        
        return sorted(candidates, key=lambda x: x["score"])
    
    def _extract_subfigure_label(self, caption_text):
        """提取子图标记"""
        # 匹配 (a), (b), (c) 或 a), b), c)
        import re
        match = re.search(r'\(([a-z])\)|\b([a-z])\)', caption_text)
        if match:
            return match.group(1) or match.group(2)
        return None
```

### 2.5 质量评估和过滤

#### 2.5.1 质量指标

为每个检测结果计算质量分数：

1. **检测置信度**：模型输出的置信度
2. **内容完整性**：是否包含足够的内容（非空白）
3. **Caption 匹配度**：是否有对应的 caption
4. **尺寸合理性**：尺寸是否在合理范围内
5. **位置合理性**：位置是否合理（不在页边缘）

#### 2.5.2 实现设计

```python
class QualityAssessor:
    """质量评估器"""
    
    def assess(self, item, page_image, captions):
        """评估 item 质量"""
        scores = {}
        
        # 1. 检测置信度
        scores["detection_conf"] = item.get("detection_score", 0.5)
        
        # 2. 内容完整性
        crop = self._crop_bbox(page_image, item["bbox"])
        scores["content_completeness"] = self._assess_content(crop)
        
        # 3. Caption 匹配度
        scores["caption_match"] = self._assess_caption_match(item, captions)
        
        # 4. 尺寸合理性
        scores["size_reasonableness"] = self._assess_size(item["bbox"], page_image.shape)
        
        # 5. 位置合理性
        scores["position_reasonableness"] = self._assess_position(item["bbox"], page_image.shape)
        
        # 综合评分
        weights = {
            "detection_conf": 0.3,
            "content_completeness": 0.3,
            "caption_match": 0.2,
            "size_reasonableness": 0.1,
            "position_reasonableness": 0.1
        }
        
        total_score = sum(scores[k] * weights[k] for k in scores)
        
        item["quality_score"] = total_score
        item["quality_details"] = scores
        
        return total_score
    
    def filter_low_quality(self, items, threshold=0.5):
        """过滤低质量检测"""
        filtered = []
        for item in items:
            if item.get("quality_score", 0) >= threshold:
                filtered.append(item)
            else:
                logger.debug(f"Filtered low quality item: {item['item_id']}, score: {item.get('quality_score')}")
        return filtered
```

## 3. 模块设计

### 3.1 新增模块

#### 3.1.1 layout_detect_v2.py
- 统一的布局检测接口
- 支持多种检测器（DocLayout-YOLO, Surya, PubLayNet）
- 自动降级机制

#### 3.1.2 bbox_merger.py
- 智能边界框合并
- 多维度判断（空间、语义、视觉）
- 噪声过滤（箭头、连接线等）

#### 3.1.3 table_extract_v2.py
- 增强的表格提取
- 支持 Table Transformer
- 多策略融合和去重

#### 3.1.4 quality_assess.py
- 质量评估和过滤
- 多维度质量指标
- 可配置的阈值

### 3.2 修改模块

#### 3.2.1 figure_extract.py
- 集成新的布局检测器
- 使用智能边界框合并
- 添加质量评估

#### 3.2.2 caption_align.py
- 改进 caption 查找算法
- 支持子图编号识别
- 更好的多行 caption 处理

#### 3.2.3 config.py
- 添加新模型的配置项
- 添加合并策略的参数
- 添加质量阈值配置

## 4. 配置设计

### 4.1 新增配置项

```json
{
  "layout_detector": {
    "strategy": "auto",  // auto, doclayout, surya, publaynet
    "doclayout": {
      "model_path": "path/to/doclayout_yolo.pt",
      "conf_threshold": 0.25,
      "iou_threshold": 0.45,
      "device": "cuda"
    },
    "surya": {
      "model_name": "vikp/surya_layout",
      "batch_size": 1
    }
  },
  
  "bbox_merger": {
    "enable_semantic_merge": true,
    "enable_visual_merge": true,
    "enable_noise_filter": true,
    "arrow_filter_threshold": 0.05,
    "connection_detection_threshold": 0.3
  },
  
  "table_extractor": {
    "strategy": "auto",  // auto, table_transformer, pdfplumber, camelot
    "table_transformer": {
      "detection_model": "microsoft/table-transformer-detection",
      "structure_model": "microsoft/table-transformer-structure-recognition",
      "conf_threshold": 0.7
    },
    "enable_deduplication": true,
    "dedup_iou_threshold": 0.7
  },
  
  "quality_assessment": {
    "enable": true,
    "min_quality_score": 0.5,
    "weights": {
      "detection_conf": 0.3,
      "content_completeness": 0.3,
      "caption_match": 0.2,
      "size_reasonableness": 0.1,
      "position_reasonableness": 0.1
    }
  }
}
```

## 5. 实现计划

### 5.1 阶段 1：修复模型加载问题（P0）
- 修复 PubLayNet 权重文件路径问题
- 添加更好的错误处理和日志
- 验证模型可以正常加载

### 5.2 阶段 2：集成 DocLayout-YOLO（P0）
- 安装和配置 DocLayout-YOLO
- 实现 DocLayoutYOLODetector
- 实现 LayoutDetector 统一接口
- 测试检测效果

### 5.3 阶段 3：实现智能边界框合并（P0）
- 实现 SmartBBoxMerger
- 实现语义合并（基于 caption）
- 实现视觉合并（连接线检测）
- 实现噪声过滤（箭头过滤）

### 5.4 阶段 4：增强表格识别（P1）
- 集成 Table Transformer
- 实现多策略融合
- 实现去重逻辑

### 5.5 阶段 5：优化 Caption 关联（P1）
- 改进 caption 查找算法
- 实现子图编号识别
- 改进多行 caption 处理

### 5.6 阶段 6：添加质量评估（P1）
- 实现 QualityAssessor
- 集成到提取流程
- 调优阈值参数

### 5.7 阶段 7：测试和调优（P1）
- 在样本 PDF 上测试
- 调优参数
- 修复发现的问题

## 6. 测试策略

### 6.1 单元测试

为每个新模块编写单元测试：
- `test_layout_detect_v2.py`
- `test_bbox_merger.py`
- `test_table_extract_v2.py`
- `test_quality_assess.py`

### 6.2 集成测试

端到端测试：
- 使用样本 PDF 测试完整流程
- 验证输出格式正确
- 验证准确率指标

### 6.3 性能测试

- 测试处理速度
- 测试内存占用
- 测试 GPU 利用率

### 6.4 回归测试

- 确保新版本不会降低原有功能的准确率
- 确保输出格式向后兼容

## 7. 风险和缓解

### 7.1 风险：DocLayout-YOLO 可能不适合所有论文
**缓解**：实现多策略融合，保留 PubLayNet 作为降级方案

### 7.2 风险：智能合并可能引入新的错误
**缓解**：每个合并策略都可以单独开关，逐步启用和调优

### 7.3 风险：Table Transformer 可能很慢
**缓解**：实现批处理和缓存，提供轻量级降级方案

### 7.4 风险：参数调优可能很耗时
**缓解**：提供合理的默认值，支持配置文件快速调整

## 8. 性能优化

### 8.1 缓存策略
- 缓存布局检测结果
- 缓存模型加载
- 缓存图像预处理结果

### 8.2 批处理
- 批量处理多页
- 批量运行模型推理

### 8.3 并行处理
- 多进程处理不同页面
- GPU 并行推理

## 9. 可观测性

### 9.1 日志设计
- 详细记录每个步骤的输入输出
- 记录合并决策的原因
- 记录质量评分的细节

### 9.2 调试工具
- 可视化检测结果
- 可视化合并过程
- 可视化质量评分

### 9.3 指标收集
- 统计检测数量
- 统计合并次数
- 统计过滤次数
- 统计质量分布

## 10. 文档更新

### 10.1 用户文档
- 更新 README.md，说明新功能
- 更新安装文档，说明新依赖
- 更新配置文档，说明新参数

### 10.2 开发文档
- 更新 design.md
- 添加 architecture.md
- 添加 troubleshooting.md

## 11. 验收标准

### 11.1 功能验收
- [ ] 模型加载成功率 100%
- [ ] 支持 DocLayout-YOLO 检测
- [ ] 支持智能边界框合并
- [ ] 支持箭头过滤
- [ ] 支持 Table Transformer
- [ ] 支持质量评估

### 11.2 性能验收
- [ ] 图表检测 F1 > 0.90
- [ ] 表格检测 F1 > 0.85
- [ ] 边界框平均 IoU > 0.85
- [ ] 过度分割率 < 5%
- [ ] 错误合并率 < 5%

### 11.3 稳定性验收
- [ ] 无未处理异常
- [ ] 降级机制正常工作
- [ ] 错误信息清晰

### 11.4 可用性验收
- [ ] 用户输入仍然简单（单个 PDF）
- [ ] 输出格式保持兼容
- [ ] 配置文件易于理解

## 12. 后续演进方向

### 12.1 短期（1-2 周）
- 支持更多文档类型（会议论文、专利等）
- 优化处理速度
- 添加更多可视化调试工具

### 12.2 中期（1-2 月）
- 支持多曲线分离
- 支持坐标轴 OCR
- 支持公式识别

### 12.3 长期（3-6 月）
- 支持矢量图直接提取数据
- 支持交互式标注和修正
- 支持模型微调和自定义训练

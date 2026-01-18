# Design Document: FigTabMiner v1.7 Accuracy Fix

## Overview

本设计文档描述了FigTabMiner v1.7准确度修复功能的技术设计。该功能旨在系统性解决v1.5和v1.6版本引入DocLayout-YOLO和Table Transformer模型后出现的严重准确度下降问题。

### 问题根源分析

通过分析现有代码和用户反馈，识别出以下根本问题：

1. **检测器融合策略不当**：当前系统简单地将多个检测器的结果合并，没有考虑各检测器的特性和可靠性，导致：
   - 假阳性增加：Table Transformer将文字区域误识别为表格
   - 检测冲突：不同检测器对同一区域产生不同的检测结果，缺乏有效的仲裁机制

2. **边界框合并逻辑缺陷**：SmartBBoxMerger虽然有多阶段合并策略，但存在：
   - 过度合并：将相邻但独立的图表错误合并
   - 合并不足：将单个复杂图表错误分割
   - 阈值不合理：IoU和距离阈值对不同类型的图表不适用

3. **置信度阈值设置不当**：
   - DocLayout-YOLO默认阈值0.25过低，导致大量假阳性
   - Table Transformer默认阈值0.7可能过高，导致漏检
   - 缺乏动态阈值调整机制

4. **图表分类器局限性**：
   - 支持的图表类型过少（仅9种）
   - 视觉特征提取不够精确
   - 对复杂图表（如拼接图、流程图）识别能力弱

5. **质量评估不充分**：
   - 质量评估维度单一
   - 缺乏对检测结果的有效过滤
   - 没有利用质量分数指导后续处理

### 设计目标

1. 实现智能检测器融合策略，根据检测器特性和场景动态调整权重
2. 优化边界框合并逻辑，减少错误合并和错误分割
3. 引入自适应置信度阈值机制
4. 扩展图表分类器，支持更多类型和更精确的分类
5. 增强质量评估和过滤机制
6. 建立完整的准确度评估框架
7. 保持向后兼容性，不破坏现有功能


## Architecture

### 系统架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                     PDF Document Input                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Multi-Detector Detection Layer                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  DocLayout   │  │    Table     │  │   Legacy     │     │
│  │    YOLO      │  │ Transformer  │  │  Detectron2  │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            │                                 │
└────────────────────────────┼─────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│           Intelligent Detection Fusion Module               │
│  • Detector Weighting & Prioritization                      │
│  • Conflict Resolution                                       │
│  • Adaptive Confidence Thresholding                          │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│          Enhanced BBox Merger & Refinement                   │
│  • Context-Aware Merging                                     │
│  • Type-Specific Strategies                                  │
│  • Boundary Refinement                                       │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              Quality Assessment & Filtering                  │
│  • Multi-Dimensional Quality Scoring                         │
│  • Confidence-Based Filtering                                │
│  • Anomaly Detection                                         │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│           Enhanced Chart Classification                      │
│  • Extended Type Support (15+ types)                         │
│  • Multi-Feature Analysis                                    │
│  • Hierarchical Classification                               │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              Enhanced AI Analysis                            │
│  • Robust Subtype Extraction                                 │
│  • Scientific Metadata Extraction                            │
│  • Error-Tolerant Processing                                 │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Output & Evaluation                        │
│  • Structured Results                                        │
│  • Accuracy Metrics                                          │
│  • Visualization Tools                                       │
└─────────────────────────────────────────────────────────────┘
```

### 关键设计决策

1. **分层架构**：将检测、融合、合并、质量评估、分类和分析分离为独立模块，便于测试和优化
2. **插件化检测器**：支持动态启用/禁用检测器，便于A/B测试和性能优化
3. **配置驱动**：所有关键参数通过配置文件控制，支持不同场景的优化
4. **质量优先**：在每个阶段引入质量评估，及早过滤低质量结果
5. **向后兼容**：保留旧版本接口，通过配置开关控制新旧实现


## Components and Interfaces

### 1. Intelligent Detection Fusion Module

**职责**：融合多个检测器的结果，解决冲突，生成统一的检测结果

**接口**：
```python
class IntelligentDetectionFusion:
    def __init__(self, config: Dict):
        """
        初始化融合模块
        
        Args:
            config: 配置字典，包含：
                - detector_weights: 各检测器的权重
                - fusion_strategy: 融合策略 ('weighted_nms', 'voting', 'cascade')
                - confidence_thresholds: 各检测器的置信度阈值
                - enable_adaptive_threshold: 是否启用自适应阈值
        """
        pass
    
    def fuse_detections(
        self,
        detections_by_detector: Dict[str, List[Detection]],
        page_image: np.ndarray,
        page_metadata: Dict
    ) -> List[Detection]:
        """
        融合多个检测器的结果
        
        Args:
            detections_by_detector: 各检测器的检测结果
            page_image: 页面图像
            page_metadata: 页面元数据
            
        Returns:
            融合后的检测结果列表
        """
        pass
    
    def resolve_conflicts(
        self,
        detections: List[Detection],
        conflict_threshold: float = 0.5
    ) -> List[Detection]:
        """
        解决检测冲突
        
        Args:
            detections: 检测结果列表
            conflict_threshold: IoU阈值，超过此值认为是冲突
            
        Returns:
            解决冲突后的检测结果
        """
        pass
    
    def adaptive_threshold(
        self,
        detections: List[Detection],
        target_precision: float = 0.95
    ) -> List[Detection]:
        """
        自适应调整置信度阈值
        
        Args:
            detections: 检测结果列表
            target_precision: 目标精确率
            
        Returns:
            过滤后的检测结果
        """
        pass
```

**核心算法**：

1. **加权NMS (Weighted Non-Maximum Suppression)**：
   - 为每个检测器分配权重（基于历史性能）
   - 计算加权置信度：`weighted_conf = detector_weight * detection_conf`
   - 使用加权置信度进行NMS

2. **冲突仲裁策略**：
   - 当多个检测器对同一区域产生不同类型的检测时：
     - 优先级：figure > table > text（基于误识别代价）
     - 如果类型相同但边界框不同，选择置信度最高的
     - 如果置信度相近，选择来自更可靠检测器的结果

3. **自适应阈值**：
   - 基于检测结果的置信度分布动态调整阈值
   - 使用Otsu方法或K-means聚类找到最优分割点
   - 考虑目标精确率和召回率的平衡

### 2. Enhanced BBox Merger

**职责**：智能合并和分割边界框，处理复杂图表

**接口**：
```python
class EnhancedBBoxMerger:
    def __init__(self, config: Dict):
        """
        初始化增强型边界框合并器
        
        Args:
            config: 配置字典，包含：
                - merge_strategies: 合并策略列表
                - type_specific_params: 各类型图表的特定参数
                - enable_boundary_refinement: 是否启用边界精炼
        """
        pass
    
    def merge_with_context(
        self,
        detections: List[Detection],
        page_image: np.ndarray,
        captions: List[Caption],
        page_layout: Dict
    ) -> List[Detection]:
        """
        基于上下文的智能合并
        
        Args:
            detections: 检测结果列表
            page_image: 页面图像
            captions: 标题列表
            page_layout: 页面布局信息
            
        Returns:
            合并后的检测结果
        """
        pass
    
    def split_complex_figures(
        self,
        detection: Detection,
        page_image: np.ndarray
    ) -> List[Detection]:
        """
        分割错误合并的复杂图表
        
        Args:
            detection: 检测结果
            page_image: 页面图像
            
        Returns:
            分割后的检测结果列表（如果不需要分割则返回原检测）
        """
        pass
    
    def refine_boundaries(
        self,
        detection: Detection,
        page_image: np.ndarray
    ) -> Detection:
        """
        精炼边界框
        
        Args:
            detection: 检测结果
            page_image: 页面图像
            
        Returns:
            边界精炼后的检测结果
        """
        pass
```

**核心算法**：

1. **上下文感知合并**：
   - 考虑标题关联：共享标题的子图应该合并
   - 考虑布局结构：在同一列/行的相似图表可能是子图
   - 考虑视觉连续性：有连接线或箭头的图表应该合并

2. **类型特定策略**：
   - 流程图：检测节点和连接线，合并所有相关元素
   - 拼接图：检测子图网格结构，合并整个网格
   - 表格：避免与周围文字合并
   - 单图：避免与其他图表合并

3. **边界精炼**：
   - 使用GrabCut或类似算法精确分割前景
   - 移除边界的空白区域
   - 确保包含所有相关元素（标签、轴、图例等）

### 3. Enhanced Chart Classifier

**职责**：精确分类图表类型，支持更多类型

**接口**：
```python
class EnhancedChartClassifier:
    # 扩展的图表类型
    CHART_TYPES = [
        'bar_chart', 'pie_chart', 'line_plot', 'scatter_plot',
        'heatmap', 'box_plot', 'violin_plot', 'histogram',
        'microscopy_sem', 'microscopy_tem', 'microscopy_optical',
        'spectrum', 'diagram_flowchart', 'diagram_schematic',
        'photo', 'unknown'
    ]
    
    def __init__(self, config: Dict):
        """
        初始化增强型图表分类器
        
        Args:
            config: 配置字典
        """
        pass
    
    def classify(
        self,
        image_path: str,
        caption: str = "",
        snippet: str = "",
        ocr_text: str = "",
        detection_metadata: Dict = None
    ) -> ClassificationResult:
        """
        分类图表类型
        
        Args:
            image_path: 图表图像路径
            caption: 标题文本
            snippet: 片段文本
            ocr_text: OCR文本
            detection_metadata: 检测元数据
            
        Returns:
            ClassificationResult包含：
                - chart_type: 图表类型
                - confidence: 置信度
                - sub_type: 子类型（可选）
                - matched_keywords: 匹配的关键词
                - visual_features: 视觉特征
                - debug_info: 调试信息
        """
        pass
    
    def classify_hierarchical(
        self,
        image_path: str,
        **kwargs
    ) -> HierarchicalClassification:
        """
        层次化分类（先分大类，再分子类）
        
        Returns:
            HierarchicalClassification包含：
                - main_category: 主类别 ('chart', 'microscopy', 'diagram', 'photo')
                - sub_category: 子类别
                - confidence_by_level: 各层级的置信度
        """
        pass
```

**核心算法**：

1. **多特征融合分类**：
   - 关键词特征：从标题、片段、OCR文本提取
   - 视觉特征：形状、颜色、纹理、布局
   - 上下文特征：页面位置、周围元素、文档类型

2. **层次化分类**：
   - 第一层：区分chart、microscopy、diagram、photo
   - 第二层：在每个大类中细分子类型
   - 提高分类准确率，减少混淆

3. **置信度校准**：
   - 使用Platt scaling或温度缩放校准置信度
   - 确保置信度反映真实的分类准确率


### 4. Enhanced Quality Assessor

**职责**：多维度评估检测质量，过滤低质量结果

**接口**：
```python
class EnhancedQualityAssessor:
    def __init__(self, config: Dict):
        """
        初始化增强型质量评估器
        
        Args:
            config: 配置字典
        """
        pass
    
    def assess_comprehensive(
        self,
        detection: Detection,
        page_image: np.ndarray,
        captions: List[Caption],
        page_layout: Dict
    ) -> QualityScore:
        """
        综合质量评估
        
        Args:
            detection: 检测结果
            page_image: 页面图像
            captions: 标题列表
            page_layout: 页面布局
            
        Returns:
            QualityScore包含：
                - overall_score: 总体质量分数 (0-1)
                - dimension_scores: 各维度分数
                - issues: 识别的问题列表
                - recommendations: 改进建议
        """
        pass
    
    def detect_anomalies(
        self,
        detections: List[Detection],
        page_image: np.ndarray
    ) -> List[Anomaly]:
        """
        检测异常检测结果
        
        Args:
            detections: 检测结果列表
            page_image: 页面图像
            
        Returns:
            异常列表
        """
        pass
    
    def filter_by_quality(
        self,
        detections: List[Detection],
        min_score: float = 0.5
    ) -> Tuple[List[Detection], List[Detection]]:
        """
        按质量过滤
        
        Args:
            detections: 检测结果列表
            min_score: 最小质量分数
            
        Returns:
            (通过的检测, 被过滤的检测)
        """
        pass
```

**质量评估维度**：

1. **检测置信度** (30%)：来自检测器的原始置信度
2. **内容完整性** (25%)：边界框是否包含完整内容
3. **边界精确度** (20%)：边界框是否紧密贴合内容
4. **标题匹配度** (15%)：是否有匹配的标题
5. **位置合理性** (10%)：位置是否合理（不在页边缘等）

**异常检测**：
- 过大的边界框（>90%页面）
- 过小的边界框（<0.1%页面）
- 极端纵横比（>10或<0.1）
- 内容过于稀疏（ink ratio < 0.5%）
- 与文字区域高度重叠

### 5. Enhanced AI Analyzer

**职责**：提取图表的详细信息和元数据

**接口**：
```python
class EnhancedAIAnalyzer:
    def __init__(self, config: Dict):
        """
        初始化增强型AI分析器
        
        Args:
            config: 配置字典
        """
        pass
    
    def analyze_robust(
        self,
        image_path: str,
        chart_type: str,
        caption: str = "",
        snippet: str = "",
        ocr_text: str = ""
    ) -> AnalysisResult:
        """
        鲁棒的图表分析
        
        Args:
            image_path: 图表图像路径
            chart_type: 图表类型
            caption: 标题
            snippet: 片段
            ocr_text: OCR文本
            
        Returns:
            AnalysisResult包含：
                - subtype: 子类型
                - subtype_confidence: 子类型置信度
                - conditions: 实验条件列表
                - materials: 材料候选列表
                - keywords: 关键词列表
                - method: 使用的方法
                - debug: 调试信息
        """
        pass
    
    def validate_input(
        self,
        image_path: str,
        chart_type: str
    ) -> ValidationResult:
        """
        验证输入数据质量
        
        Args:
            image_path: 图表图像路径
            chart_type: 图表类型
            
        Returns:
            ValidationResult包含：
                - is_valid: 是否有效
                - issues: 问题列表
                - suggestions: 建议
        """
        pass
    
    def extract_scientific_metadata(
        self,
        image_path: str,
        chart_type: str,
        ocr_text: str
    ) -> ScientificMetadata:
        """
        提取科学元数据
        
        Args:
            image_path: 图表图像路径
            chart_type: 图表类型
            ocr_text: OCR文本
            
        Returns:
            ScientificMetadata包含：
                - experimental_conditions: 实验条件
                - materials: 材料
                - measurements: 测量值
                - units: 单位
        """
        pass
```

**核心改进**：

1. **输入验证**：
   - 检查图像质量（分辨率、清晰度）
   - 检查图表类型的合理性
   - 处理缺失或错误的输入

2. **鲁棒的子类型识别**：
   - 使用多种特征（OCR、视觉、上下文）
   - 设置最小置信度阈值（0.6）
   - 提供详细的评分和推理过程

3. **科学元数据提取**：
   - 识别实验条件（温度、压力、时间等）
   - 提取材料名称和化学式
   - 识别测量值和单位

### 6. Accuracy Evaluation Framework

**职责**：系统化评估检测准确度

**接口**：
```python
class AccuracyEvaluator:
    def __init__(self, config: Dict):
        """
        初始化准确度评估器
        
        Args:
            config: 配置字典
        """
        pass
    
    def evaluate(
        self,
        predictions: List[Detection],
        ground_truth: List[Detection],
        iou_threshold: float = 0.5
    ) -> EvaluationMetrics:
        """
        评估检测准确度
        
        Args:
            predictions: 预测结果
            ground_truth: 真实标注
            iou_threshold: IoU阈值
            
        Returns:
            EvaluationMetrics包含：
                - precision: 精确率
                - recall: 召回率
                - f1_score: F1分数
                - mean_iou: 平均IoU
                - false_positives: 假阳性列表
                - false_negatives: 假阴性列表
                - true_positives: 真阳性列表
        """
        pass
    
    def evaluate_by_type(
        self,
        predictions: List[Detection],
        ground_truth: List[Detection]
    ) -> Dict[str, EvaluationMetrics]:
        """
        按类型分组评估
        
        Returns:
            各类型的评估指标字典
        """
        pass
    
    def generate_report(
        self,
        metrics: EvaluationMetrics,
        output_path: str
    ) -> None:
        """
        生成评估报告
        
        Args:
            metrics: 评估指标
            output_path: 输出路径
        """
        pass
    
    def visualize_results(
        self,
        predictions: List[Detection],
        ground_truth: List[Detection],
        page_image: np.ndarray,
        output_path: str
    ) -> None:
        """
        可视化检测结果
        
        Args:
            predictions: 预测结果
            ground_truth: 真实标注
            page_image: 页面图像
            output_path: 输出路径
        """
        pass
```

**评估指标**：

1. **基础指标**：
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
   - F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
   - Mean IoU = Σ IoU(pred, gt) / N

2. **分组统计**：
   - 按图表类型（figure、table）
   - 按页面类型（单栏、双栏、复杂布局）
   - 按文档类型（学术论文、技术报告、专利）

3. **错误分析**：
   - 假阳性分析：错误识别的类型和原因
   - 假阴性分析：漏检的类型和原因
   - 边界框偏差分析：IoU分布


## Data Models

### Detection

```python
@dataclass
class Detection:
    """检测结果数据模型"""
    bbox: List[float]  # [x0, y0, x1, y1]
    type: str  # 'figure' or 'table'
    score: float  # 置信度 (0-1)
    detector: str  # 检测器名称
    class_id: int  # 原始类别ID
    label: str  # 语义标签
    
    # 可选字段
    merged_from: Optional[int] = None  # 合并自多少个检测
    quality_score: Optional[float] = None  # 质量分数
    quality_details: Optional[Dict] = None  # 质量详情
    chart_type: Optional[str] = None  # 图表类型
    chart_confidence: Optional[float] = None  # 图表类型置信度
    
    def area(self) -> float:
        """计算面积"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    
    def iou(self, other: 'Detection') -> float:
        """计算与另一个检测的IoU"""
        return bbox_utils.bbox_iou(self.bbox, other.bbox)
```

### ClassificationResult

```python
@dataclass
class ClassificationResult:
    """图表分类结果"""
    chart_type: str  # 图表类型
    confidence: float  # 置信度
    sub_type: Optional[str] = None  # 子类型
    matched_keywords: List[str] = field(default_factory=list)  # 匹配的关键词
    visual_features: Optional[Dict] = None  # 视觉特征
    debug_info: Optional[Dict] = None  # 调试信息
```

### QualityScore

```python
@dataclass
class QualityScore:
    """质量评分"""
    overall_score: float  # 总体分数 (0-1)
    dimension_scores: Dict[str, float]  # 各维度分数
    issues: List[str] = field(default_factory=list)  # 问题列表
    recommendations: List[str] = field(default_factory=list)  # 建议列表
```

### AnalysisResult

```python
@dataclass
class AnalysisResult:
    """AI分析结果"""
    subtype: str  # 子类型
    subtype_confidence: float  # 子类型置信度
    conditions: List[str] = field(default_factory=list)  # 实验条件
    materials: List[str] = field(default_factory=list)  # 材料候选
    keywords: List[str] = field(default_factory=list)  # 关键词
    method: str = "enhanced_analyzer_v1.7"  # 方法
    debug: Optional[Dict] = None  # 调试信息
```

### EvaluationMetrics

```python
@dataclass
class EvaluationMetrics:
    """评估指标"""
    precision: float  # 精确率
    recall: float  # 召回率
    f1_score: float  # F1分数
    mean_iou: float  # 平均IoU
    
    true_positives: List[Tuple[Detection, Detection]]  # (pred, gt)
    false_positives: List[Detection]  # 假阳性
    false_negatives: List[Detection]  # 假阴性
    
    # 统计信息
    total_predictions: int
    total_ground_truth: int
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'mean_iou': self.mean_iou,
            'tp_count': len(self.true_positives),
            'fp_count': len(self.false_positives),
            'fn_count': len(self.false_negatives),
            'total_predictions': self.total_predictions,
            'total_ground_truth': self.total_ground_truth
        }
```

### Configuration Schema

```python
@dataclass
class DetectionConfig:
    """检测配置"""
    # 检测器配置
    enable_doclayout: bool = True
    enable_table_transformer: bool = True
    enable_legacy_detectron: bool = False
    
    # 置信度阈值
    doclayout_confidence: float = 0.35  # 提高阈值减少假阳性
    table_transformer_confidence: float = 0.75
    legacy_confidence: float = 0.5
    
    # 融合配置
    fusion_strategy: str = 'weighted_nms'  # 'weighted_nms', 'voting', 'cascade'
    detector_weights: Dict[str, float] = field(default_factory=lambda: {
        'doclayout': 0.4,
        'table_transformer': 0.4,
        'legacy': 0.2
    })
    
    # 合并配置
    merge_iou_threshold: float = 0.3
    merge_distance_threshold: float = 50
    enable_context_aware_merge: bool = True
    enable_boundary_refinement: bool = True
    
    # 质量配置
    min_quality_score: float = 0.4
    enable_anomaly_detection: bool = True
    
    # 分类配置
    enable_hierarchical_classification: bool = True
    min_classification_confidence: float = 0.5
    
    # 性能配置
    enable_parallel_detection: bool = True
    enable_model_caching: bool = True
```


## Correctness Properties

*属性（Property）是一个特征或行为，应该在系统的所有有效执行中保持为真——本质上是关于系统应该做什么的形式化陈述。属性是人类可读规范和机器可验证正确性保证之间的桥梁。*

### Property 1: 假阳性率控制

*对于任何*包含图表的PDF页面和标注数据，当检测系统处理该页面时，假阳性率（错误识别数量/总检测数量）应该小于5%

**Validates: Requirements 1.1**

### Property 2: 文字区域不被误识别

*对于任何*包含文字内容（正文、标题、作者、尾注、脚注、参考文献）的页面区域，检测系统不应该将这些区域标记为图表

**Validates: Requirements 1.2**

### Property 3: 独立图表不被错误合并

*对于任何*包含多个独立图表的页面，检测系统应该为每个图表生成独立的检测结果，且这些检测结果的边界框之间的IoU应该小于0.1（表示不重叠）

**Validates: Requirements 1.3**

### Property 4: 复杂图表不被错误分割

*对于任何*包含复杂元素（箭头、复杂曲线、数字文字标识）的单个图表，检测系统应该生成恰好一个检测结果，而不是多个分割的检测

**Validates: Requirements 1.4**

### Property 5: 边界框精确度

*对于任何*检测到的图表，其边界框与真实标注边界框的IoU值应该大于0.85

**Validates: Requirements 1.5, 3.1**

### Property 6: 召回率保证

*对于任何*包含图表的PDF页面和标注数据，检测系统的召回率（检测到的真实图表数/总真实图表数）应该大于95%

**Validates: Requirements 2.1**

### Property 7: 拼接图完整性

*对于任何*由多张小图拼接成的大图，检测系统应该生成一个包含所有子图的完整边界框，而不是为每个子图生成独立的检测

**Validates: Requirements 3.2**

### Property 8: 流程图完整性

*对于任何*流程图，检测系统应该生成一个包含所有节点和连接线的完整边界框

**Validates: Requirements 3.3**

### Property 9: Unknown类型比例控制

*对于任何*图表数据集，图表分类器将图表分类为unknown类型的比例应该小于10%

**Validates: Requirements 4.1**

### Property 10: 分类准确率

*对于任何*带有类型标注的图表数据集，图表分类器的分类准确率应该大于85%

**Validates: Requirements 4.2**

### Property 11: AI分析子类型置信度

*对于任何*图表，AI分析器提取的子类型置信度应该大于0.6，如果低于此阈值则应该标记为低置信度

**Validates: Requirements 5.1**

### Property 12: 科学图表元数据提取

*对于任何*科学图表（显微镜图像、光谱图等），AI分析器应该尝试提取实验条件、材料候选和关键词字段，即使某些字段为空

**Validates: Requirements 5.2**

### Property 13: 错误输入处理

*对于任何*无效或错误的输入（损坏的图像、错误的类型标签），AI分析器应该捕获错误并返回有意义的错误信息，而不是崩溃

**Validates: Requirements 5.5**

### Property 14: 质量分数范围

*对于任何*检测结果，质量评估器计算的质量分数应该在0到1的范围内

**Validates: Requirements 7.1**

### Property 15: 低质量检测过滤

*对于任何*质量分数低于配置阈值的检测结果，质量评估器应该将其标记为低质量或过滤掉

**Validates: Requirements 7.2**

### Property 16: 不合理边界框识别

*对于任何*边界框，如果其大小、纵横比或位置不合理（如面积<0.1%页面或>95%页面，纵横比>10或<0.1），质量评估器应该给予较低的质量分数

**Validates: Requirements 7.3**

### Property 17: 配置验证

*对于任何*配置文件，如果包含无效的参数值（如负数阈值、超出范围的权重），系统应该检测到错误并使用默认值或拒绝加载

**Validates: Requirements 9.2**

### Property 18: 错误捕获和记录

*对于任何*运行时错误，系统应该捕获异常并记录详细的错误信息（包括堆栈跟踪），而不是让程序崩溃

**Validates: Requirements 10.1**

### Property 19: 警告不中断处理

*对于任何*警告情况（如某个检测器失败、OCR文本缺失），系统应该记录警告但继续处理，不应该中断整个流程

**Validates: Requirements 10.2**

### Property 20: 向后兼容性

*对于任何*v1.1到v1.6版本支持的功能和API调用，v1.7版本应该继续支持，返回兼容的结果格式

**Validates: Requirements 12.1, 12.3**

### Property 21: 旧配置文件兼容性

*对于任何*v1.6或更早版本的配置文件，v1.7系统应该能够成功加载并解析，对于新增的配置项使用默认值

**Validates: Requirements 12.2**


## Error Handling

### 错误分类

1. **检测器错误**：
   - 模型加载失败
   - CUDA内存不足
   - 图像格式不支持
   - 处理：回退到其他检测器或CPU模式

2. **融合错误**：
   - 检测结果格式不一致
   - 冲突无法解决
   - 处理：记录警告，使用保守策略

3. **合并错误**：
   - 边界框无效
   - 图像访问失败
   - 处理：跳过该检测，记录错误

4. **分类错误**：
   - 图像损坏
   - OCR失败
   - 处理：返回unknown类型，低置信度

5. **质量评估错误**：
   - 图像访问失败
   - 计算异常
   - 处理：使用默认质量分数

6. **配置错误**：
   - 配置文件缺失
   - 参数无效
   - 处理：使用默认配置，记录警告

### 错误处理策略

```python
class ErrorHandler:
    """统一的错误处理器"""
    
    @staticmethod
    def handle_detector_error(
        detector_name: str,
        error: Exception,
        fallback_detectors: List[str]
    ) -> List[Detection]:
        """
        处理检测器错误
        
        策略：
        1. 记录详细错误信息
        2. 尝试使用fallback检测器
        3. 如果所有检测器都失败，返回空列表
        """
        logger.error(f"Detector {detector_name} failed: {error}")
        logger.debug("Traceback:", exc_info=True)
        
        for fallback in fallback_detectors:
            try:
                logger.info(f"Trying fallback detector: {fallback}")
                # 尝试fallback检测器
                return run_detector(fallback)
            except Exception as e:
                logger.warning(f"Fallback {fallback} also failed: {e}")
                continue
        
        logger.error("All detectors failed")
        return []
    
    @staticmethod
    def handle_classification_error(
        image_path: str,
        error: Exception
    ) -> ClassificationResult:
        """
        处理分类错误
        
        策略：
        1. 记录错误
        2. 返回unknown类型，低置信度
        3. 在debug信息中包含错误详情
        """
        logger.error(f"Classification failed for {image_path}: {error}")
        logger.debug("Traceback:", exc_info=True)
        
        return ClassificationResult(
            chart_type='unknown',
            confidence=0.0,
            debug_info={
                'error': str(error),
                'error_type': type(error).__name__
            }
        )
    
    @staticmethod
    def handle_config_error(
        config_path: str,
        error: Exception
    ) -> Dict:
        """
        处理配置错误
        
        策略：
        1. 记录错误
        2. 返回默认配置
        3. 警告用户
        """
        logger.error(f"Failed to load config from {config_path}: {error}")
        logger.warning("Using default configuration")
        
        return get_default_config()
```

### 日志记录规范

```python
# 日志级别使用规范
logger.debug("Detailed information for debugging")  # 调试信息
logger.info("Normal operation information")  # 正常操作
logger.warning("Warning: potential issue")  # 警告
logger.error("Error occurred, but recoverable")  # 可恢复错误
logger.critical("Critical error, system cannot continue")  # 严重错误

# 结构化日志示例
logger.info(
    "Detection completed",
    extra={
        'page_index': page_idx,
        'detector': detector_name,
        'detection_count': len(detections),
        'processing_time': elapsed_time
    }
)
```


## Testing Strategy

### 双重测试方法

本项目采用**单元测试**和**基于属性的测试（Property-Based Testing, PBT）**相结合的方法，以确保全面的测试覆盖：

- **单元测试**：验证特定示例、边缘情况和错误条件
- **属性测试**：通过随机生成的输入验证通用属性，确保系统在各种情况下的正确性

两种测试方法是互补的：单元测试捕获具体的错误，属性测试验证通用的正确性。

### 单元测试策略

单元测试应该专注于：

1. **特定示例**：
   - 已知的问题案例（如v1.6中的错误识别案例）
   - 典型的使用场景
   - 边界情况

2. **集成点**：
   - 检测器与融合模块的集成
   - 融合模块与合并模块的集成
   - 各模块与配置系统的集成

3. **边缘情况和错误条件**：
   - 空输入、无效输入
   - 极端参数值
   - 资源不足情况

**注意**：避免编写过多的单元测试。属性测试已经通过随机化覆盖了大量输入，单元测试应该专注于属性测试难以覆盖的特定场景。

### 基于属性的测试策略

**测试库选择**：使用Python的`hypothesis`库进行属性测试

**配置要求**：
- 每个属性测试至少运行100次迭代（由于随机化）
- 每个测试必须引用设计文档中的属性
- 标签格式：`# Feature: figtabminer-v17-accuracy-fix, Property N: [property text]`

**属性测试示例**：

```python
from hypothesis import given, strategies as st
import hypothesis

# 配置hypothesis
hypothesis.settings.register_profile("ci", max_examples=100)
hypothesis.settings.load_profile("ci")

# Feature: figtabminer-v17-accuracy-fix, Property 5: 边界框精确度
@given(
    detections=st.lists(
        st.builds(
            Detection,
            bbox=st.lists(st.floats(min_value=0, max_value=1000), min_size=4, max_size=4),
            type=st.sampled_from(['figure', 'table']),
            score=st.floats(min_value=0.0, max_value=1.0),
            detector=st.sampled_from(['doclayout', 'table_transformer'])
        ),
        min_size=1,
        max_size=10
    ),
    ground_truth=st.lists(
        st.builds(
            Detection,
            bbox=st.lists(st.floats(min_value=0, max_value=1000), min_size=4, max_size=4),
            type=st.sampled_from(['figure', 'table']),
            score=st.floats(min_value=0.0, max_value=1.0),
            detector=st.just('ground_truth')
        ),
        min_size=1,
        max_size=10
    )
)
def test_bbox_accuracy_property(detections, ground_truth):
    """
    Property 5: 对于任何检测到的图表，其边界框与真实标注边界框的IoU值应该大于0.85
    """
    # 匹配检测结果和真实标注
    matched_pairs = match_detections(detections, ground_truth, iou_threshold=0.5)
    
    # 验证每个匹配对的IoU
    for pred, gt in matched_pairs:
        iou = bbox_utils.bbox_iou(pred.bbox, gt.bbox)
        assert iou > 0.85, f"IoU {iou} is not greater than 0.85"

# Feature: figtabminer-v17-accuracy-fix, Property 14: 质量分数范围
@given(
    detection=st.builds(
        Detection,
        bbox=st.lists(st.floats(min_value=0, max_value=1000), min_size=4, max_size=4),
        type=st.sampled_from(['figure', 'table']),
        score=st.floats(min_value=0.0, max_value=1.0),
        detector=st.sampled_from(['doclayout', 'table_transformer'])
    )
)
def test_quality_score_range_property(detection):
    """
    Property 14: 对于任何检测结果，质量评估器计算的质量分数应该在0到1的范围内
    """
    assessor = EnhancedQualityAssessor()
    quality_score = assessor.assess_comprehensive(
        detection,
        page_image=create_dummy_image(),
        captions=[],
        page_layout={}
    )
    
    assert 0.0 <= quality_score.overall_score <= 1.0, \
        f"Quality score {quality_score.overall_score} is out of range [0, 1]"

# Feature: figtabminer-v17-accuracy-fix, Property 17: 配置验证
@given(
    config=st.fixed_dictionaries({
        'doclayout_confidence': st.floats(min_value=-1.0, max_value=2.0),
        'table_transformer_confidence': st.floats(min_value=-1.0, max_value=2.0),
        'merge_iou_threshold': st.floats(min_value=-1.0, max_value=2.0)
    })
)
def test_config_validation_property(config):
    """
    Property 17: 对于任何配置文件，如果包含无效的参数值，系统应该检测到错误并使用默认值或拒绝加载
    """
    try:
        validated_config = validate_config(config)
        
        # 验证所有值都在有效范围内
        assert 0.0 <= validated_config['doclayout_confidence'] <= 1.0
        assert 0.0 <= validated_config['table_transformer_confidence'] <= 1.0
        assert 0.0 <= validated_config['merge_iou_threshold'] <= 1.0
    except ConfigValidationError as e:
        # 如果配置无效，应该抛出明确的错误
        assert 'invalid' in str(e).lower() or 'out of range' in str(e).lower()
```

### 测试数据准备

1. **标注数据集**：
   - 创建包含各种图表类型的标注数据集
   - 包括正常案例和困难案例
   - 标注格式：JSON，包含bbox、type、chart_type等

2. **测试PDF集合**：
   - 学术论文（单栏、双栏）
   - 技术报告
   - 专利文档
   - 包含已知问题的PDF

3. **合成测试数据**：
   - 使用hypothesis生成随机检测结果
   - 创建各种边界框配置
   - 生成各种配置参数组合

### 性能测试

虽然性能不是主要的功能测试目标，但应该监控关键操作的性能：

1. **基准测试**：
   - 单页PDF处理时间
   - 各检测器的平均处理时间
   - 融合和合并的处理时间

2. **性能回归测试**：
   - 确保v1.7不比v1.6慢太多
   - 目标：单页处理时间<30秒

3. **资源使用监控**：
   - 内存使用
   - GPU内存使用
   - CPU使用率

### 集成测试

1. **端到端测试**：
   - 从PDF输入到JSON输出的完整流程
   - 验证输出格式和内容

2. **向后兼容性测试**：
   - 使用v1.6的测试用例
   - 验证API兼容性
   - 验证配置文件兼容性

3. **错误恢复测试**：
   - 模拟各种错误情况
   - 验证系统的错误处理和恢复能力

### 测试覆盖率目标

- 代码覆盖率：>80%
- 属性测试覆盖：所有21个correctness properties
- 单元测试覆盖：所有关键功能和边缘情况
- 集成测试覆盖：所有主要使用场景


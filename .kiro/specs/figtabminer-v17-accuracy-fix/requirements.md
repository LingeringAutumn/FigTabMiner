# Requirements Document

## Introduction

本文档定义了FigTabMiner v1.7准确度修复功能的需求。该功能旨在系统性解决v1.5和v1.6版本引入DocLayout-YOLO和Table Transformer模型后出现的严重准确度下降问题，包括错误识别、完全漏检、错误合并/分割、边界框不精确、图表类型识别质量差以及AI分析数据质量极差等问题。

## Glossary

- **Detection_System**: 图表检测系统，负责从PDF页面中识别和定位图表区域
- **YOLO_Detector**: DocLayout-YOLO检测器，用于检测文档布局元素
- **Table_Transformer_Detector**: Table Transformer检测器，专门用于检测表格
- **BBox_Merger**: 边界框合并器，负责合并重叠或相邻的检测框
- **Chart_Classifier**: 图表分类器，负责识别图表的具体类型
- **Quality_Assessor**: 质量评估器，评估检测结果的质量
- **AI_Analyzer**: AI分析器，提取图表的详细信息和元数据
- **Detection_Result**: 检测结果，包含边界框坐标、类型、置信度等信息
- **Ground_Truth**: 人工标注的真实数据，用于评估系统准确度
- **False_Positive**: 假阳性，系统错误地将非图表内容识别为图表
- **False_Negative**: 假阴性，系统未能检测到实际存在的图表
- **IoU**: Intersection over Union，交并比，用于衡量检测框与真实框的重叠程度
- **Confidence_Threshold**: 置信度阈值，用于过滤低置信度的检测结果
- **NMS**: Non-Maximum Suppression，非极大值抑制，用于去除重复检测框

## Requirements

### Requirement 1: 检测准确度改进

**User Story:** 作为系统用户，我希望系统能够准确识别PDF中的所有图表，避免错误识别、漏检、错误合并和错误分割，以便获得可靠的图表提取结果。

#### Acceptance Criteria

1. WHEN THE Detection_System处理包含图表的PDF页面 THEN THE Detection_System SHALL正确识别所有真实图表且假阳性率低于5%
2. WHEN THE Detection_System处理包含文字内容（正文、标题、作者、尾注、脚注、参考文献）的区域 THEN THE Detection_System SHALL NOT将这些文字内容错误识别为图表
3. WHEN THE Detection_System处理包含多个独立图表的页面 THEN THE Detection_System SHALL为每个图表生成独立的检测结果且不将它们错误合并
4. WHEN THE Detection_System处理包含复杂图表（带箭头、复杂曲线、数字文字标识）的页面 THEN THE Detection_System SHALL将整个图表识别为单一实体且不进行错误分割
5. WHEN THE Detection_System检测到图表 THEN THE Detection_System SHALL生成精确的边界框，IoU值应大于0.85

### Requirement 2: 漏检问题解决

**User Story:** 作为系统用户，我希望系统不会遗漏任何图表，确保所有图表都能被检测到，以便进行完整的文档分析。

#### Acceptance Criteria

1. WHEN THE Detection_System处理包含图表的PDF页面 THEN THE Detection_System SHALL检测到所有图表且召回率应大于95%
2. WHEN THE Detection_System使用多个检测器 THEN THE Detection_System SHALL融合所有检测器的结果以最大化召回率
3. IF THE Detection_System的某个检测器未能检测到图表 THEN THE Detection_System SHALL依靠其他检测器的结果进行补充
4. WHEN THE Detection_System调整置信度阈值 THEN THE Detection_System SHALL在保持精确率的同时最大化召回率

### Requirement 3: 边界框精确度提升

**User Story:** 作为系统用户，我希望系统生成的图表边界框精确且一致，特别是对于多张小图拼接的大图和流程图，以便准确提取图表内容。

#### Acceptance Criteria

1. WHEN THE Detection_System检测到图表 THEN THE Detection_System SHALL生成的边界框与真实边界的IoU值应大于0.85
2. WHEN THE Detection_System处理多张小图拼接成的大图 THEN THE Detection_System SHALL生成包含所有子图的完整边界框
3. WHEN THE Detection_System处理流程图 THEN THE Detection_System SHALL生成包含所有节点和连接线的完整边界框
4. WHEN THE BBox_Merger合并重叠的检测框 THEN THE BBox_Merger SHALL使用智能策略避免过度合并或合并不足

### Requirement 4: 图表类型识别改进

**User Story:** 作为系统用户，我希望系统能够准确识别图表的具体类型，减少unknown类型的数量，并扩展支持的图表类型种类，以便进行更精细的图表分析。

#### Acceptance Criteria

1. WHEN THE Chart_Classifier对图表进行分类 THEN THE Chart_Classifier SHALL将unknown类型的比例降低到10%以下
2. WHEN THE Chart_Classifier识别图表类型 THEN THE Chart_Classifier SHALL达到85%以上的分类准确率
3. THE Chart_Classifier SHALL支持至少15种不同的图表类型（包括折线图、柱状图、饼图、散点图、热图、流程图、显微镜图像、光谱图等）
4. WHEN THE Chart_Classifier对图表进行分类 THEN THE Chart_Classifier SHALL提供置信度分数以指示分类的可靠性
5. WHEN THE Chart_Classifier遇到低置信度的分类结果 THEN THE Chart_Classifier SHALL使用多种特征（OCR文本、视觉特征、上下文信息）进行综合判断

### Requirement 5: AI分析质量提升

**User Story:** 作为系统用户，我希望系统提供高质量的AI分析数据，包括准确的子类型、条件、材料候选、关键词等信息，以便深入理解图表内容。

#### Acceptance Criteria

1. WHEN THE AI_Analyzer分析图表 THEN THE AI_Analyzer SHALL提取准确的子类型且置信度应大于0.6
2. WHEN THE AI_Analyzer分析科学图表 THEN THE AI_Analyzer SHALL识别并提取实验条件、材料候选和关键词
3. WHEN THE AI_Analyzer处理OCR文本 THEN THE AI_Analyzer SHALL正确解析文本并用于类型判断和信息提取
4. WHEN THE AI_Analyzer的分析结果置信度较低 THEN THE AI_Analyzer SHALL在debug信息中提供详细的评分和推理过程
5. WHEN THE AI_Analyzer依赖前序检测结果 THEN THE AI_Analyzer SHALL验证输入数据的质量并处理错误输入

### Requirement 6: 检测器融合策略优化

**User Story:** 作为系统架构师，我希望系统能够智能地融合多个检测器的结果，发挥各检测器的优势，避免简单叠加导致的问题，以便提高整体检测性能。

#### Acceptance Criteria

1. WHEN THE Detection_System使用多个检测器 THEN THE Detection_System SHALL根据检测器的特性分配不同的权重和优先级
2. WHEN THE Detection_System融合检测结果 THEN THE Detection_System SHALL使用加权NMS或其他智能策略而非简单的IoU阈值合并
3. WHEN THE Detection_System的检测器产生冲突结果 THEN THE Detection_System SHALL使用置信度、检测器可靠性和上下文信息进行仲裁
4. WHEN THE Detection_System评估检测器性能 THEN THE Detection_System SHALL为每个检测器维护性能指标（精确率、召回率、F1分数）
5. WHERE THE Detection_System支持配置检测器组合 THEN THE Detection_System SHALL允许用户启用或禁用特定检测器

### Requirement 7: 质量评估和过滤

**User Story:** 作为系统用户，我希望系统能够评估检测结果的质量，过滤掉低质量的检测，并提供质量指标，以便获得可靠的输出结果。

#### Acceptance Criteria

1. WHEN THE Quality_Assessor评估检测结果 THEN THE Quality_Assessor SHALL计算质量分数（0-1范围）
2. WHEN THE Quality_Assessor识别低质量检测 THEN THE Quality_Assessor SHALL标记或过滤置信度低于阈值的结果
3. WHEN THE Quality_Assessor评估边界框质量 THEN THE Quality_Assessor SHALL检查边界框的大小、纵横比和位置的合理性
4. WHEN THE Quality_Assessor评估图表内容 THEN THE Quality_Assessor SHALL使用OCR文本、视觉特征和上下文信息进行综合评估
5. WHEN THE Quality_Assessor完成评估 THEN THE Quality_Assessor SHALL在输出中包含质量指标和评估依据

### Requirement 8: 准确度评估框架

**User Story:** 作为开发者，我希望有一个系统化的准确度评估框架，能够使用标注数据量化评估系统性能，跟踪改进效果，以便持续优化系统。

#### Acceptance Criteria

1. THE Detection_System SHALL支持使用人工标注的Ground_Truth数据进行准确度评估
2. WHEN THE Detection_System进行评估 THEN THE Detection_System SHALL计算精确率、召回率、F1分数和平均IoU
3. WHEN THE Detection_System进行评估 THEN THE Detection_System SHALL生成详细的错误分析报告（假阳性、假阴性、边界框偏差）
4. WHEN THE Detection_System进行评估 THEN THE Detection_System SHALL支持按图表类型、页面类型和文档类型分组统计
5. THE Detection_System SHALL提供可视化工具展示检测结果与Ground_Truth的对比

### Requirement 9: 配置和参数优化

**User Story:** 作为系统管理员，我希望能够灵活配置检测器参数、融合策略和质量阈值，并能够针对不同类型的文档进行优化，以便适应不同的使用场景。

#### Acceptance Criteria

1. THE Detection_System SHALL通过配置文件支持调整所有关键参数（置信度阈值、IoU阈值、NMS参数等）
2. WHEN THE Detection_System加载配置 THEN THE Detection_System SHALL验证配置的有效性并提供默认值
3. THE Detection_System SHALL支持为不同文档类型（学术论文、技术报告、专利文档）使用不同的配置预设
4. WHEN THE Detection_System更新配置 THEN THE Detection_System SHALL在不重启服务的情况下应用新配置
5. THE Detection_System SHALL提供配置优化工具，基于标注数据自动搜索最优参数组合

### Requirement 10: 错误处理和日志记录

**User Story:** 作为开发者，我希望系统能够妥善处理运行时错误和警告，提供详细的日志记录，以便诊断问题和监控系统运行状态。

#### Acceptance Criteria

1. WHEN THE Detection_System遇到运行时错误 THEN THE Detection_System SHALL捕获异常并记录详细的错误信息
2. WHEN THE Detection_System产生警告 THEN THE Detection_System SHALL记录警告信息但继续处理
3. THE Detection_System SHALL使用结构化日志记录关键操作（检测开始、检测完成、错误、性能指标）
4. WHEN THE Detection_System记录日志 THEN THE Detection_System SHALL包含时间戳、日志级别、模块名称和详细消息
5. THE Detection_System SHALL支持配置日志级别（DEBUG、INFO、WARNING、ERROR）以控制日志详细程度

### Requirement 11: 性能优化

**User Story:** 作为系统用户，我希望系统能够在合理的时间内完成PDF分析，减少计算时间，提高用户体验。

#### Acceptance Criteria

1. WHEN THE Detection_System处理单页PDF THEN THE Detection_System SHALL在30秒内完成检测和分析
2. WHEN THE Detection_System使用多个检测器 THEN THE Detection_System SHALL支持并行执行以减少总处理时间
3. WHEN THE Detection_System处理大型PDF THEN THE Detection_System SHALL使用批处理和缓存策略优化性能
4. WHEN THE Detection_System加载模型 THEN THE Detection_System SHALL复用已加载的模型避免重复加载
5. THE Detection_System SHALL提供性能分析工具，识别性能瓶颈并提供优化建议

### Requirement 12: 向后兼容性

**User Story:** 作为系统维护者，我希望v1.7版本保持与之前版本的兼容性，不丢失已有功能，确保平滑升级。

#### Acceptance Criteria

1. THE Detection_System SHALL保留v1.1到v1.6版本的所有核心功能
2. WHEN THE Detection_System升级到v1.7 THEN THE Detection_System SHALL支持读取和处理旧版本的配置文件
3. THE Detection_System SHALL保持API接口的向后兼容性，避免破坏现有集成
4. WHEN THE Detection_System引入新功能 THEN THE Detection_System SHALL通过配置开关允许用户选择使用新旧实现
5. THE Detection_System SHALL提供迁移指南，说明从旧版本升级到v1.7的步骤和注意事项

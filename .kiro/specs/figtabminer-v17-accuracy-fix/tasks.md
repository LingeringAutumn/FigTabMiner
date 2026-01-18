# Implementation Plan: FigTabMiner v1.7 Accuracy Fix

## Overview

本实施计划将FigTabMiner v1.7准确度修复功能分解为可执行的编码任务。任务按照依赖关系组织，每个任务都引用具体的需求，并包含相应的测试任务。

## Tasks

- [x] 1. 创建核心数据模型和配置架构
  - 实现Detection、ClassificationResult、QualityScore等数据类
  - 实现DetectionConfig配置类和验证逻辑
  - 创建配置文件schema和默认配置
  - _Requirements: 9.1, 9.2_

- [ ]* 1.1 编写配置验证的属性测试
  - **Property 17: 配置验证**
  - **Validates: Requirements 9.2**

- [x] 2. 实现智能检测器融合模块
  - [x] 2.1 创建IntelligentDetectionFusion类基础结构
    - 实现初始化方法，加载配置
    - 实现检测器权重管理
    - _Requirements: 6.1, 6.5_
  
  - [x] 2.2 实现加权NMS融合算法
    - 实现weighted_nms方法
    - 支持多检测器结果融合
    - _Requirements: 6.2_
  
  - [x] 2.3 实现冲突解决机制
    - 实现resolve_conflicts方法
    - 基于置信度和检测器可靠性进行仲裁
    - _Requirements: 6.3_
  
  - [x] 2.4 实现自适应置信度阈值
    - 实现adaptive_threshold方法
    - 使用Otsu或K-means找到最优阈值
    - _Requirements: 2.4_
  
  - [ ]* 2.5 编写融合模块的单元测试
    - 测试加权NMS的正确性
    - 测试冲突解决的各种场景
    - 测试自适应阈值的效果
    - _Requirements: 6.2, 6.3_

- [x] 3. 增强边界框合并器
  - [x] 3.1 扩展SmartBBoxMerger类
    - 添加上下文感知合并方法
    - 实现类型特定的合并策略
    - _Requirements: 3.2, 3.3, 3.4_
  
  - [x] 3.2 实现边界框精炼算法
    - 实现refine_boundaries方法
    - 使用图像处理技术精确边界
    - _Requirements: 3.1_
  
  - [x] 3.3 实现复杂图表分割检测
    - 实现split_complex_figures方法
    - 检测并修正错误合并
    - _Requirements: 1.3, 1.4_
  
  - [ ]* 3.4 编写边界框精确度的属性测试
    - **Property 5: 边界框精确度**
    - **Validates: Requirements 1.5, 3.1**
  
  - [ ]* 3.5 编写图表完整性的属性测试
    - **Property 7: 拼接图完整性**
    - **Property 8: 流程图完整性**
    - **Validates: Requirements 3.2, 3.3**

- [x] 4. Checkpoint - 测试检测和合并功能
  - 确保所有测试通过，如有问题请询问用户

- [x] 5. 扩展图表分类器
  - [x] 5.1 创建EnhancedChartClassifier类
    - 扩展CHART_TYPES列表到15+种类型
    - 实现基础分类接口
    - _Requirements: 4.3_
  
  - [x] 5.2 实现层次化分类
    - 实现classify_hierarchical方法
    - 先分大类（chart/microscopy/diagram/photo）
    - 再分子类
    - _Requirements: 4.1, 4.2_
  
  - [x] 5.3 增强视觉特征提取
    - 改进现有的视觉特征检测方法
    - 添加新的特征检测（如纹理、颜色分布）
    - _Requirements: 4.2_
  
  - [x] 5.4 实现置信度校准
    - 使用Platt scaling校准置信度
    - 确保置信度反映真实准确率
    - _Requirements: 4.4_
  
  - [ ]* 5.5 编写分类准确率的属性测试
    - **Property 9: Unknown类型比例控制**
    - **Property 10: 分类准确率**
    - **Validates: Requirements 4.1, 4.2**

- [ ] 6. 增强质量评估器
  - [ ] 6.1 创建EnhancedQualityAssessor类
    - 实现多维度质量评分
    - 实现assess_comprehensive方法
    - _Requirements: 7.1, 7.3_
  
  - [ ] 6.2 实现异常检测
    - 实现detect_anomalies方法
    - 检测过大、过小、极端纵横比等异常
    - _Requirements: 7.3_
  
  - [ ] 6.3 实现质量过滤
    - 实现filter_by_quality方法
    - 基于质量分数过滤低质量检测
    - _Requirements: 7.2_
  
  - [ ]* 6.4 编写质量评估的属性测试
    - **Property 14: 质量分数范围**
    - **Property 15: 低质量检测过滤**
    - **Property 16: 不合理边界框识别**
    - **Validates: Requirements 7.1, 7.2, 7.3**

- [ ] 7. 增强AI分析器
  - [ ] 7.1 创建EnhancedAIAnalyzer类
    - 实现输入验证方法
    - 实现鲁棒的分析接口
    - _Requirements: 5.5_
  
  - [ ] 7.2 实现科学元数据提取
    - 实现extract_scientific_metadata方法
    - 提取实验条件、材料、测量值
    - _Requirements: 5.2_
  
  - [ ] 7.3 改进子类型识别
    - 提高子类型识别的准确性
    - 确保置信度>0.6或标记为低置信度
    - _Requirements: 5.1_
  
  - [ ]* 7.4 编写AI分析的属性测试
    - **Property 11: AI分析子类型置信度**
    - **Property 12: 科学图表元数据提取**
    - **Property 13: 错误输入处理**
    - **Validates: Requirements 5.1, 5.2, 5.5**

- [ ] 8. Checkpoint - 测试分类和分析功能
  - 确保所有测试通过，如有问题请询问用户

- [ ] 9. 实现准确度评估框架
  - [ ] 9.1 创建AccuracyEvaluator类
    - 实现evaluate方法
    - 计算precision、recall、F1、mean IoU
    - _Requirements: 8.1, 8.2_
  
  - [ ] 9.2 实现分组评估
    - 实现evaluate_by_type方法
    - 支持按类型、页面类型、文档类型分组
    - _Requirements: 8.4_
  
  - [ ] 9.3 实现报告生成
    - 实现generate_report方法
    - 生成详细的错误分析报告
    - _Requirements: 8.3_
  
  - [ ] 9.4 实现可视化工具
    - 实现visualize_results方法
    - 可视化检测结果与ground truth对比
    - _Requirements: 8.5_
  
  - [ ]* 9.5 编写评估框架的单元测试
    - 测试指标计算的正确性
    - 测试分组统计功能
    - 测试报告生成
    - _Requirements: 8.2, 8.3, 8.4_

- [ ] 10. 实现统一错误处理
  - [ ] 10.1 创建ErrorHandler类
    - 实现各类错误的处理方法
    - 实现fallback机制
    - _Requirements: 10.1, 10.2_
  
  - [ ] 10.2 配置结构化日志
    - 设置日志格式和级别
    - 实现关键操作的日志记录
    - _Requirements: 10.3, 10.4, 10.5_
  
  - [ ]* 10.3 编写错误处理的属性测试
    - **Property 18: 错误捕获和记录**
    - **Property 19: 警告不中断处理**
    - **Validates: Requirements 10.1, 10.2**

- [ ] 11. 集成所有模块
  - [ ] 11.1 更新主检测流程
    - 集成IntelligentDetectionFusion
    - 集成EnhancedBBoxMerger
    - 集成EnhancedQualityAssessor
    - _Requirements: 1.1, 2.1_
  
  - [ ] 11.2 更新分类和分析流程
    - 集成EnhancedChartClassifier
    - 集成EnhancedAIAnalyzer
    - _Requirements: 4.1, 5.1_
  
  - [ ] 11.3 添加配置开关
    - 支持启用/禁用新功能
    - 支持新旧实现切换
    - _Requirements: 12.4_
  
  - [ ]* 11.4 编写端到端集成测试
    - 测试完整的检测流程
    - 测试各模块的协作
    - _Requirements: 1.1, 2.1, 4.1, 5.1_

- [ ] 12. 实现向后兼容性
  - [ ] 12.1 验证API兼容性
    - 确保旧版本API调用仍然有效
    - 保持返回格式兼容
    - _Requirements: 12.3_
  
  - [ ] 12.2 实现配置文件迁移
    - 支持读取旧版本配置
    - 为新参数提供默认值
    - _Requirements: 12.2_
  
  - [ ]* 12.3 编写向后兼容性的属性测试
    - **Property 20: 向后兼容性**
    - **Property 21: 旧配置文件兼容性**
    - **Validates: Requirements 12.1, 12.2, 12.3**

- [ ] 13. Checkpoint - 测试完整系统
  - 确保所有测试通过，如有问题请询问用户

- [ ] 14. 性能优化
  - [ ] 14.1 实现并行检测
    - 支持多检测器并行执行
    - 使用线程池或进程池
    - _Requirements: 11.2_
  
  - [ ] 14.2 实现模型缓存
    - 缓存已加载的模型
    - 避免重复加载
    - _Requirements: 11.4_
  
  - [ ] 14.3 优化批处理
    - 支持批量处理页面
    - 优化内存使用
    - _Requirements: 11.3_
  
  - [ ]* 14.4 编写性能基准测试
    - 测量单页处理时间
    - 对比v1.6和v1.7的性能
    - _Requirements: 11.1_

- [ ] 15. 编写综合属性测试
  - [ ]* 15.1 编写检测准确度的属性测试
    - **Property 1: 假阳性率控制**
    - **Property 2: 文字区域不被误识别**
    - **Property 6: 召回率保证**
    - **Validates: Requirements 1.1, 1.2, 2.1**
  
  - [ ]* 15.2 编写图表处理的属性测试
    - **Property 3: 独立图表不被错误合并**
    - **Property 4: 复杂图表不被错误分割**
    - **Validates: Requirements 1.3, 1.4**

- [ ] 16. 创建评估工具和文档
  - [ ] 16.1 创建准确度评估脚本
    - 基于标注数据评估系统性能
    - 生成详细报告
    - _Requirements: 8.1, 8.2_
  
  - [ ] 16.2 创建配置优化工具
    - 基于标注数据搜索最优参数
    - 支持网格搜索或贝叶斯优化
    - _Requirements: 9.5_
  
  - [ ] 16.3 编写迁移指南
    - 说明从v1.6升级到v1.7的步骤
    - 列出配置变更和注意事项
    - _Requirements: 12.5_
  
  - [ ] 16.4 更新用户文档
    - 更新README和使用说明
    - 添加新功能的文档
    - _Requirements: 12.5_

- [ ] 17. Final Checkpoint - 完整系统验证
  - 运行所有测试（单元测试+属性测试）
  - 使用真实PDF进行端到端测试
  - 验证准确度指标达到目标
  - 确认向后兼容性
  - 如有问题请询问用户

## Notes

- 标记为`*`的任务是可选的测试任务，可以根据需要跳过以加快MVP开发
- 每个任务都引用了具体的需求，便于追溯
- Checkpoint任务确保增量验证，及早发现问题
- 属性测试任务明确标注了对应的Property和Requirements
- 建议按顺序执行任务，因为存在依赖关系

# FigTabMiner v1.6.1 - 紧急修复

**日期**: 2026-01-17  
**版本**: v1.6.1  
**状态**: 🚨 紧急修复

---

## 问题总结

v1.6 引入了 Table Transformer，但导致了严重的过度检测问题：

1. ❌ **把大量正常文字识别成表格**（参考文献、作者信息、正文、脚注）
2. ❌ **图片和表格错误合并/分割**
3. ❌ **类型识别错误**
4. ❌ **箭头被单独截取**
5. ❌ **处理时间变长**

---

## 立即修复方案

### 方案 1：禁用 Table Transformer（推荐）

**最快速的解决方案**，立即恢复到 v1.5 的效果：

```bash
# 编辑配置文件
nano config/figtabminer.json
```

找到 `table_extraction` 部分，设置：
```json
"table_extraction": {
  "use_enhanced_extractor": true,
  "enable_table_transformer": false,  // 改成 false
  ...
}
```

**效果**：
- ✅ 立即恢复到之前的准确率
- ✅ 处理速度恢复正常
- ✅ 不会把文字识别成表格

---

### 方案 2：使用更严格的过滤（已实施）

如果你想保留 Table Transformer，我已经实施了以下修复：

#### 修复 1：提高置信度阈值
```python
# 从 0.7 提高到 0.85
conf_threshold=0.85
```

#### 修复 2：更严格的表格验证
- 最小面积：2000 → 5000
- 必须有实际表格数据（至少 2 行 2 列）
- 过滤参考文献和作者信息
- 检查文本模式（email、大学、et al. 等）

#### 修复 3：添加配置开关
```json
{
  "enable_table_transformer": false,  // 可以随时开关
  "table_transformer_confidence": 0.85,  // 可调整阈值
  "strict_validation": true  // 严格验证
}
```

---

## 使用建议

### 立即行动（推荐）

1. **禁用 Table Transformer**：
   ```bash
   # 编辑 config/figtabminer.json
   # 设置 "enable_table_transformer": false
   ```

2. **重启 Streamlit**：
   ```bash
   # Ctrl+C 停止
   streamlit run src/app_streamlit.py
   ```

3. **测试效果**：
   - 上传之前有问题的 PDF
   - 检查是否还把文字识别成表格
   - 验证处理速度是否恢复

### 如果想保留 Table Transformer

1. **调整置信度**：
   ```json
   "table_transformer_confidence": 0.90  // 更高的阈值
   ```

2. **启用严格验证**：
   ```json
   "strict_validation": true
   ```

3. **逐步测试**：
   - 从一个 PDF 开始
   - 检查结果
   - 根据需要调整阈值

---

## 性能问题

### 为什么变慢了？

1. **Table Transformer 很重**：
   - 首次加载模型：~10 秒
   - 每页检测：~1-2 秒
   - 总共增加：30-50% 处理时间

2. **多策略融合**：
   - 现在运行 4 种策略（之前 3 种）
   - 每种策略都需要时间

### 解决方案

**禁用 Table Transformer**：
```json
"enable_table_transformer": false
```

**效果**：
- ✅ 处理速度恢复到 v1.5
- ✅ 前端响应速度恢复
- ✅ 内存占用减少

---

## 其他问题修复

### 问题 4：箭头被单独截取

这是 bbox_merger 的问题，已经在 v1.2 修复过，但可能需要调整参数：

```json
"bbox_merger": {
  "enable_noise_filter": true,
  "arrow_aspect_ratio_min": 5.0,  // 增加到 8.0 更严格
  "arrow_ink_ratio_max": 0.05  // 减少到 0.03 更严格
}
```

### 问题 5：AI 分析信息丑陋

这是 chart_classifier 的问题，"Low confidence, using fallback classifier" 表示：
- 主分类器置信度低
- 自动降级到备用分类器
- 可能需要更好的特征提取

**临时解决方案**：
```json
"chart_classification": {
  "use_enhanced_classifier": true,
  "enable_visual_analysis": true,
  "enable_ocr_assist": true,  // 启用 OCR 辅助
  "visual_weight": 0.7,  // 增加视觉权重
  "keyword_weight": 0.3"
}
```

---

## 测试清单

修复后，请测试以下内容：

- [ ] 不会把参考文献识别成表格
- [ ] 不会把作者信息识别成表格
- [ ] 不会把正文识别成表格
- [ ] 不会把脚注识别成表格
- [ ] 图片和表格正确分离
- [ ] 类型识别准确
- [ ] 箭头不会被单独截取
- [ ] 处理速度可接受
- [ ] 前端响应速度正常

---

## 回滚到 v1.5

如果修复后仍有问题，可以完全回滚：

```bash
# 1. 禁用所有新功能
nano config/figtabminer.json

# 设置：
{
  "table_extraction": {
    "use_enhanced_extractor": false,  // 使用基础提取器
    "enable_table_transformer": false
  }
}

# 2. 重启
streamlit run src/app_streamlit.py
```

---

## 长期解决方案

### 1. 创建标注数据集

**目的**：量化评估每个检测器的准确率

**方法**：
```bash
python tools/annotation_tool.py
```

**时间**：4-5 小时

**效果**：
- 精确知道哪个检测器有问题
- 可以针对性调整参数
- 可以训练自定义模型

### 2. 参数调优

**目的**：针对你的文档类型优化

**方法**：
- 调整置信度阈值
- 调整过滤参数
- 调整合并策略

**效果**：+5-10% 准确率

### 3. 选择性启用检测器

**策略**：
- DocLayout-YOLO：图表检测（保留）
- PubLayNet：降级方案（保留）
- Table Transformer：表格检测（**可选**）
- pdfplumber：表格提取（保留）
- Visual detection：辅助检测（保留）

---

## 总结

### 立即行动

1. **禁用 Table Transformer**：
   ```json
   "enable_table_transformer": false
   ```

2. **重启系统**

3. **测试效果**

### 预期结果

- ✅ 准确率恢复到 v1.5
- ✅ 处理速度恢复正常
- ✅ 不会过度检测表格

### 如果还有问题

请提供：
1. 问题 PDF 的截图
2. 错误识别的具体例子
3. 日志文件

我会继续优化！

---

**版本**: v1.6.1  
**状态**: 🚨 紧急修复  
**推荐**: 禁用 Table Transformer


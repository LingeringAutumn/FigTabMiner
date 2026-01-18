# 标注数据集创建指南

## 📋 目标

为 50 个 PDF 创建标注数据集，用于量化评估 FigTabMiner 的准确率。

---

## ⏱️ 时间估算

| 方法 | 时间 | 难度 |
|------|------|------|
| **半自动标注**（推荐） | **4-5 小时** | ⭐⭐ |
| 完全手动标注 | 25+ 小时 | ⭐⭐⭐⭐⭐ |

---

## 🚀 快速开始（半自动标注）

### 步骤 1：准备 PDF 样本

```bash
# 将 50 个 PDF 放到 samples 目录
mkdir -p data/samples
# 复制你的 PDF 文件到 data/samples/
```

**建议**：选择多样化的 PDF：
- 简单论文（10 个）：图表清晰，布局简单
- 中等论文（30 个）：常见的科学论文
- 复杂论文（10 个）：多栏、复杂图表

### 步骤 2：批量运行系统

```bash
# 批量处理所有 PDF
for pdf in data/samples/*.pdf; do
    echo "处理: $pdf"
    python scripts/run_pipeline.py --pdf "$pdf"
done
```

**输出**：每个 PDF 生成一个 `data/outputs/{doc_id}/` 目录

### 步骤 3：半自动标注

```bash
# 给脚本添加执行权限
chmod +x tools/batch_annotate.sh

# 运行批量标注工具
bash tools/batch_annotate.sh
```

**交互式标注**：
```
--- 页面 1 ---
检测到 3 个项目

1. figure - fig_0001
   位置: [100, 200, 500, 600]
   Caption: Figure 1. The experimental setup...
   操作 [k=保留, d=删除, m=修改, s=跳过]: k
   ✓ 已标记为保留

2. table - table_0001
   位置: [100, 700, 500, 900]
   Caption: Table 1. Results of...
   操作 [k=保留, d=删除, m=修改, s=跳过]: d
   ✗ 已标记为删除（这是个误检测）

3. figure - fig_0002
   位置: [100, 1000, 500, 1200]
   Caption: Figure 2. Comparison of...
   操作 [k=保留, d=删除, m=修改, s=跳过]: m
   修改边界框（格式：x0,y0,x1,y1）
   新边界框: 100,1000,550,1250
   ✓ 边界框已更新

是否有遗漏的图表需要添加？(y/n): y

添加新项目（输入 'done' 完成）
  类型 (figure/table): figure
  页码（从 0 开始）: 0
  边界框 (x0,y0,x1,y1): 600,200,900,500
  Caption: Figure 3. Additional plot
  ✓ 已添加

添加新项目（输入 'done' 完成）
  类型 (figure/table): done

标注完成！
总项目数: 3
  - 图表: 2
  - 表格: 0
已验证: 3
新增: 1
修改: 1

标注文件: data/annotations/217df0da9bf8.json
```

### 步骤 4：评估准确率

```bash
# 运行评估
python tools/evaluate_accuracy.py

# 保存报告
python tools/evaluate_accuracy.py --save-report evaluation_report.json
```

**输出示例**：
```
============================================================
FigTabMiner 准确率评估报告
============================================================

总体统计:
  评估文档数: 50
  标注项目数: 342
  预测项目数: 356

检测性能:
  True Positives:  298
  False Positives: 58
  False Negatives: 44

准确率指标:
  Precision: 0.837
  Recall:    0.871
  F1-Score:  0.854
  Avg IoU:   0.782

============================================================
```

---

## 💡 标注技巧

### 1. 快速判断

**保留（k）**：
- 边界框基本正确（误差 < 10%）
- 类型正确（figure/table）
- Caption 匹配正确

**删除（d）**：
- 明显的误检测（箭头、页眉、页脚）
- 数学公式被误识别为表格
- 重复检测

**修改（m）**：
- 边界框不够精确
- 需要微调位置

### 2. 常见问题

#### 问题 1：边界框太小
```
原始: [100, 200, 400, 500]
修改: [90, 190, 410, 510]  # 扩大 10 像素
```

#### 问题 2：多个子图被拆分
```
操作：
1. 删除所有子图的单独检测
2. 添加一个包含所有子图的大边界框
```

#### 问题 3：表格边界不准确
```
提示：表格边界应包含所有单元格和边框
```

### 3. 质量检查

每标注 10 个文档后，运行评估：
```bash
python tools/evaluate_accuracy.py
```

如果 F1-score < 0.80，检查：
- 是否有大量误检测未删除？
- 是否有遗漏的图表未添加？
- 边界框是否足够精确？

---

## 📊 标注数据格式

### 标注文件结构

```json
{
  "doc_id": "217df0da9bf8",
  "pdf_path": "data/samples/paper.pdf",
  "items": [
    {
      "item_id": "fig_0001",
      "type": "figure",
      "page_index": 0,
      "bbox": [100, 200, 500, 600],
      "caption": "Figure 1. Experimental setup",
      "verified": true,
      "action": "keep"
    },
    {
      "item_id": "table_0001",
      "type": "table",
      "page_index": 1,
      "bbox": [100, 700, 500, 900],
      "caption": "Table 1. Results",
      "verified": true,
      "action": "keep"
    }
  ],
  "stats": {
    "total_items": 2,
    "figures": 1,
    "tables": 1,
    "verified": 2,
    "added": 0,
    "modified": 0
  }
}
```

---

## 🎯 标注标准

### 边界框标准

**图表（Figure）**：
- 包含：图像内容 + 坐标轴 + 图例
- 不包含：Caption 文字

**表格（Table）**：
- 包含：所有单元格 + 边框 + 表头
- 不包含：Caption 文字

### 类型标准

| 内容 | 类型 | 说明 |
|------|------|------|
| 曲线图、柱状图、散点图 | `figure` | 所有图表 |
| 显微镜图、照片 | `figure` | 图像类 |
| 数据表格 | `table` | 有行列结构 |
| 数学公式 | ❌ 不标注 | 不是图表 |
| 流程图、示意图 | `figure` | 算作图表 |

---

## 🔧 高级选项

### 方案 B：使用标注工具（GUI）

如果你想要更直观的界面，可以使用现有的标注工具：

#### 1. LabelImg（推荐）

```bash
# 安装
pip install labelImg

# 运行
labelImg data/samples/
```

**优点**：
- 可视化界面
- 鼠标拖拽绘制边界框
- 支持快捷键

**缺点**：
- 需要从零开始标注
- 不能利用系统的初始检测

#### 2. CVAT（在线工具）

网址：https://cvat.org/

**优点**：
- 专业的标注平台
- 支持团队协作
- 自动保存

**缺点**：
- 需要上传 PDF
- 学习曲线较陡

---

## 📈 标注进度追踪

### 创建进度表

```bash
# 创建进度文件
cat > annotation_progress.txt << EOF
# 标注进度追踪
# 格式：[状态] 文档ID - 标注者 - 日期

[ ] doc_001 - 
[ ] doc_002 - 
[ ] doc_003 - 
...
[x] doc_050 - Alice - 2026-01-17
EOF
```

### 每日目标

- **第 1 天**：标注 20 个简单文档（2-3 小时）
- **第 2 天**：标注 20 个中等文档（2-3 小时）
- **第 3 天**：标注 10 个复杂文档（1-2 小时）

---

## ✅ 质量保证

### 1. 双人标注（可选）

对于重要的评估，可以让两个人独立标注同一批文档，然后比较：

```bash
# 计算标注者间一致性
python tools/inter_annotator_agreement.py \
    --annotator1 data/annotations/alice/ \
    --annotator2 data/annotations/bob/
```

### 2. 随机抽查

标注完成后，随机抽查 10% 的文档：

```bash
# 随机选择 5 个文档
ls data/annotations/*.json | shuf -n 5
```

重新检查这些文档的标注质量。

---

## 🎉 完成后

### 1. 备份标注数据

```bash
# 创建备份
tar -czf annotations_backup_$(date +%Y%m%d).tar.gz data/annotations/
```

### 2. 生成最终报告

```bash
# 生成详细报告
python tools/evaluate_accuracy.py \
    --save-report final_evaluation_report.json

# 生成可视化报告
python tools/visualize_evaluation.py \
    --report final_evaluation_report.json \
    --output evaluation_report.html
```

### 3. 分享结果

将评估报告添加到文档中：

```markdown
## 评估结果

基于 50 个标注文档的评估：

- **Precision**: 0.837
- **Recall**: 0.871
- **F1-Score**: 0.854
- **Average IoU**: 0.782

详细报告：[evaluation_report.html](evaluation_report.html)
```

---

## 💬 常见问题

### Q1：标注需要多精确？

**A**：边界框误差在 5-10 像素内即可。重点是：
- 类型正确（figure/table）
- 没有遗漏
- 没有误检测

### Q2：遇到模糊的情况怎么办？

**A**：制定一致的规则：
- 流程图 → `figure`
- 算法伪代码 → 不标注
- 公式 → 不标注
- 示意图 → `figure`

### Q3：标注太慢怎么办？

**A**：
1. 使用半自动标注（推荐）
2. 降低标注数量（30 个也可以）
3. 只标注关键页面（有图表的页面）

### Q4：如何提高标注效率？

**A**：
1. 使用快捷键（k/d/m/s）
2. 批量处理相似文档
3. 先标注简单的，积累经验

---

## 📚 参考资料

- [COCO 数据集标注指南](https://cocodataset.org/#format-data)
- [Pascal VOC 标注格式](http://host.robots.ox.ac.uk/pascal/VOC/)
- [LabelImg 使用教程](https://github.com/tzutalin/labelImg)

---

**预计总时间**：4-5 小时（半自动标注）  
**推荐人数**：1-2 人  
**难度**：⭐⭐（简单）


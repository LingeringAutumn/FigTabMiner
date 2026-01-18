# FigTabMiner 安装指南

## 快速安装（推荐）

### 方法 1：一键安装所有功能

```bash
# 使用安装脚本（推荐）
bash install_all.sh
```

### 方法 2：使用 pip 安装

```bash
# 1. 安装基础依赖（必需）
pip install -r requirements.txt

# 2. 安装所有增强功能（推荐）
pip install -r requirements-extra.txt
```

---

## 分步安装

如果你想逐步安装，或者遇到问题：

### 步骤 1：基础依赖（必需）

```bash
pip install -r requirements.txt
```

这会安装：
- PyMuPDF（PDF 处理）
- pdfplumber（表格提取）
- OpenCV（图像处理）
- NumPy, Pandas, Matplotlib（数据处理）
- Streamlit（Web UI）

### 步骤 2：增强功能（可选但推荐）

#### 2.1 OCR 支持

```bash
pip install 'easyocr>=1.7.0'
```

#### 2.2 表格提取增强

```bash
pip install 'camelot-py[cv]>=0.11.0' 'ghostscript>=0.7'
```

#### 2.3 布局检测 - PubLayNet

```bash
pip install 'layoutparser>=0.3.4'

# detectron2 可能需要特殊安装
pip install detectron2
# 如果失败，尝试：
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

#### 2.4 布局检测 - DocLayout-YOLO（v1.5，推荐）

```bash
pip install doclayout-yolo
```

**效果**：图表检测准确率 +15-20%

#### 2.5 表格检测 - Table Transformer（v1.6，推荐）

```bash
# 注意：在 zsh 中需要用引号
pip install 'transformers>=4.30.0' 'torch>=2.0.0' 'torchvision>=0.15.0'
```

**效果**：表格检测准确率 +10-15%，无边框表格 +88%

---

## 验证安装

### 运行测试

```bash
# 测试所有功能
python tests/test_v1.5_v1.6_improvements.py
```

### 检查系统状态

```python
from figtabminer import layout_detect

status = layout_detect.get_layout_status()
print(f"Primary detector: {status['primary_detector']}")
print(f"Status: {status['status']}")
```

### 测试完整流程

```bash
# 测试一个 PDF
python scripts/run_pipeline.py --pdf data/samples/2110.14774v1.pdf

# 启动 Web UI
streamlit run src/app_streamlit.py
```

---

## 常见问题

### Q1: zsh 安装失败 `4.30.0 not found`

**原因**：zsh 把 `>=` 解释成重定向符号

**解决**：用引号包裹包名
```bash
pip install 'transformers>=4.30.0'
```

或者直接用 requirements 文件：
```bash
pip install -r requirements-extra.txt
```

### Q2: detectron2 安装失败

**解决**：
```bash
# 方法 1：从源安装
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 方法 2：跳过 detectron2，使用 DocLayout-YOLO
pip install doclayout-yolo
```

### Q3: CUDA/GPU 相关错误

**检查 CUDA**：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**如果没有 GPU**：系统会自动使用 CPU，速度会慢一些但仍然可用

### Q4: 内存不足

**解决**：
- 关闭其他程序
- 或者跳过 Table Transformer（较大的模型）
- 系统会自动降级到其他检测器

---

## 最小安装（仅基础功能）

如果你只想要基础功能：

```bash
# 只安装基础依赖
pip install -r requirements.txt

# 测试
python scripts/run_pipeline.py --pdf test.pdf
```

**功能**：
- ✅ PDF 处理
- ✅ 基础图表和表格提取
- ✅ pdfplumber 表格提取
- ✅ 视觉线条检测
- ❌ 高级布局检测
- ❌ 高级表格检测

---

## 推荐配置

### 配置 1：完整功能（推荐）

```bash
pip install -r requirements.txt
pip install -r requirements-extra.txt
```

**适合**：需要最佳准确率，有 GPU

**功能**：所有功能，最佳性能

### 配置 2：基础 + DocLayout-YOLO

```bash
pip install -r requirements.txt
pip install doclayout-yolo
```

**适合**：需要好的图表检测，但不需要高级表格检测

**功能**：图表检测 +15-20%

### 配置 3：基础 + Table Transformer

```bash
pip install -r requirements.txt
pip install 'transformers>=4.30.0' 'torch>=2.0.0' 'torchvision>=0.15.0'
```

**适合**：主要处理表格，特别是无边框表格

**功能**：表格检测 +10-15%

---

## 依赖说明

### 必需依赖（requirements.txt）

| 包 | 用途 | 大小 |
|----|------|------|
| PyMuPDF | PDF 渲染 | ~50MB |
| pdfplumber | 表格提取 | ~10MB |
| opencv-python | 图像处理 | ~50MB |
| numpy, pandas | 数据处理 | ~100MB |
| streamlit | Web UI | ~50MB |

**总计**：~260MB

### 增强依赖（requirements-extra.txt）

| 包 | 用途 | 大小 | 效果 |
|----|------|------|------|
| easyocr | OCR 文本识别 | ~500MB | 文本提取 |
| layoutparser | PubLayNet 检测 | ~100MB | 基础布局检测 |
| detectron2 | 深度学习框架 | ~200MB | 支持 PubLayNet |
| doclayout-yolo | 文档布局检测 | ~200MB | +15-20% 图表检测 |
| transformers | Transformer 模型 | ~300MB | +10-15% 表格检测 |
| torch | 深度学习框架 | ~2GB | GPU 加速 |

**总计**：~3.3GB（包含 torch）

---

## 安装时间估计

| 配置 | 下载大小 | 安装时间 | 首次运行 |
|------|---------|---------|---------|
| 基础 | ~260MB | 2-5 分钟 | 立即可用 |
| 基础 + DocLayout-YOLO | ~460MB | 5-10 分钟 | 首次下载模型 ~1 分钟 |
| 基础 + Table Transformer | ~2.5GB | 10-20 分钟 | 首次下载模型 ~2 分钟 |
| 完整 | ~3.5GB | 15-30 分钟 | 首次下载模型 ~3 分钟 |

**注意**：首次运行时，模型会自动下载到 `~/.cache/`

---

## 下一步

安装完成后：

1. **运行测试**：
   ```bash
   python tests/test_v1.5_v1.6_improvements.py
   ```

2. **测试 PDF**：
   ```bash
   python scripts/run_pipeline.py --pdf data/samples/2110.14774v1.pdf
   ```

3. **启动 UI**：
   ```bash
   streamlit run src/app_streamlit.py
   ```

4. **查看文档**：
   - `IMPROVEMENTS_V1.5_V1.6.md` - v1.5 & v1.6 改进说明
   - `FINAL_SUMMARY.md` - 项目总结
   - `QUICK_START.md` - 快速开始指南

---

## 获取帮助

如果遇到问题：

1. 查看 `IMPROVEMENTS_V1.5_V1.6.md` 的故障排除部分
2. 运行测试查看详细错误信息
3. 检查日志文件

**常用命令**：
```bash
# 查看已安装的包
pip list | grep -E "(doclayout|transformers|torch|layoutparser)"

# 检查 Python 版本
python --version

# 检查 CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

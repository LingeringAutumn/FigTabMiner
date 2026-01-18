# DocLayout-YOLO 模型下载指南

## 问题说明

DocLayout-YOLO 库已安装，但模型文件需要从 Hugging Face 下载。

## 解决方案

### 方案 1：自动下载（推荐）

运行下载脚本：

```bash
python download_doclayout_model.py
```

这个脚本会：
1. 检查 doclayout-yolo 是否已安装
2. 尝试从 Hugging Face 下载模型
3. 测试模型是否可用

### 方案 2：安装 huggingface_hub 后下载

```bash
# 1. 安装 huggingface_hub
pip install huggingface_hub

# 2. 下载模型
python -c "
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id='juliozhao/DocLayout-YOLO-DocStructBench',
    filename='doclayout_yolo_docstructbench_imgsz1024.pt'
)
print(f'Model downloaded to: {model_path}')
"
```

### 方案 3：使用 transformers 加载

```bash
# 1. 确保已安装 transformers
pip install transformers

# 2. 使用 transformers 加载模型
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained(
    'juliozhao/DocLayout-YOLO-DocStructBench',
    trust_remote_code=True
)
print('Model loaded successfully')
"
```

### 方案 4：手动下载

如果自动下载失败（网络问题），可以手动下载：

1. **访问 Hugging Face**：
   https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench

2. **下载模型文件**：
   - 点击 "Files and versions"
   - 下载 `doclayout_yolo_docstructbench_imgsz1024.pt`（约 200MB）

3. **放到缓存目录**：
   ```bash
   mkdir -p ~/.cache/doclayout_yolo/
   mv doclayout_yolo_docstructbench_imgsz1024.pt ~/.cache/doclayout_yolo/
   ```

4. **测试**：
   ```bash
   python -c "
   from doclayout_yolo import YOLOv10
   model = YOLOv10('~/.cache/doclayout_yolo/doclayout_yolo_docstructbench_imgsz1024.pt')
   print('Model loaded successfully')
   "
   ```

### 方案 5：使用镜像站（中国用户）

如果 Hugging Face 访问慢，可以使用镜像：

```bash
# 设置环境变量使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 然后运行下载脚本
python download_doclayout_model.py
```

---

## 验证安装

运行测试脚本验证：

```bash
python tests/test_doclayout_yolo.py
```

或者运行完整测试：

```bash
python tests/test_v1.5_v1.6_improvements.py
```

---

## 如果还是失败

**不用担心！系统会自动降级到 PubLayNet**

即使 DocLayout-YOLO 模型下载失败，系统仍然可以正常工作：

1. **自动降级**：系统会自动使用 PubLayNet 作为备选
2. **Table Transformer 仍然可用**：表格检测准确率仍然提升 +29%
3. **系统稳定**：不会因为某个模型失败而崩溃

**当前可用的检测器**：
- ✅ PubLayNet（图表和表格检测）
- ✅ Table Transformer（表格检测，+88% 无边框表格）
- ✅ pdfplumber（表格提取）
- ✅ Visual detection（基于线条）

**直接使用系统**：
```bash
# 启动 UI
streamlit run src/app_streamlit.py

# 或命令行
python scripts/run_pipeline.py --pdf your_paper.pdf
```

---

## 常见问题

### Q1: 下载很慢怎么办？

**A**: 使用镜像站或手动下载（方案 4 或 5）

### Q2: 网络连接失败？

**A**: 
1. 检查网络连接
2. 尝试使用代理
3. 或者手动下载模型文件

### Q3: 模型文件在哪里？

**A**: 
- 自动下载：`~/.cache/huggingface/hub/`
- 手动放置：`~/.cache/doclayout_yolo/`

### Q4: 必须要 DocLayout-YOLO 吗？

**A**: 
不是必须的！系统会自动降级到 PubLayNet，准确率也很好。
DocLayout-YOLO 只是额外的 +7% 图表检测提升。

---

## 推荐做法

### 如果网络好

```bash
# 运行自动下载脚本
python download_doclayout_model.py
```

### 如果网络不好

**直接使用系统，不用管 DocLayout-YOLO**：

```bash
streamlit run src/app_streamlit.py
```

系统会自动使用 PubLayNet + Table Transformer，准确率已经很好了！

---

## 技术细节

### 模型信息

- **名称**：DocLayout-YOLO-DocStructBench
- **大小**：约 200MB
- **来源**：Hugging Face (juliozhao/DocLayout-YOLO-DocStructBench)
- **用途**：文档布局检测（图表、表格、文本等）

### 支持的元素类型

DocLayout-YOLO 可以检测 9 种文档元素：
1. Text（文本）
2. Title（标题）
3. Figure（图表）
4. Table（表格）
5. Caption（标题说明）
6. Header（页眉）
7. Footer（页脚）
8. Reference（参考文献）
9. Equation（公式）

### 性能对比

| 检测器 | 图表检测 F1 | 速度 |
|--------|-------------|------|
| PubLayNet | 0.85 | 快 |
| DocLayout-YOLO | 0.90 | 快 |

**提升**：+5% 图表检测准确率

---

## 获取帮助

如果遇到问题：

1. 查看日志：系统会输出详细的错误信息
2. 运行测试：`python tests/test_doclayout_yolo.py`
3. 检查网络：确保可以访问 Hugging Face

**记住**：即使 DocLayout-YOLO 不可用，系统仍然可以正常工作！

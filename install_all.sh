#!/bin/bash
# FigTabMiner 完整安装脚本
# 
# 这个脚本会安装所有依赖，包括：
# - 基础依赖（必需）
# - 增强功能依赖（可选，但推荐）
# - v1.5 DocLayout-YOLO（推荐）
# - v1.6 Table Transformer（推荐）

set -e  # 遇到错误立即退出

echo "============================================================"
echo "FigTabMiner 完整安装脚本"
echo "============================================================"
echo ""

# 检查 Python 版本
echo "检查 Python 版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 版本: $python_version"
echo ""

# 升级 pip
echo "升级 pip..."
python3 -m pip install --upgrade pip
echo ""

# 安装基础依赖
echo "============================================================"
echo "步骤 1/3: 安装基础依赖"
echo "============================================================"
echo ""
echo "这些是运行 FigTabMiner 基本功能所需的依赖..."
python3 -m pip install -r requirements.txt
echo ""
echo "✓ 基础依赖安装完成"
echo ""

# 安装增强功能依赖
echo "============================================================"
echo "步骤 2/3: 安装增强功能依赖"
echo "============================================================"
echo ""
echo "这些依赖提供更好的准确率和额外功能..."
echo ""

# 分步安装，避免一次性安装失败
echo "安装 OCR 支持..."
python3 -m pip install 'easyocr>=1.7.0' || echo "⚠️  OCR 安装失败（可选功能）"
echo ""

echo "安装表格提取增强..."
python3 -m pip install 'camelot-py[cv]>=0.11.0' 'ghostscript>=0.7' || echo "⚠️  Camelot 安装失败（可选功能）"
echo ""

echo "安装 PubLayNet 布局检测..."
python3 -m pip install 'layoutparser>=0.3.4' || echo "⚠️  LayoutParser 安装失败"
echo ""

# detectron2 需要特殊处理
echo "安装 detectron2..."
if python3 -m pip install detectron2 2>/dev/null; then
    echo "✓ detectron2 安装成功"
else
    echo "⚠️  detectron2 安装失败，尝试从源安装..."
    python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git' || echo "⚠️  detectron2 安装失败（将使用其他检测器）"
fi
echo ""

echo "✓ 增强功能依赖安装完成"
echo ""

# 安装 v1.5 和 v1.6 新功能
echo "============================================================"
echo "步骤 3/3: 安装 v1.5 & v1.6 新功能"
echo "============================================================"
echo ""

echo "安装 DocLayout-YOLO (v1.5)..."
echo "这将显著提升图表和表格检测准确率（+15-20%）"
python3 -m pip install doclayout-yolo || echo "⚠️  DocLayout-YOLO 安装失败（将使用 PubLayNet）"
echo ""

echo "安装 Table Transformer (v1.6)..."
echo "这将显著提升表格检测准确率，特别是无边框表格（+10-15%）"
python3 -m pip install 'transformers>=4.30.0' 'torch>=2.0.0' 'torchvision>=0.15.0' || echo "⚠️  Table Transformer 安装失败（将使用 pdfplumber）"
echo ""

echo "✓ v1.5 & v1.6 新功能安装完成"
echo ""

# 安装完成
echo "============================================================"
echo "安装完成！"
echo "============================================================"
echo ""
echo "下一步："
echo "1. 运行测试验证安装："
echo "   python tests/test_v1.5_v1.6_improvements.py"
echo ""
echo "2. 测试完整流程："
echo "   python scripts/run_pipeline.py --pdf data/samples/2110.14774v1.pdf"
echo ""
echo "3. 启动 Web UI："
echo "   streamlit run src/app_streamlit.py"
echo ""
echo "如果遇到问题，请查看 IMPROVEMENTS_V1.5_V1.6.md 中的故障排除部分"
echo ""

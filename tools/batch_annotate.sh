#!/bin/bash
# 批量标注脚本

echo "=========================================="
echo "FigTabMiner 批量标注工具"
echo "=========================================="
echo ""

# 配置
OUTPUT_DIR="data/outputs"
ANNOTATION_DIR="data/annotations"
SAMPLES_DIR="data/samples"

# 创建标注目录
mkdir -p "$ANNOTATION_DIR"

# 统计
total=0
annotated=0

# 遍历所有输出
for doc_dir in "$OUTPUT_DIR"/*; do
    if [ -d "$doc_dir" ]; then
        doc_id=$(basename "$doc_dir")
        annotation_file="$ANNOTATION_DIR/${doc_id}.json"
        
        total=$((total + 1))
        
        # 检查是否已标注
        if [ -f "$annotation_file" ]; then
            echo "✓ 已标注: $doc_id"
            annotated=$((annotated + 1))
        else
            echo "○ 待标注: $doc_id"
            
            # 询问是否标注
            read -p "  是否现在标注？(y/n): " choice
            
            if [ "$choice" = "y" ]; then
                python tools/annotation_tool.py --doc-id "$doc_id"
                
                if [ $? -eq 0 ]; then
                    annotated=$((annotated + 1))
                    echo "  ✓ 标注完成"
                else
                    echo "  ✗ 标注失败"
                fi
            fi
        fi
        
        echo ""
    fi
done

echo "=========================================="
echo "标注进度: $annotated / $total"
echo "=========================================="


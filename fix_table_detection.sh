#!/bin/bash

echo "=========================================="
echo "FigTabMiner v1.6.1 - 紧急修复脚本"
echo "=========================================="
echo ""

# 备份配置文件
echo "1. 备份配置文件..."
cp config/figtabminer.json config/figtabminer.json.backup
echo "   ✓ 备份到: config/figtabminer.json.backup"
echo ""

# 禁用 Table Transformer
echo "2. 禁用 Table Transformer..."
python3 << 'EOF'
import json

# 读取配置
with open('config/figtabminer.json', 'r') as f:
    config = json.load(f)

# 修改配置
if 'table_extraction' not in config:
    config['table_extraction'] = {}

config['table_extraction']['enable_table_transformer'] = False
config['table_extraction']['strict_validation'] = True
config['table_extraction']['table_transformer_confidence'] = 0.85

# 保存配置
with open('config/figtabminer.json', 'w') as f:
    json.dump(config, f, indent=2)

print("   ✓ Table Transformer 已禁用")
EOF
echo ""

# 显示修改
echo "3. 配置已更新："
echo "   - enable_table_transformer: false"
echo "   - strict_validation: true"
echo "   - table_transformer_confidence: 0.85"
echo ""

echo "=========================================="
echo "修复完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 重启 Streamlit:"
echo "   streamlit run src/app_streamlit.py"
echo ""
echo "2. 测试效果"
echo ""
echo "3. 如果还有问题，查看: QUICK_FIX_v1.6.1.md"
echo ""
echo "恢复备份:"
echo "   cp config/figtabminer.json.backup config/figtabminer.json"
echo ""

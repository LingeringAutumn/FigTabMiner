#!/usr/bin/env python3
"""
DocLayout-YOLO 模型下载脚本

这个脚本会从 Hugging Face 下载 DocLayout-YOLO 模型
"""

import os
import sys
from pathlib import Path

def download_model():
    """下载 DocLayout-YOLO 模型"""
    
    print("=" * 60)
    print("DocLayout-YOLO 模型下载脚本")
    print("=" * 60)
    print()
    
    try:
        from doclayout_yolo import YOLOv10
        print("✓ doclayout-yolo 库已安装")
    except ImportError:
        print("❌ doclayout-yolo 未安装")
        print("   请运行: pip install doclayout-yolo")
        return False
    
    print()
    print("尝试从 Hugging Face 下载模型...")
    print("(首次下载可能需要几分钟，取决于网络速度)")
    print()
    
    # 方法 1: 使用 Hugging Face Hub
    try:
        print("方法 1: 从 Hugging Face Hub 下载...")
        from huggingface_hub import hf_hub_download
        
        # 下载模型文件
        model_path = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt"
        )
        
        print(f"✓ 模型下载成功！")
        print(f"  路径: {model_path}")
        
        # 测试模型
        print()
        print("测试模型...")
        model = YOLOv10(model_path)
        print("✓ 模型加载成功！")
        
        return model_path
        
    except Exception as e:
        print(f"⚠️  方法 1 失败: {e}")
        print()
    
    # 方法 2: 直接使用 transformers
    try:
        print("方法 2: 使用 transformers 加载...")
        from transformers import AutoModel
        
        model = AutoModel.from_pretrained(
            "juliozhao/DocLayout-YOLO-DocStructBench",
            trust_remote_code=True
        )
        
        print("✓ 模型加载成功！")
        return True
        
    except Exception as e:
        print(f"⚠️  方法 2 失败: {e}")
        print()
    
    # 方法 3: 手动下载链接
    print("=" * 60)
    print("自动下载失败，请手动下载：")
    print("=" * 60)
    print()
    print("1. 访问 Hugging Face:")
    print("   https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench")
    print()
    print("2. 下载模型文件:")
    print("   doclayout_yolo_docstructbench_imgsz1024.pt")
    print()
    print("3. 将文件放到以下目录:")
    print(f"   {Path.home()}/.cache/doclayout_yolo/")
    print()
    print("4. 或者直接使用绝对路径加载模型")
    print()
    
    return False


def test_model(model_path=None):
    """测试模型是否可用"""
    
    print()
    print("=" * 60)
    print("测试模型")
    print("=" * 60)
    print()
    
    try:
        from doclayout_yolo import YOLOv10
        
        if model_path:
            print(f"使用模型: {model_path}")
            model = YOLOv10(model_path)
        else:
            print("尝试使用默认模型...")
            # 尝试几个可能的模型名称
            model_names = [
                "doclayout_yolo_docstructbench_imgsz1024.pt",
                "DocLayout-YOLO-DocStructBench",
                "yolov10n.pt",
            ]
            
            model = None
            for name in model_names:
                try:
                    print(f"  尝试: {name}")
                    model = YOLOv10(name)
                    print(f"  ✓ 成功!")
                    break
                except Exception as e:
                    print(f"  ✗ 失败: {e}")
            
            if model is None:
                print("❌ 所有模型名称都失败了")
                return False
        
        print()
        print("✓ 模型加载成功！")
        print("✓ DocLayout-YOLO 已准备就绪")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    """主函数"""
    
    # 下载模型
    model_path = download_model()
    
    # 测试模型
    if model_path:
        test_model(model_path)
    else:
        test_model()
    
    print()
    print("=" * 60)
    print("完成")
    print("=" * 60)
    print()
    print("如果模型下载成功，你现在可以:")
    print("1. 运行测试: python tests/test_v1.5_v1.6_improvements.py")
    print("2. 使用系统: streamlit run src/app_streamlit.py")
    print()


if __name__ == "__main__":
    main()

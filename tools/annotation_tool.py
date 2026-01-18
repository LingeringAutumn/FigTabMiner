#!/usr/bin/env python3
"""
简单的标注修正工具

功能：
1. 加载系统自动提取的结果
2. 显示图片和边界框
3. 允许用户修正：
   - 删除错误的检测
   - 添加遗漏的图表
   - 调整边界框
4. 保存为标注数据集
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from figtabminer import utils

logger = utils.setup_logging(__name__)


class AnnotationTool:
    """简单的标注修正工具"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.annotations = []
        self.current_image = None
        self.current_boxes = []
        self.selected_box = None
        
    def load_extraction_results(self, doc_id: str):
        """加载系统提取结果作为初始标注"""
        doc_dir = self.output_dir / doc_id
        manifest_path = doc_dir / "manifest.json"
        
        if not manifest_path.exists():
            logger.error(f"Manifest not found: {manifest_path}")
            return None
        
        manifest = utils.read_json(manifest_path)
        
        # 转换为标注格式
        annotations = {
            "doc_id": doc_id,
            "pdf_path": manifest["meta"]["source_pdf"],
            "items": []
        }
        
        for item in manifest["items"]:
            ann_item = {
                "item_id": item["item_id"],
                "type": item["type"],
                "page_index": item["page_index"],
                "bbox": item["bbox"],
                "caption": item.get("caption", ""),
                "verified": False,  # 需要人工验证
                "action": "keep"    # keep, delete, modify
            }
            annotations["items"].append(ann_item)
        
        return annotations
    
    def display_page_with_boxes(self, page_image_path: str, boxes: list):
        """显示页面和边界框"""
        img = cv2.imread(page_image_path)
        if img is None:
            logger.error(f"Failed to load image: {page_image_path}")
            return None
        
        # 绘制所有边界框
        for i, box in enumerate(boxes):
            bbox = box["bbox"]
            x0, y0, x1, y1 = bbox
            
            # 颜色：绿色=已验证，红色=未验证
            color = (0, 255, 0) if box.get("verified") else (0, 0, 255)
            
            # 绘制矩形
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
            
            # 绘制标签
            label = f"{box['type']}_{i}"
            cv2.putText(img, label, (int(x0), int(y0)-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img
    
    def interactive_annotation(self, doc_id: str):
        """交互式标注界面"""
        # 加载提取结果
        annotations = self.load_extraction_results(doc_id)
        if annotations is None:
            return None
        
        print(f"\n{'='*60}")
        print(f"标注文档: {doc_id}")
        print(f"{'='*60}")
        print(f"共 {len(annotations['items'])} 个检测项")
        print()
        
        # 按页面分组
        pages = {}
        for item in annotations["items"]:
            page_idx = item["page_index"]
            if page_idx not in pages:
                pages[page_idx] = []
            pages[page_idx].append(item)
        
        # 逐页标注
        for page_idx in sorted(pages.keys()):
            items = pages[page_idx]
            
            print(f"\n--- 页面 {page_idx + 1} ---")
            print(f"检测到 {len(items)} 个项目")
            
            for i, item in enumerate(items):
                print(f"\n{i+1}. {item['type']} - {item['item_id']}")
                print(f"   位置: {item['bbox']}")
                print(f"   Caption: {item['caption'][:50]}...")
                
                # 用户选择
                while True:
                    choice = input("   操作 [k=保留, d=删除, m=修改, s=跳过]: ").lower()
                    
                    if choice == 'k':
                        item["verified"] = True
                        item["action"] = "keep"
                        print("   ✓ 已标记为保留")
                        break
                    elif choice == 'd':
                        item["verified"] = True
                        item["action"] = "delete"
                        print("   ✗ 已标记为删除")
                        break
                    elif choice == 'm':
                        print("   修改边界框（格式：x0,y0,x1,y1）")
                        new_bbox = input("   新边界框: ")
                        try:
                            bbox = [float(x) for x in new_bbox.split(',')]
                            if len(bbox) == 4:
                                item["bbox"] = bbox
                                item["verified"] = True
                                item["action"] = "modify"
                                print("   ✓ 边界框已更新")
                                break
                        except:
                            print("   ✗ 格式错误，请重试")
                    elif choice == 's':
                        print("   ⊙ 跳过")
                        break
                    else:
                        print("   无效选择，请重试")
        
        # 询问是否添加遗漏项
        print(f"\n{'='*60}")
        add_more = input("是否有遗漏的图表需要添加？(y/n): ").lower()
        
        if add_more == 'y':
            while True:
                print("\n添加新项目（输入 'done' 完成）")
                item_type = input("  类型 (figure/table): ")
                if item_type == 'done':
                    break
                
                page_idx = input("  页码（从 0 开始）: ")
                bbox = input("  边界框 (x0,y0,x1,y1): ")
                caption = input("  Caption: ")
                
                try:
                    new_item = {
                        "item_id": f"manual_{len(annotations['items'])}",
                        "type": item_type,
                        "page_index": int(page_idx),
                        "bbox": [float(x) for x in bbox.split(',')],
                        "caption": caption,
                        "verified": True,
                        "action": "add"
                    }
                    annotations["items"].append(new_item)
                    print("  ✓ 已添加")
                except Exception as e:
                    print(f"  ✗ 添加失败: {e}")
        
        return annotations
    
    def save_annotations(self, annotations: dict, output_path: Path):
        """保存标注结果"""
        # 过滤掉删除的项目
        filtered_items = [
            item for item in annotations["items"]
            if item["action"] != "delete"
        ]
        
        annotations["items"] = filtered_items
        
        # 添加统计信息
        annotations["stats"] = {
            "total_items": len(filtered_items),
            "figures": len([i for i in filtered_items if i["type"] == "figure"]),
            "tables": len([i for i in filtered_items if i["type"] == "table"]),
            "verified": len([i for i in filtered_items if i.get("verified")]),
            "added": len([i for i in filtered_items if i["action"] == "add"]),
            "modified": len([i for i in filtered_items if i["action"] == "modify"])
        }
        
        utils.write_json(annotations, output_path)
        logger.info(f"标注已保存: {output_path}")
        
        return annotations


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="标注修正工具")
    parser.add_argument("--doc-id", required=True, help="文档 ID")
    parser.add_argument("--output-dir", default="data/outputs", help="输出目录")
    parser.add_argument("--annotation-dir", default="data/annotations", help="标注保存目录")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    annotation_dir = Path(args.annotation_dir)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建标注工具
    tool = AnnotationTool(output_dir)
    
    # 交互式标注
    annotations = tool.interactive_annotation(args.doc_id)
    
    if annotations:
        # 保存标注
        output_path = annotation_dir / f"{args.doc_id}.json"
        tool.save_annotations(annotations, output_path)
        
        # 显示统计
        stats = annotations["stats"]
        print(f"\n{'='*60}")
        print("标注完成！")
        print(f"{'='*60}")
        print(f"总项目数: {stats['total_items']}")
        print(f"  - 图表: {stats['figures']}")
        print(f"  - 表格: {stats['tables']}")
        print(f"已验证: {stats['verified']}")
        print(f"新增: {stats['added']}")
        print(f"修改: {stats['modified']}")
        print(f"\n标注文件: {output_path}")
    else:
        print("标注失败")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


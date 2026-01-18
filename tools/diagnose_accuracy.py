#!/usr/bin/env python3
"""
命令行诊断工具 - FigTabMiner准确度问题诊断

用法:
    python tools/diagnose_accuracy.py <image_path>
    python tools/diagnose_accuracy.py <pdf_path> --page 1
    python tools/diagnose_accuracy.py <directory> --batch

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""

import sys
import argparse
from pathlib import Path
import json

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from figtabminer.diagnostic_analyzer import DiagnosticAnalyzer, extract_detection_features
from figtabminer.models import Detection
from figtabminer.layout_detect import detect_layout
from figtabminer import utils

logger = utils.setup_logging(__name__)


def load_detections_from_layout(image_path: str) -> list:
    """从layout_detect加载检测结果"""
    logger.info(f"Running layout detection on: {image_path}")
    
    # 运行布局检测
    layout_results = detect_layout(image_path)
    
    # 转换为Detection对象
    detections = []
    for result in layout_results:
        det = Detection(
            bbox=result['bbox'],
            type=result['type'],
            score=result['score'],
            detector=result.get('detector', 'unknown')
        )
        detections.append(det)
    
    logger.info(f"Loaded {len(detections)} detections")
    return detections


def diagnose_single_image(image_path: str, output_dir: str = None):
    """诊断单个图像"""
    logger.info(f"Diagnosing: {image_path}")
    
    # 加载检测结果
    detections = load_detections_from_layout(image_path)
    
    if not detections:
        logger.warning("No detections found")
        print(f"\n⚠️  未检测到任何图表或表格")
        return
    
    # 运行诊断
    analyzer = DiagnosticAnalyzer()
    report = analyzer.analyze_detections(image_path, detections)
    
    # 打印报告
    print("\n" + report.summary)
    
    # 保存详细报告
    if output_dir:
        output_path = Path(output_dir) / f"{Path(image_path).stem}_diagnostic_report.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed report saved to: {output_path}")
    
    print(f"\n📊 可视化图像: {report.visualization_path}")
    
    return report


def diagnose_batch(directory: str, output_dir: str = None):
    """批量诊断目录中的所有图像"""
    dir_path = Path(directory)
    
    # 查找所有图像文件
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(dir_path.glob(f'*{ext}'))
    
    if not image_files:
        logger.error(f"No image files found in: {directory}")
        return
    
    logger.info(f"Found {len(image_files)} images to diagnose")
    
    # 诊断每个图像
    all_reports = []
    for image_file in image_files:
        try:
            print(f"\n{'='*60}")
            print(f"Processing: {image_file.name}")
            print('='*60)
            
            report = diagnose_single_image(str(image_file), output_dir)
            if report:
                all_reports.append({
                    'image': image_file.name,
                    'report': report.to_dict()
                })
        except Exception as e:
            logger.error(f"Failed to diagnose {image_file}: {e}")
            continue
    
    # 生成汇总报告
    if all_reports and output_dir:
        summary_path = Path(output_dir) / "batch_diagnostic_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_reports, f, indent=2, ensure_ascii=False)
        logger.info(f"Batch summary saved to: {summary_path}")
        
        # 打印汇总统计
        print(f"\n{'='*60}")
        print("批量诊断汇总")
        print('='*60)
        
        total_arxiv = sum(r['report']['arxiv_suspects_count'] for r in all_reports)
        total_text = sum(r['report']['text_suspects_count'] for r in all_reports)
        total_missed = sum(r['report']['missed_tables_count'] for r in all_reports)
        
        print(f"总图像数: {len(all_reports)}")
        print(f"arXiv误报嫌疑总数: {total_arxiv}")
        print(f"正文误报嫌疑总数: {total_text}")
        print(f"漏检三线表总数: {total_missed}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='FigTabMiner准确度问题诊断工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 诊断单个图像
  python tools/diagnose_accuracy.py page_001.png
  
  # 诊断并保存详细报告
  python tools/diagnose_accuracy.py page_001.png --output reports/
  
  # 批量诊断目录中的所有图像
  python tools/diagnose_accuracy.py data/samples/ --batch --output reports/
        """
    )
    
    parser.add_argument('input', help='图像文件路径或目录路径')
    parser.add_argument('--batch', action='store_true', 
                       help='批量处理模式（输入为目录）')
    parser.add_argument('--output', '-o', help='输出目录（保存详细报告）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # 执行诊断
    if args.batch:
        diagnose_batch(args.input, args.output)
    else:
        diagnose_single_image(args.input, args.output)


if __name__ == '__main__':
    main()

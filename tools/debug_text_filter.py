#!/usr/bin/env python3
"""
调试脚本：分析文本误报过滤器的行为
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from figtabminer.pdf_ingest import ingest_pdf
from figtabminer.layout_detect import detect_layout
from figtabminer.detectors.doclayout_detector import DocLayoutYOLODetector
from figtabminer.models import Detection
from figtabminer.text_false_positive_filter import TextFalsePositiveFilter
import cv2

def analyze_detection(pdf_path: str, page_idx: int = 0):
    """分析特定页面的检测结果"""
    print(f"分析 PDF: {pdf_path}, 页面 {page_idx}")
    print("=" * 80)
    
    # 1. Ingest PDF
    print("\n1. 提取 PDF 信息...")
    ingest_data = ingest_pdf(pdf_path)
    page_img_path = ingest_data["page_images"][page_idx]
    page_text_lines = ingest_data.get("page_text_lines", [[]])[page_idx]
    page_text = " ".join([line.get("text", "") for line in page_text_lines]) if page_text_lines else None
    
    print(f"   页面图像: {page_img_path}")
    print(f"   页面文本长度: {len(page_text) if page_text else 0} 字符")
    
    # 2. 原始检测
    print("\n2. 原始检测结果...")
    detector = DocLayoutYOLODetector()
    raw_detections = detector.detect(page_img_path, conf_threshold=0.25)
    
    print(f"   总检测数: {len(raw_detections)}")
    
    # 统计各类型
    from collections import Counter
    label_counts = Counter(det['label'] for det in raw_detections)
    for label, count in sorted(label_counts.items()):
        print(f"     {label}: {count}")
    
    # 3. 只看 table 检测
    table_detections = [d for d in raw_detections if d['label'] == 'table']
    print(f"\n3. Table 检测: {len(table_detections)} 个")
    
    if not table_detections:
        print("   没有检测到 table，无需过滤")
        return
    
    # 转换为 Detection 对象
    detection_objects = []
    for det in table_detections:
        detection_objects.append(Detection(
            bbox=det['bbox'],
            type='table',
            score=det['score'],
            detector='doclayout_yolo'
        ))
    
    # 4. 应用过滤器
    print("\n4. 应用文本误报过滤器...")
    text_filter = TextFalsePositiveFilter(
        table_confidence_threshold=0.75,
        text_density_threshold=0.05,
        min_table_structure_score=300,
        enable_position_heuristics=True,
        enable_ocr_pattern_matching=True,
        enable_text_line_detection=True
    )
    
    # 加载图像
    image = cv2.imread(page_img_path)
    
    # 逐个分析
    for i, det in enumerate(detection_objects):
        print(f"\n   Table {i+1}/{len(detection_objects)}:")
        print(f"     BBox: {det.bbox}")
        print(f"     Score: {det.score:.3f}")
        
        # 检查各个过滤条件
        # 置信度
        if det.score < text_filter.table_confidence_threshold:
            print(f"     ❌ 置信度过低: {det.score:.3f} < {text_filter.table_confidence_threshold}")
            continue
        else:
            print(f"     ✓ 置信度通过: {det.score:.3f} >= {text_filter.table_confidence_threshold}")
        
        # 位置启发式
        is_pos_fp, pos_reason = text_filter.check_position_heuristics(det, image.shape)
        if is_pos_fp:
            print(f"     ❌ 位置启发式: {pos_reason}")
        else:
            print(f"     ✓ 位置启发式通过")
        
        # OCR 文本模式
        is_pattern_fp, pattern_reason = text_filter.detect_text_pattern(det, image, page_text)
        if is_pattern_fp:
            print(f"     ❌ 文本模式: {pattern_reason}")
        else:
            print(f"     ✓ 文本模式通过")
        
        # 连续文本行
        is_text_line_fp, text_line_reason = text_filter.detect_continuous_text_lines(det, image)
        if is_text_line_fp:
            print(f"     ❌ 连续文本行: {text_line_reason}")
        else:
            print(f"     ✓ 连续文本行通过")
        
        # 内容特征
        is_content_fp, content_reason = text_filter.is_text_false_positive(det, image)
        if is_content_fp:
            print(f"     ❌ 内容特征: {content_reason}")
        else:
            print(f"     ✓ 内容特征通过")
    
    # 5. 最终过滤结果
    print("\n5. 最终过滤结果...")
    kept, removed = text_filter.filter(detection_objects, page_img_path, page_text)
    print(f"   保留: {len(kept)} 个")
    print(f"   移除: {len(removed)} 个")
    
    if removed:
        print("\n   移除的检测:")
        for det in removed:
            print(f"     - BBox: {det.bbox}, Score: {det.score:.3f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python debug_text_filter.py <pdf_path> [page_idx]")
        print("示例: python debug_text_filter.py data/samples/2508.08441v1.pdf 0")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    page_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    analyze_detection(pdf_path, page_idx)

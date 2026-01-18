#!/usr/bin/env python3
"""
Diagnostic Analyzer for FigTabMiner Critical Accuracy Fixes.

This module provides tools to analyze detection results and identify
common accuracy problems including:
- arXiv编号被误识别为figure
- 正文被误识别为table
- 三线表漏检

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import cv2
from pathlib import Path

from .models import Detection
from . import bbox_utils
from . import utils

logger = utils.setup_logging(__name__)


@dataclass
class DiagnosticReport:
    """诊断报告数据模型"""
    total_detections: int
    detections_by_type: Dict[str, int]
    arxiv_suspects: List[Tuple[Detection, Dict[str, Any]]]
    text_suspects: List[Tuple[Detection, Dict[str, Any]]]
    missed_tables: List[Dict[str, Any]]
    visualization_path: str
    summary: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_detections': self.total_detections,
            'detections_by_type': self.detections_by_type,
            'arxiv_suspects_count': len(self.arxiv_suspects),
            'text_suspects_count': len(self.text_suspects),
            'missed_tables_count': len(self.missed_tables),
            'visualization_path': self.visualization_path,
            'summary': self.summary
        }


def extract_detection_features(
    detection: Detection,
    image: np.ndarray
) -> Dict[str, Any]:
    """
    提取检测框的特征
    
    Features:
        - position: (x_ratio, y_ratio) 相对位置
        - size: (width, height) 绝对尺寸
        - area_ratio: 占页面面积比例
        - aspect_ratio: 宽高比
        - ink_density: 墨水密度（黑色像素比例）
        - edge_density: 边缘密度
        - horizontal_lines: 水平线条像素数
        - vertical_lines: 垂直线条像素数
        - has_table_structure: 是否有表格结构特征
    
    Args:
        detection: Detection对象
        image: 页面图像
        
    Returns:
        特征字典
    """
    features = {}
    
    # 获取图像尺寸
    h, w = image.shape[:2]
    page_area = h * w
    
    # 获取bbox坐标
    x0, y0, x1, y1 = [int(c) for c in detection.bbox]
    
    # Clamp到图像边界
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    
    # 基本几何特征
    width = x1 - x0
    height = y1 - y0
    area = width * height
    
    features['position'] = (x0 / w, y0 / h)  # 相对位置
    features['center'] = ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h)  # 中心点相对位置
    features['size'] = (width, height)
    features['area'] = area
    features['area_ratio'] = area / page_area if page_area > 0 else 0
    features['aspect_ratio'] = width / height if height > 0 else float('inf')
    
    # 提取图像区域
    if x1 > x0 and y1 > y0:
        crop = image[y0:y1, x0:x1]
        
        if crop.size > 0:
            # 转换为灰度图
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop
            
            # 墨水密度（黑色像素比例）
            _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
            ink_pixels = np.count_nonzero(binary)
            features['ink_density'] = ink_pixels / binary.size if binary.size > 0 else 0
            
            # 边缘密度
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = np.count_nonzero(edges)
            features['edge_density'] = edge_pixels / edges.size if edges.size > 0 else 0
            
            # 检测水平和垂直线条
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
            v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
            
            features['horizontal_lines'] = np.count_nonzero(h_lines)
            features['vertical_lines'] = np.count_nonzero(v_lines)
            
            # 表格结构特征（有明显的水平和垂直线条）
            features['has_table_structure'] = (
                features['horizontal_lines'] > 100 and 
                features['vertical_lines'] > 100
            )
        else:
            # 空区域
            features['ink_density'] = 0
            features['edge_density'] = 0
            features['horizontal_lines'] = 0
            features['vertical_lines'] = 0
            features['has_table_structure'] = False
    else:
        # 无效区域
        features['ink_density'] = 0
        features['edge_density'] = 0
        features['horizontal_lines'] = 0
        features['vertical_lines'] = 0
        features['has_table_structure'] = False
    
    return features


class DiagnosticAnalyzer:
    """
    诊断分析器，用于分析检测结果并识别问题模式
    
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
    """
    
    def __init__(self):
        """初始化诊断分析器"""
        pass
    
    def analyze_detections(
        self,
        image_path: str,
        detections: List[Detection],
        ground_truth: Optional[List[Detection]] = None
    ) -> DiagnosticReport:
        """
        分析检测结果
        
        Args:
            image_path: 图像路径
            detections: 检测结果列表
            ground_truth: 可选的标注真值
            
        Returns:
            DiagnosticReport: 包含统计信息、问题模式和可视化
        """
        logger.info(f"Analyzing detections for: {image_path}")
        
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return DiagnosticReport(
                total_detections=0,
                detections_by_type={},
                arxiv_suspects=[],
                text_suspects=[],
                missed_tables=[],
                visualization_path="",
                summary="Failed to load image"
            )
        
        image_shape = image.shape[:2]  # (height, width)
        
        # 统计检测数量
        total_detections = len(detections)
        detections_by_type = {}
        for det in detections:
            det_type = det.type
            detections_by_type[det_type] = detections_by_type.get(det_type, 0) + 1
        
        # 识别arXiv误报
        arxiv_suspects = self.identify_arxiv_false_positives(detections, image_shape)
        
        # 识别正文误报
        text_suspects = self.identify_text_false_positives(detections, image)
        
        # 识别漏检的三线表
        missed_tables = self.identify_missed_three_line_tables(image, detections)
        
        # 生成可视化
        problems = {
            'arxiv_suspects': [det for det, _ in arxiv_suspects],
            'text_suspects': [det for det, _ in text_suspects],
            'missed_tables': missed_tables
        }
        visualization_path = self.generate_visualization(image_path, detections, problems)
        
        # 生成报告
        summary = self.generate_report(detections, problems)
        
        report = DiagnosticReport(
            total_detections=total_detections,
            detections_by_type=detections_by_type,
            arxiv_suspects=arxiv_suspects,
            text_suspects=text_suspects,
            missed_tables=missed_tables,
            visualization_path=visualization_path,
            summary=summary
        )
        
        logger.info(f"Diagnostic complete: {len(arxiv_suspects)} arXiv suspects, "
                   f"{len(text_suspects)} text suspects, {len(missed_tables)} missed tables")
        
        return report
    
    def identify_arxiv_false_positives(
        self,
        detections: List[Detection],
        image_shape: Tuple[int, int]
    ) -> List[Tuple[Detection, Dict[str, Any]]]:
        """
        识别可能的arXiv编号误报
        
        arXiv编号特征：
        1. 位置：左上角（y < 10% page height）
        2. 尺寸：小框（area < 5% page area）
        3. 形状：横向矩形（1.5 < aspect_ratio < 8.0）
        
        Returns:
            List of (detection, features) tuples
        """
        suspects = []
        h, w = image_shape
        page_area = h * w
        
        for det in detections:
            # 只检查figure类型
            if det.type != 'figure':
                continue
            
            x0, y0, x1, y1 = det.bbox
            
            # 计算特征
            center_y = (y0 + y1) / 2
            width = x1 - x0
            height = y1 - y0
            area = width * height
            aspect_ratio = width / height if height > 0 else float('inf')
            
            # 判断条件
            is_top = center_y < h * 0.1  # 上方10%
            is_small = area < page_area * 0.05  # 小于5%页面面积
            is_horizontal = 1.5 < aspect_ratio < 8.0  # 横向矩形
            
            if is_top and is_small and is_horizontal:
                features = {
                    'center_y_ratio': center_y / h,
                    'area_ratio': area / page_area,
                    'aspect_ratio': aspect_ratio,
                    'reason': '位置、尺寸、纵横比符合arXiv特征'
                }
                suspects.append((det, features))
                logger.debug(f"arXiv suspect: bbox={det.bbox}, features={features}")
        
        return suspects
    
    def identify_text_false_positives(
        self,
        detections: List[Detection],
        image: np.ndarray
    ) -> List[Tuple[Detection, Dict[str, Any]]]:
        """
        识别可能的正文误报为table
        
        正文误报特征：
        1. 高墨水密度（> 80%）
        2. 无明显表格线条
        3. 低表格结构分数
        4. 置信度不高（0.5-0.7之间）
        
        Returns:
            List of (detection, features) tuples
        """
        suspects = []
        
        for det in detections:
            # 只检查table类型
            if det.type != 'table':
                continue
            
            # 提取特征
            features = extract_detection_features(det, image)
            
            # 判断条件
            high_text_density = features['ink_density'] > 0.8
            no_table_lines = (
                features['horizontal_lines'] < 100 and 
                features['vertical_lines'] < 100
            )
            low_structure_score = not features['has_table_structure']
            moderate_confidence = 0.5 <= det.score <= 0.7
            
            if high_text_density and no_table_lines and low_structure_score:
                features['reason'] = '高文字密度但无表格结构'
                suspects.append((det, features))
                logger.debug(f"Text FP suspect: bbox={det.bbox}, "
                           f"ink_density={features['ink_density']:.2f}, "
                           f"h_lines={features['horizontal_lines']}, "
                           f"v_lines={features['vertical_lines']}")
        
        return suspects
    
    def identify_missed_three_line_tables(
        self,
        image: np.ndarray,
        detections: List[Detection]
    ) -> List[Dict[str, Any]]:
        """
        识别可能漏检的三线表（包括两线表）
        
        使用img2table库进行表格检测，尝试多种参数组合以检测不同类型的表格
        
        Returns:
            List of candidate regions with features
        """
        candidates = []
        
        try:
            # 尝试使用img2table库进行表格检测
            from img2table.document import Image as Img2TableImage
            
            # 创建临时文件保存图像
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, image)
            
            try:
                # 使用img2table检测表格
                img_doc = Img2TableImage(src=tmp_path, detect_rotation=False)
                
                # 尝试多种参数组合以检测不同类型的表格
                all_tables = []
                param_combinations = [
                    # 三线表（有隐式行）
                    {"implicit_rows": True, "borderless_tables": False, "min_confidence": 20},
                    # 两线表或无边框表格
                    {"implicit_rows": True, "borderless_tables": True, "min_confidence": 20},
                    # 标准表格
                    {"implicit_rows": False, "borderless_tables": False, "min_confidence": 20},
                    # 更低置信度
                    {"implicit_rows": True, "borderless_tables": True, "min_confidence": 10},
                ]
                
                for params in param_combinations:
                    try:
                        tables = img_doc.extract_tables(**params)
                        logger.debug(f"img2table with params {params}: found {len(tables)} tables")
                        all_tables.extend(tables)
                    except Exception as e:
                        logger.debug(f"img2table failed with params {params}: {e}")
                        continue
                
                logger.info(f"img2table found {len(all_tables)} tables total (with duplicates)")
                
                # 去重：合并重叠的表格
                unique_tables = []
                for table in all_tables:
                    table_bbox = table.bbox
                    x1, y1, x2, y2 = table_bbox.x1, table_bbox.y1, table_bbox.x2, table_bbox.y2
                    candidate_bbox = [float(x1), float(y1), float(x2), float(y2)]
                    
                    # 检查是否与已有表格重复
                    is_duplicate = False
                    for existing_bbox in unique_tables:
                        iou = bbox_utils.bbox_iou(candidate_bbox, existing_bbox)
                        if iou > 0.7:  # 高度重叠认为是重复
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        unique_tables.append(candidate_bbox)
                
                logger.info(f"After deduplication: {len(unique_tables)} unique tables")
                
                # 检查每个检测到的表格是否已被现有检测覆盖
                for candidate_bbox in unique_tables:
                    is_covered = False
                    max_iou = 0.0
                    for det in detections:
                        if det.type == 'table':
                            iou = bbox_utils.bbox_iou(candidate_bbox, det.bbox)
                            max_iou = max(max_iou, iou)
                            if iou > 0.3:  # IoU > 0.3认为已覆盖
                                is_covered = True
                                break
                    
                    if not is_covered:
                        # 计算表格特征
                        x1, y1, x2, y2 = candidate_bbox
                        width = x2 - x1
                        height = y2 - y1
                        
                        candidate = {
                            'bbox': candidate_bbox,
                            'width': width,
                            'height': height,
                            'max_iou_with_existing': max_iou,
                            'reason': f'img2table检测到表格但未被现有检测覆盖 (size={width:.0f}x{height:.0f})'
                        }
                        candidates.append(candidate)
                        logger.info(f"Missed table candidate from img2table: bbox={candidate_bbox}")
                
            finally:
                # 清理临时文件
                import os
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except ImportError:
            logger.warning("img2table library not available, falling back to OpenCV-based detection")
            # 如果img2table不可用，使用原来的OpenCV方法
            candidates = self._identify_missed_tables_opencv(image, detections)
        except Exception as e:
            logger.error(f"Error using img2table: {e}, falling back to OpenCV-based detection")
            candidates = self._identify_missed_tables_opencv(image, detections)
        
        logger.info(f"Found {len(candidates)} unique missed table candidates")
        return candidates
    
    def _identify_missed_tables_opencv(
        self,
        image: np.ndarray,
        detections: List[Detection]
    ) -> List[Dict[str, Any]]:
        """
        使用OpenCV方法检测漏检的三线表（备用方法）
        
        Returns:
            List of candidate regions with features
        """
        candidates = []
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用多种方法检测水平线条
        # 方法1: Canny边缘检测 + 形态学操作
        edges = cv2.Canny(gray, 30, 100)
        
        # 使用更小的kernel以检测更细的线条
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        h_lines_canny = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
        
        # 方法2: 直接在灰度图上检测暗线
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        h_lines_thresh = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        
        # 合并两种方法的结果
        h_lines = cv2.bitwise_or(h_lines_canny, h_lines_thresh)
        
        # 查找线条轮廓
        contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 提取线条信息
        min_line_length = image.shape[1] * 0.2
        lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_line_length and h < 10:
                center_y = y + h // 2
                lines.append({
                    'x0': x,
                    'y': center_y,
                    'x1': x + w,
                    'width': w,
                    'bbox': (x, y, x + w, y + h)
                })
        
        # 按y坐标排序
        lines.sort(key=lambda line: line['y'])
        
        # 合并相近的线条
        merged_lines = []
        if lines:
            current_line = lines[0]
            for next_line in lines[1:]:
                if abs(next_line['y'] - current_line['y']) < 5:
                    if next_line['width'] > current_line['width']:
                        current_line = next_line
                else:
                    merged_lines.append(current_line)
                    current_line = next_line
            merged_lines.append(current_line)
        
        logger.debug(f"OpenCV detected {len(merged_lines)} horizontal lines after merging")
        
        # 查找3条线的组合
        if len(merged_lines) >= 3:
            for i in range(len(merged_lines) - 2):
                for j in range(i + 1, len(merged_lines) - 1):
                    for k in range(j + 1, len(merged_lines)):
                        line1 = merged_lines[i]
                        line2 = merged_lines[j]
                        line3 = merged_lines[k]
                        
                        gap1 = line2['y'] - line1['y']
                        gap2 = line3['y'] - line2['y']
                        total_height = line3['y'] - line1['y']
                        
                        if (gap1 >= 20 and gap2 >= 20 and 
                            50 <= total_height <= 500 and
                            min(gap1, gap2) / max(gap1, gap2) > 0.2):
                            
                            x_min = min(line1['x0'], line2['x0'], line3['x0'])
                            y_min = line1['y'] - 5
                            x_max = max(line1['x1'], line2['x1'], line3['x1'])
                            y_max = line3['y'] + 5
                            
                            if x_max > x_min and y_max > y_min:
                                candidate_bbox = [float(x_min), float(y_min), float(x_max), float(y_max)]
                                
                                # 检查是否已被现有检测覆盖
                                is_covered = False
                                max_iou = 0.0
                                for det in detections:
                                    if det.type == 'table':
                                        iou = bbox_utils.bbox_iou(candidate_bbox, det.bbox)
                                        max_iou = max(max_iou, iou)
                                        if iou > 0.3:
                                            is_covered = True
                                            break
                                
                                if not is_covered:
                                    y0, y1 = int(y_min), int(y_max)
                                    x0, x1 = int(x_min), int(x_max)
                                    
                                    y0 = max(0, y0)
                                    y1 = min(image.shape[0], y1)
                                    x0 = max(0, x0)
                                    x1 = min(image.shape[1], x1)
                                    
                                    if y1 > y0 and x1 > x0:
                                        region = gray[y0:y1, x0:x1]
                                        
                                        _, region_binary = cv2.threshold(region, 245, 255, cv2.THRESH_BINARY_INV)
                                        ink_density = np.count_nonzero(region_binary) / region_binary.size if region_binary.size > 0 else 0
                                        
                                        if ink_density > 0.05:
                                            candidate = {
                                                'bbox': candidate_bbox,
                                                'line_count': 3,
                                                'gap1': gap1,
                                                'gap2': gap2,
                                                'total_height': total_height,
                                                'ink_density': ink_density,
                                                'max_iou_with_existing': max_iou,
                                                'reason': f'OpenCV检测到3条水平线 (gaps={gap1:.0f},{gap2:.0f}, ink={ink_density:.2%})'
                                            }
                                            candidates.append(candidate)
        
        # 去重
        unique_candidates = []
        for cand in candidates:
            is_duplicate = False
            for existing in unique_candidates:
                iou = bbox_utils.bbox_iou(cand['bbox'], existing['bbox'])
                if iou > 0.7:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_candidates.append(cand)
        
        return unique_candidates
    
    def generate_visualization(
        self,
        image_path: str,
        detections: List[Detection],
        problems: Dict[str, List]
    ) -> str:
        """
        生成可视化图像，标注检测框和问题区域
        
        Returns:
            Path to visualization image
        """
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image for visualization: {image_path}")
            return ""
        
        # 绘制所有检测框（绿色）
        for det in detections:
            x0, y0, x1, y1 = [int(c) for c in det.bbox]
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
            
            # 添加标签
            label = f"{det.type} {det.score:.2f}"
            cv2.putText(image, label, (x0, y0 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 绘制arXiv suspects（红色）
        for det in problems.get('arxiv_suspects', []):
            x0, y0, x1, y1 = [int(c) for c in det.bbox]
            color = (0, 0, 255)  # 红色
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 3)
            cv2.putText(image, "arXiv?", (x0, y0 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 绘制text suspects（橙色）
        for det in problems.get('text_suspects', []):
            x0, y0, x1, y1 = [int(c) for c in det.bbox]
            color = (0, 165, 255)  # 橙色
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 3)
            cv2.putText(image, "Text?", (x0, y0 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 绘制missed tables（蓝色虚线）
        for candidate in problems.get('missed_tables', []):
            bbox = candidate['bbox']
            x0, y0, x1, y1 = [int(c) for c in bbox]
            color = (255, 0, 0)  # 蓝色
            
            # 绘制虚线矩形
            thickness = 2
            line_type = cv2.LINE_AA
            # 顶边
            cv2.line(image, (x0, y0), (x1, y0), color, thickness, line_type)
            # 底边
            cv2.line(image, (x0, y1), (x1, y1), color, thickness, line_type)
            # 左边
            cv2.line(image, (x0, y0), (x0, y1), color, thickness, line_type)
            # 右边
            cv2.line(image, (x1, y0), (x1, y1), color, thickness, line_type)
            
            cv2.putText(image, "Missed?", (x0, y0 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 保存可视化图像
        output_path = image_path.replace('.png', '_diagnostic.png').replace('.jpg', '_diagnostic.jpg')
        cv2.imwrite(output_path, image)
        logger.info(f"Visualization saved to: {output_path}")
        
        return output_path
    
    def generate_report(
        self,
        detections: List[Detection],
        problems: Dict[str, List]
    ) -> str:
        """
        生成文本报告
        
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("诊断报告 - FigTabMiner准确度分析")
        lines.append("=" * 60)
        lines.append("")
        
        # 总体统计
        lines.append(f"总检测数: {len(detections)}")
        
        # 按类型统计
        by_type = {}
        for det in detections:
            by_type[det.type] = by_type.get(det.type, 0) + 1
        
        for det_type, count in by_type.items():
            lines.append(f"  - {det_type}: {count}")
        lines.append("")
        
        # arXiv误报
        arxiv_suspects = problems.get('arxiv_suspects', [])
        lines.append(f"🚨 arXiv编号误报嫌疑: {len(arxiv_suspects)}")
        if arxiv_suspects:
            lines.append("  特征: 左上角小框，横向矩形")
            for i, det in enumerate(arxiv_suspects[:5], 1):  # 只显示前5个
                lines.append(f"  {i}. bbox={[f'{c:.1f}' for c in det.bbox]}, "
                           f"score={det.score:.3f}, detector={det.detector}")
        lines.append("")
        
        # 正文误报
        text_suspects = problems.get('text_suspects', [])
        lines.append(f"🚨 正文误报为table嫌疑: {len(text_suspects)}")
        if text_suspects:
            lines.append("  特征: 高文字密度，无表格线条")
            for i, det in enumerate(text_suspects[:5], 1):
                lines.append(f"  {i}. bbox={[f'{c:.1f}' for c in det.bbox]}, "
                           f"score={det.score:.3f}, detector={det.detector}")
        lines.append("")
        
        # 漏检的三线表
        missed_tables = problems.get('missed_tables', [])
        lines.append(f"🚨 可能漏检的表格: {len(missed_tables)}")
        if missed_tables:
            lines.append("  特征: 未被现有检测覆盖的表格区域")
            for i, candidate in enumerate(missed_tables[:5], 1):
                bbox = candidate['bbox']
                reason = candidate.get('reason', '未知原因')
                
                # 根据候选来源显示不同的信息
                if 'gap1' in candidate and 'gap2' in candidate:
                    # OpenCV检测的三线表
                    lines.append(f"  {i}. bbox={[f'{c:.1f}' for c in bbox]}, "
                               f"gaps=({candidate['gap1']:.1f}, {candidate['gap2']:.1f})")
                else:
                    # img2table检测的表格
                    lines.append(f"  {i}. bbox={[f'{c:.1f}' for c in bbox]}")
                
                lines.append(f"      原因: {reason}")
        lines.append("")
        
        # 建议
        lines.append("💡 修复建议:")
        if arxiv_suspects:
            lines.append("  1. 实施arXiv过滤器（基于位置+尺寸+OCR验证）")
        if text_suspects:
            lines.append("  2. 提高table检测置信度阈值到0.6")
            lines.append("  3. 添加文字密度检测过滤器")
        if missed_tables:
            lines.append("  4. 使用img2table或降低检测器置信度阈值以检测更多表格")
            lines.append("  5. 考虑使用Table Transformer进行二次验证")
        
        lines.append("")
        lines.append("=" * 60)
        
        report = "\n".join(lines)
        return report

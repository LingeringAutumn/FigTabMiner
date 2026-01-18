#!/usr/bin/env python3
"""
Text False Positive Filter for FigTabMiner Critical Accuracy Fixes.

This module filters out text paragraphs that are incorrectly detected as tables.

Strategy:
1. Confidence threshold filtering (table_confidence >= 0.6)
2. Content feature filtering (high text density + no table structure)
3. Optional: Table Transformer secondary verification

Requirements: 3.1, 3.2, 3.3
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import cv2
from pathlib import Path

from .models import Detection
from . import utils

logger = utils.setup_logging(__name__)


class TextFalsePositiveFilter:
    """
    过滤正文误报为table
    
    Requirements: 3.1, 3.2, 3.3
    """
    
    def __init__(
        self,
        table_confidence_threshold: float = 0.7,
        enable_transformer_verification: bool = False,
        text_density_threshold: float = 0.08,
        min_table_structure_score: float = 200
    ):
        """
        初始化正文过滤器
        
        Args:
            table_confidence_threshold: table检测的最低置信度（默认0.7，更严格）
            enable_transformer_verification: 是否启用Table Transformer验证（默认False）
            text_density_threshold: 文字密度阈值（默认8%，更严格）
            min_table_structure_score: 最低表格结构分数（线条像素数，默认200，更严格）
        """
        self.table_confidence_threshold = table_confidence_threshold
        self.enable_transformer_verification = enable_transformer_verification
        self.text_density_threshold = text_density_threshold
        self.min_table_structure_score = min_table_structure_score
        
        logger.info(f"TextFalsePositiveFilter initialized: "
                   f"confidence_threshold={table_confidence_threshold}, "
                   f"text_density_threshold={text_density_threshold}, "
                   f"min_structure_score={min_table_structure_score}")
    
    def filter(
        self,
        detections: List[Detection],
        image_path: str
    ) -> Tuple[List[Detection], List[Detection]]:
        """
        过滤正文误报
        
        Args:
            detections: 检测结果列表
            image_path: 图像路径
            
        Returns:
            (filtered_detections, removed_detections)
        """
        logger.info(f"Filtering text false positives from {len(detections)} detections")
        
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return detections, []
        
        filtered = []
        removed = []
        
        for det in detections:
            # 只检查table类型
            if det.type != 'table':
                filtered.append(det)
                continue
            
            # 策略1: 置信度阈值过滤
            if det.score < self.table_confidence_threshold:
                removed.append(det)
                logger.debug(f"Removed table (low confidence): score={det.score:.3f} < {self.table_confidence_threshold}")
                continue
            
            # 策略2: 内容特征过滤
            is_false_positive, reason = self.is_text_false_positive(det, image)
            
            if is_false_positive:
                removed.append(det)
                logger.debug(f"Removed table (text FP): {reason}")
                continue
            
            # 策略3: Table Transformer验证（可选）
            if self.enable_transformer_verification and 0.5 <= det.score <= 0.7:
                if not self.verify_with_transformer(det, image_path):
                    removed.append(det)
                    logger.debug(f"Removed table (failed transformer verification)")
                    continue
            
            # 保留这个检测
            filtered.append(det)
        
        logger.info(f"Filtered: {len(filtered)} kept, {len(removed)} removed")
        return filtered, removed
    
    def is_text_false_positive(
        self,
        detection: Detection,
        image: np.ndarray
    ) -> Tuple[bool, str]:
        """
        判断是否为正文误报
        
        正文误报特征：
        1. 高墨水密度（> text_density_threshold）
        2. 无明显表格线条（< min_table_structure_score）
        
        Args:
            detection: Detection对象
            image: 页面图像
            
        Returns:
            (is_false_positive, reason)
        """
        # 提取检测区域
        x0, y0, x1, y1 = [int(c) for c in detection.bbox]
        
        # Clamp到图像边界
        h, w = image.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        
        if x1 <= x0 or y1 <= y0:
            return False, ""
        
        crop = image[y0:y1, x0:x1]
        
        if crop.size == 0:
            return False, ""
        
        # 转换为灰度图
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        
        # 计算墨水密度（黑色像素比例）
        _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
        ink_pixels = np.count_nonzero(binary)
        ink_density = ink_pixels / binary.size if binary.size > 0 else 0
        
        # 计算表格结构分数（线条像素数）
        structure_score = self.calculate_table_structure_score(gray)
        
        # 判断条件
        high_text_density = ink_density > self.text_density_threshold
        no_table_structure = structure_score < self.min_table_structure_score
        
        if high_text_density and no_table_structure:
            reason = f"高文字密度({ink_density:.1%})但无表格结构(score={structure_score:.0f})"
            return True, reason
        
        return False, ""
    
    def calculate_table_structure_score(
        self,
        image_crop: np.ndarray
    ) -> float:
        """
        计算表格结构分数
        
        基于：
        - 水平线条数量和质量
        - 垂直线条数量和质量
        
        Args:
            image_crop: 裁剪的灰度图像
            
        Returns:
            Score (线条像素总数), higher means more table-like
        """
        # 边缘检测
        edges = cv2.Canny(image_crop, 50, 150)
        
        # 检测水平线条
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
        h_line_pixels = np.count_nonzero(h_lines)
        
        # 检测垂直线条
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
        v_line_pixels = np.count_nonzero(v_lines)
        
        # 总分数 = 水平线 + 垂直线
        score = h_line_pixels + v_line_pixels
        
        return float(score)
    
    def verify_with_transformer(
        self,
        detection: Detection,
        image_path: str
    ) -> bool:
        """
        使用Table Transformer验证是否为真实table
        
        Args:
            detection: Detection对象
            image_path: 图像路径
            
        Returns:
            True if verified as table
        """
        try:
            from .detectors.table_transformer_detector import TableTransformerDetector
            
            # 初始化Table Transformer
            detector = TableTransformerDetector()
            
            # 检测表格
            transformer_detections = detector.detect(image_path)
            
            # 检查是否有与当前检测重叠的Table Transformer检测
            for trans_det in transformer_detections:
                if trans_det.type == 'table':
                    # 计算IoU
                    iou = detection.iou(trans_det)
                    if iou > 0.5:  # 重叠度 > 50%
                        logger.debug(f"Table Transformer verified: IoU={iou:.2f}")
                        return True
            
            logger.debug(f"Table Transformer did not verify this detection")
            return False
            
        except ImportError:
            logger.warning("Table Transformer not available, skipping verification")
            return True  # 如果不可用，默认通过
        except Exception as e:
            logger.error(f"Table Transformer verification failed: {e}")
            return True  # 出错时默认通过

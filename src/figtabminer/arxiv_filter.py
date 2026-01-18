#!/usr/bin/env python3
"""
arXiv False Positive Filter for FigTabMiner Critical Accuracy Fixes.

This module filters out arXiv identifiers that are incorrectly detected as figures.

Strategy:
1. Position filtering (top 10% of page)
2. Size filtering (< 5% of page area)
3. Aspect ratio filtering (1.5 - 8.0, horizontal rectangles)
4. Optional: OCR verification for "arXiv" text

Requirements: 2.1, 2.2, 2.5
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import cv2
from pathlib import Path

from .models import Detection
from . import utils

logger = utils.setup_logging(__name__)


class ArxivFilter:
    """
    过滤arXiv编号误报为figure
    
    Requirements: 2.1, 2.2, 2.5
    """
    
    def __init__(
        self,
        enable_ocr: bool = True,
        position_threshold: float = 0.1,
        area_threshold: float = 0.05,
        aspect_ratio_range: Tuple[float, float] = (1.5, 8.0),
        check_left_margin: bool = True,
        left_margin_threshold: float = 0.15,
        check_rotation: bool = True
    ):
        """
        初始化arXiv过滤器
        
        Args:
            enable_ocr: 是否启用OCR验证（默认True）
            position_threshold: 左上角位置阈值（y < page_height * threshold，默认0.1）
            area_threshold: 面积阈值（area < page_area * threshold，默认0.05）
            aspect_ratio_range: 纵横比范围 (min, max)，默认(1.5, 8.0)
            check_left_margin: 是否检查左侧边缘（默认True）
            left_margin_threshold: 左侧边缘阈值（x < page_width * threshold，默认0.15）
            check_rotation: 是否检查旋转文字（默认True）
        """
        self.enable_ocr = enable_ocr
        self.position_threshold = position_threshold
        self.area_threshold = area_threshold
        self.aspect_ratio_range = aspect_ratio_range
        self.check_left_margin = check_left_margin
        self.left_margin_threshold = left_margin_threshold
        self.check_rotation = check_rotation
        
        logger.info(f"ArxivFilter initialized: "
                   f"position_threshold={position_threshold}, "
                   f"area_threshold={area_threshold}, "
                   f"aspect_ratio_range={aspect_ratio_range}, "
                   f"check_left_margin={check_left_margin}, "
                   f"left_margin_threshold={left_margin_threshold}, "
                   f"enable_ocr={enable_ocr}")
    
    def filter(
        self,
        detections: List[Detection],
        image_path: str
    ) -> Tuple[List[Detection], List[Detection]]:
        """
        过滤arXiv编号误报
        
        Args:
            detections: 检测结果列表
            image_path: 图像路径
            
        Returns:
            (filtered_detections, removed_detections)
        """
        logger.info(f"Filtering arXiv false positives from {len(detections)} detections")
        
        # 加载图像获取尺寸
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return detections, []
        
        image_shape = image.shape[:2]  # (height, width)
        
        filtered = []
        removed = []
        
        for det in detections:
            # 只检查figure类型
            if det.type != 'figure':
                filtered.append(det)
                continue
            
            # 判断是否为arXiv候选
            is_candidate, reason = self.is_arxiv_candidate(det, image_shape)
            
            if not is_candidate:
                # 不是arXiv候选，保留
                filtered.append(det)
                continue
            
            # 是arXiv候选，进行OCR验证（如果启用）
            if self.enable_ocr:
                if self.verify_with_ocr(det, image):
                    # OCR确认包含"arXiv"，移除
                    removed.append(det)
                    logger.debug(f"Removed figure (arXiv confirmed by OCR): {reason}")
                else:
                    # OCR未确认，保留（可能是真实的小图）
                    filtered.append(det)
                    logger.debug(f"Kept figure (arXiv candidate but OCR negative): {reason}")
            else:
                # 不使用OCR，直接基于特征移除
                removed.append(det)
                logger.debug(f"Removed figure (arXiv candidate, no OCR): {reason}")
        
        logger.info(f"Filtered: {len(filtered)} kept, {len(removed)} removed")
        return filtered, removed
    
    def is_arxiv_candidate(
        self,
        detection: Detection,
        image_shape: Tuple[int, int]
    ) -> Tuple[bool, str]:
        """
        判断是否为arXiv标识候选
        
        arXiv标识有两种形式：
        1. 传统形式：左上角小框（横向）
        2. 新形式：左侧边缘大框（纵向，可能旋转90度）
        
        Args:
            detection: Detection对象
            image_shape: 图像尺寸 (height, width)
            
        Returns:
            (is_candidate, reason)
        """
        h, w = image_shape
        x0, y0, x1, y1 = detection.bbox
        
        # 计算特征
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        width = x1 - x0
        height = y1 - y0
        area = width * height
        page_area = h * w
        aspect_ratio = width / height if height > 0 else 0
        
        # 形式1：传统arXiv编号（左上角小框，横向）
        is_top = center_y < h * self.position_threshold
        is_small = area < page_area * self.area_threshold
        is_horizontal = self.aspect_ratio_range[0] < aspect_ratio < self.aspect_ratio_range[1]
        
        if is_top and is_small and is_horizontal:
            reason = f"传统arXiv: 位置={center_y/h:.1%}, 面积={area/page_area:.1%}, 纵横比={aspect_ratio:.2f}"
            return True, reason
        
        # 形式2：新式arXiv标识（左侧边缘，纵向大框）
        if self.check_left_margin:
            is_left = center_x < w * self.left_margin_threshold
            is_middle_height = 0.2 < center_y / h < 0.8  # 在页面中间高度
            is_large_vertical = height > width and height > h * 0.15  # 纵向且高度>15%页面高度
            is_moderate_area = 0.02 < area / page_area < 0.15  # 面积在2%-15%之间
            
            if is_left and is_middle_height and is_large_vertical and is_moderate_area:
                reason = f"新式arXiv: 左侧={center_x/w:.1%}, 高度={center_y/h:.1%}, 纵向={height/width:.2f}, 面积={area/page_area:.1%}"
                return True, reason
        
        return False, ""
    
    def verify_with_ocr(
        self,
        detection: Detection,
        image: np.ndarray
    ) -> bool:
        """
        使用OCR验证是否包含"arXiv"文本
        
        Args:
            detection: Detection对象
            image: 页面图像
            
        Returns:
            True if contains "arXiv" text
        """
        try:
            import pytesseract
            
            # 提取检测区域
            x0, y0, x1, y1 = [int(c) for c in detection.bbox]
            
            # Clamp到图像边界
            h, w = image.shape[:2]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            
            if x1 <= x0 or y1 <= y0:
                return False
            
            crop = image[y0:y1, x0:x1]
            
            if crop.size == 0:
                return False
            
            # 转换为灰度图
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop
            
            # 二值化增强对比度
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR识别
            text = pytesseract.image_to_string(binary, config='--psm 7')  # PSM 7: single line
            text = text.strip().lower()
            
            # 检查是否包含"arxiv"
            contains_arxiv = 'arxiv' in text
            
            if contains_arxiv:
                logger.debug(f"OCR detected arXiv text: '{text}'")
            
            return contains_arxiv
            
        except ImportError:
            logger.warning("pytesseract not available, skipping OCR verification")
            return False  # OCR不可用时，默认不确认（保守策略）
        except Exception as e:
            logger.error(f"OCR verification failed: {e}")
            return False  # 出错时默认不确认

#!/usr/bin/env python3
"""
Table Enhancer for FigTabMiner Critical Accuracy Fixes.

This module enhances table detection by using img2table as a supplementary detector
to find missed tables (especially three-line tables and two-line tables).

Strategy:
1. Run img2table detection on the page
2. Find tables detected by img2table but not by primary detector
3. Filter out duplicates using IoU threshold
4. Add new table detections to results

Requirements: 4.1, 4.2, 4.3
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import cv2
from pathlib import Path

from .models import Detection
from . import utils

logger = utils.setup_logging(__name__)


class TableEnhancer:
    """
    使用img2table增强表格检测，找到漏检的表格
    
    Requirements: 4.1, 4.2, 4.3
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        min_confidence: float = 0.5,
        enable_img2table: bool = True,
        shrink_bbox: bool = True,
        shrink_ratio: float = 0.05
    ):
        """
        初始化表格增强器
        
        Args:
            iou_threshold: IoU阈值，用于判断是否为重复检测（默认0.3）
            min_confidence: img2table检测的最低置信度（默认0.5）
            enable_img2table: 是否启用img2table检测（默认True）
            shrink_bbox: 是否收缩bbox以避免包含过多正文（默认True）
            shrink_ratio: bbox收缩比例（默认0.05，即收缩5%）
        """
        self.iou_threshold = iou_threshold
        self.min_confidence = min_confidence
        self.enable_img2table = enable_img2table
        self.shrink_bbox = shrink_bbox
        self.shrink_ratio = shrink_ratio
        
        # Check if img2table is available
        self.img2table_available = False
        if enable_img2table:
            try:
                import img2table
                self.img2table_available = True
                logger.info("img2table is available for table enhancement")
            except ImportError:
                logger.warning("img2table not available, table enhancement disabled")
        
        logger.info(f"TableEnhancer initialized: "
                   f"iou_threshold={iou_threshold}, "
                   f"min_confidence={min_confidence}, "
                   f"enable_img2table={enable_img2table}, "
                   f"shrink_bbox={shrink_bbox}, "
                   f"shrink_ratio={shrink_ratio}")
    
    def enhance(
        self,
        detections: List[Detection],
        image_path: str
    ) -> Tuple[List[Detection], List[Detection]]:
        """
        增强表格检测，添加漏检的表格
        
        Args:
            detections: 现有检测结果列表
            image_path: 图像路径
            
        Returns:
            (enhanced_detections, added_detections)
        """
        if not self.enable_img2table or not self.img2table_available:
            logger.debug("Table enhancement disabled or img2table not available")
            return detections, []
        
        logger.info(f"Enhancing table detection for {len(detections)} existing detections")
        
        # 使用img2table检测表格
        img2table_detections = self._detect_with_img2table(image_path)
        
        if not img2table_detections:
            logger.debug("img2table found no tables")
            return detections, []
        
        # 找到新的表格（不与现有检测重叠）
        added = []
        for img2table_det in img2table_detections:
            # 检查是否与现有检测重叠
            is_duplicate = False
            for existing_det in detections:
                if existing_det.type == 'table':
                    iou = self._calculate_iou(img2table_det.bbox, existing_det.bbox)
                    if iou > self.iou_threshold:
                        is_duplicate = True
                        logger.debug(f"Skipping duplicate table (IoU={iou:.2f})")
                        break
            
            if not is_duplicate:
                added.append(img2table_det)
                logger.debug(f"Added new table: bbox={img2table_det.bbox}")
        
        # 合并检测结果
        enhanced = detections + added
        
        logger.info(f"Enhanced: {len(enhanced)} total ({len(added)} added)")
        return enhanced, added
    
    def _detect_with_img2table(
        self,
        image_path: str
    ) -> List[Detection]:
        """
        使用img2table检测表格
        
        Args:
            image_path: 图像路径
            
        Returns:
            List of Detection objects
        """
        try:
            from img2table.document import Image
            from img2table.ocr import TesseractOCR
            
            # 创建OCR对象（可选，用于提取表格内容）
            # 这里我们只需要检测位置，不需要OCR
            ocr = None
            
            # 加载图像
            img_doc = Image(src=image_path)
            
            # 检测表格
            # implicit_rows=True: 允许检测没有明显行分隔线的表格
            # borderless_tables=True: 允许检测无边框表格
            # min_confidence=self.min_confidence: 最低置信度
            tables = img_doc.extract_tables(
                ocr=ocr,
                implicit_rows=True,
                borderless_tables=False,  # 只检测有边框的表格（三线表、两线表）
                min_confidence=int(self.min_confidence * 100)  # img2table使用0-100的置信度
            )
            
            # 转换为Detection对象
            detections = []
            for table in tables:
                # img2table返回的bbox格式: (x1, y1, x2, y2)
                bbox = [table.bbox.x1, table.bbox.y1, table.bbox.x2, table.bbox.y2]
                
                # 收缩bbox以避免包含过多正文
                if self.shrink_bbox:
                    bbox = self._shrink_bbox(bbox, self.shrink_ratio)
                
                # 创建Detection对象
                det = Detection(
                    type='table',
                    bbox=bbox,
                    score=0.8,  # img2table不提供置信度，使用固定值
                    page_num=0
                )
                detections.append(det)
            
            logger.info(f"img2table detected {len(detections)} tables")
            return detections
            
        except ImportError as e:
            logger.warning(f"img2table import failed: {e}")
            return []
        except Exception as e:
            logger.error(f"img2table detection failed: {e}")
            logger.debug("Traceback:", exc_info=True)
            return []
    
    def _shrink_bbox(
        self,
        bbox: List[float],
        shrink_ratio: float
    ) -> List[float]:
        """
        收缩边界框以避免包含过多正文
        
        Args:
            bbox: [x0, y0, x1, y1]
            shrink_ratio: 收缩比例（0.05 = 5%）
            
        Returns:
            收缩后的bbox
        """
        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0
        
        # 计算收缩量
        shrink_x = width * shrink_ratio
        shrink_y = height * shrink_ratio
        
        # 收缩bbox
        new_bbox = [
            x0 + shrink_x,
            y0 + shrink_y,
            x1 - shrink_x,
            y1 - shrink_y
        ]
        
        logger.debug(f"Shrunk bbox from {bbox} to {new_bbox}")
        return new_bbox
    
    def _calculate_iou(
        self,
        bbox1: List[float],
        bbox2: List[float]
    ) -> float:
        """
        计算两个边界框的IoU（Intersection over Union）
        
        Args:
            bbox1: [x0, y0, x1, y1]
            bbox2: [x0, y0, x1, y1]
            
        Returns:
            IoU value (0.0 - 1.0)
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # 计算并集
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        iou = inter_area / union_area
        return iou

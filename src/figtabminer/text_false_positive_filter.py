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
        table_confidence_threshold: float = 0.75,  # 提高默认值
        enable_transformer_verification: bool = False,
        text_density_threshold: float = 0.05,  # 降低默认值（更严格）
        min_table_structure_score: float = 300,  # 提高默认值
        enable_position_heuristics: bool = True,
        enable_ocr_pattern_matching: bool = True,
        enable_text_line_detection: bool = True,
        header_margin: float = 0.1,
        footer_margin: float = 0.1,
        edge_margin: float = 0.05
    ):
        """
        初始化正文过滤器
        
        Args:
            table_confidence_threshold: table检测的最低置信度（默认0.75，更严格）
            enable_transformer_verification: 是否启用Table Transformer验证（默认False）
            text_density_threshold: 文字密度阈值（默认5%，更严格）
            min_table_structure_score: 最低表格结构分数（线条像素数，默认300，更严格）
            enable_position_heuristics: 是否启用位置启发式规则（默认True）
            enable_ocr_pattern_matching: 是否启用OCR文本模式识别（默认True）
            enable_text_line_detection: 是否启用连续文本行检测（默认True）
            header_margin: 页眉区域比例（默认10%）
            footer_margin: 页脚区域比例（默认10%）
            edge_margin: 边缘区域比例（默认5%）
        """
        self.table_confidence_threshold = table_confidence_threshold
        self.enable_transformer_verification = enable_transformer_verification
        self.text_density_threshold = text_density_threshold
        self.min_table_structure_score = min_table_structure_score
        self.enable_position_heuristics = enable_position_heuristics
        self.enable_ocr_pattern_matching = enable_ocr_pattern_matching
        self.enable_text_line_detection = enable_text_line_detection
        self.header_margin = header_margin
        self.footer_margin = footer_margin
        self.edge_margin = edge_margin
        
        logger.info(f"TextFalsePositiveFilter initialized: "
                   f"confidence_threshold={table_confidence_threshold}, "
                   f"text_density_threshold={text_density_threshold}, "
                   f"min_structure_score={min_table_structure_score}, "
                   f"position_heuristics={enable_position_heuristics}, "
                   f"ocr_pattern={enable_ocr_pattern_matching}, "
                   f"text_line_detection={enable_text_line_detection}")
    
    def filter(
        self,
        detections: List[Detection],
        image_path: str,
        page_text: Optional[str] = None
    ) -> Tuple[List[Detection], List[Detection]]:
        """
        过滤正文误报
        
        Args:
            detections: 检测结果列表
            image_path: 图像路径
            page_text: 页面文本内容（用于OCR模式匹配）
            
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
                logger.info(f"Removed table (low confidence): score={det.score:.3f} < {self.table_confidence_threshold}, bbox={det.bbox}")
                continue
            
            # 策略2: 位置启发式规则
            if self.enable_position_heuristics:
                is_position_fp, pos_reason = self.check_position_heuristics(det, image.shape)
                if is_position_fp:
                    removed.append(det)
                    logger.info(f"Removed table (position heuristic): {pos_reason}, bbox={det.bbox}")
                    continue
            
            # 策略3: OCR文本模式识别
            if self.enable_ocr_pattern_matching:
                is_pattern_fp, pattern_reason = self.detect_text_pattern(det, image, page_text)
                if is_pattern_fp:
                    removed.append(det)
                    logger.info(f"Removed table (text pattern): {pattern_reason}, bbox={det.bbox}")
                    continue
            
            # 策略4: 连续文本行检测
            if self.enable_text_line_detection:
                is_text_line_fp, text_line_reason = self.detect_continuous_text_lines(det, image)
                if is_text_line_fp:
                    removed.append(det)
                    logger.info(f"Removed table (continuous text lines): {text_line_reason}, bbox={det.bbox}")
                    continue
            
            # 策略5: 内容特征过滤
            is_false_positive, reason = self.is_text_false_positive(det, image)
            
            if is_false_positive:
                removed.append(det)
                logger.info(f"Removed table (content features): {reason}, bbox={det.bbox}")
                continue
            
            # 策略6: Table Transformer验证（可选）
            if self.enable_transformer_verification and 0.5 <= det.score <= 0.7:
                if not self.verify_with_transformer(det, image_path):
                    removed.append(det)
                    logger.debug(f"Removed table (failed transformer verification)")
                    continue
            
            # 保留这个检测
            filtered.append(det)
        
        logger.info(f"Filtered: {len(filtered)} kept, {len(removed)} removed")
        return filtered, removed
    
    def check_position_heuristics(
        self,
        detection: Detection,
        image_shape: Tuple[int, int, int]
    ) -> Tuple[bool, str]:
        """
        检查位置启发式规则
        
        检测是否位于页眉、页脚、边缘等容易误报的区域
        
        Args:
            detection: Detection对象
            image_shape: 图像形状 (height, width, channels)
            
        Returns:
            (is_false_positive, reason)
        """
        h, w = image_shape[:2]
        x0, y0, x1, y1 = detection.bbox
        
        # 计算检测框的中心点和尺寸
        center_y = (y0 + y1) / 2
        center_x = (x0 + x1) / 2
        det_height = y1 - y0
        det_width = x1 - x0
        
        # 检查页眉区域（顶部）
        header_threshold = h * self.header_margin
        if center_y < header_threshold:
            # 页眉区域的小框更可能是文本
            if det_height < h * 0.05:  # 高度小于页面5%
                return True, f"位于页眉区域且尺寸较小 (y={center_y:.0f} < {header_threshold:.0f})"
        
        # 检查页脚区域（底部）
        footer_threshold = h * (1 - self.footer_margin)
        if center_y > footer_threshold:
            # 页脚区域的小框更可能是文本
            if det_height < h * 0.05:
                return True, f"位于页脚区域且尺寸较小 (y={center_y:.0f} > {footer_threshold:.0f})"
        
        # 检查左右边缘
        left_edge_threshold = w * self.edge_margin
        right_edge_threshold = w * (1 - self.edge_margin)
        
        # 左边缘
        if center_x < left_edge_threshold:
            if det_width < w * 0.1:  # 宽度小于页面10%
                return True, f"位于左边缘且宽度较小 (x={center_x:.0f} < {left_edge_threshold:.0f})"
        
        # 右边缘
        if center_x > right_edge_threshold:
            if det_width < w * 0.1:
                return True, f"位于右边缘且宽度较小 (x={center_x:.0f} > {right_edge_threshold:.0f})"
        
        return False, ""
    
    def detect_text_pattern(
        self,
        detection: Detection,
        image: np.ndarray,
        page_text: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        检测文本模式（标题、作者、参考文献等）
        
        使用OCR识别检测区域的文本内容，并匹配特定模式
        
        Args:
            detection: Detection对象
            image: 页面图像
            page_text: 页面文本内容（可选，用于辅助判断）
            
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
        
        # 尝试使用OCR识别文本
        try:
            import pytesseract
            
            # 转换为灰度图
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop
            
            # 预处理：二值化
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR识别
            text = pytesseract.image_to_string(binary, lang='eng')
            text = text.strip().lower()
            
            if not text:
                return False, ""
            
            # 定义文本模式
            # 标题模式
            title_patterns = [
                'abstract', 'introduction', 'conclusion', 'discussion',
                'methods', 'results', 'acknowledgment', 'appendix',
                'background', 'related work', 'future work'
            ]
            
            # 作者模式
            author_patterns = [
                'author', 'affiliation', 'university', 'department',
                'institute', 'laboratory', 'email', '@'
            ]
            
            # 参考文献模式
            reference_patterns = [
                'references', 'bibliography', 'citation',
                '[1]', '[2]', '[3]', '[4]', '[5]',
                'et al', 'proc.', 'ieee', 'acm', 'springer'
            ]
            
            # 页眉页脚模式
            header_footer_patterns = [
                'page', 'copyright', '©', 'preprint', 'submitted',
                'under review', 'draft', 'version'
            ]
            
            # 检查标题模式
            for pattern in title_patterns:
                if pattern in text:
                    # 检查是否是单行标题（高度较小）
                    if (y1 - y0) < h * 0.08:  # 高度小于页面8%
                        return True, f"检测到标题模式: '{pattern}'"
            
            # 检查作者模式
            for pattern in author_patterns:
                if pattern in text:
                    if (y1 - y0) < h * 0.1:  # 高度小于页面10%
                        return True, f"检测到作者信息模式: '{pattern}'"
            
            # 检查参考文献模式
            reference_count = sum(1 for pattern in reference_patterns if pattern in text)
            if reference_count >= 2:  # 至少匹配2个参考文献特征
                return True, f"检测到参考文献模式 (匹配{reference_count}个特征)"
            
            # 检查页眉页脚模式
            for pattern in header_footer_patterns:
                if pattern in text:
                    return True, f"检测到页眉/页脚模式: '{pattern}'"
            
            return False, ""
            
        except ImportError:
            logger.debug("pytesseract not available, skipping OCR pattern matching")
            return False, ""
        except Exception as e:
            logger.debug(f"OCR pattern matching failed: {e}")
            return False, ""
    
    def detect_continuous_text_lines(
        self,
        detection: Detection,
        image: np.ndarray
    ) -> Tuple[bool, str]:
        """
        检测连续文本行
        
        使用形态学操作检测文本行，计算文本行密度和连续性
        
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
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 使用形态学操作连接文本行
        # 水平方向的膨胀核，用于连接同一行的文字
        kernel_width = max(int(gray.shape[1] * 0.02), 5)  # 宽度的2%，最小5像素
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
        
        # 膨胀操作，连接同一行的文字
        dilated = cv2.dilate(binary, h_kernel, iterations=1)
        
        # 查找轮廓（文本行）
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return False, ""
        
        # 过滤掉太小的轮廓
        min_line_width = gray.shape[1] * 0.3  # 至少占宽度的30%
        min_line_height = 5  # 最小高度5像素
        
        text_lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_line_width and h >= min_line_height:
                text_lines.append((y, y + h))  # 记录行的y坐标范围
        
        if len(text_lines) < 3:  # 少于3行，不认为是连续文本
            return False, ""
        
        # 按y坐标排序
        text_lines.sort(key=lambda line: line[0])
        
        # 计算行间距
        line_gaps = []
        for i in range(len(text_lines) - 1):
            gap = text_lines[i + 1][0] - text_lines[i][1]
            line_gaps.append(gap)
        
        if not line_gaps:
            return False, ""
        
        # 计算平均行间距和标准差
        avg_gap = np.mean(line_gaps)
        std_gap = np.std(line_gaps)
        
        # 判断是否为连续文本行
        # 条件：
        # 1. 至少3行
        # 2. 行间距相对均匀（标准差小于平均值的50%）
        # 3. 行间距适中（不太大也不太小）
        
        crop_height = gray.shape[0]
        
        # 行间距应该在合理范围内（5-30像素，或者占高度的5%-15%）
        reasonable_gap_min = max(5, crop_height * 0.05)
        reasonable_gap_max = min(30, crop_height * 0.15)
        
        is_uniform_spacing = std_gap < avg_gap * 0.5 if avg_gap > 0 else False
        is_reasonable_gap = reasonable_gap_min <= avg_gap <= reasonable_gap_max
        
        if len(text_lines) >= 3 and is_uniform_spacing and is_reasonable_gap:
            # 计算文本行密度（文本行占总高度的比例）
            total_line_height = sum(line[1] - line[0] for line in text_lines)
            line_density = total_line_height / crop_height if crop_height > 0 else 0
            
            # 如果文本行密度高（>40%），更可能是正文段落
            if line_density > 0.4:
                return True, f"检测到{len(text_lines)}行连续文本 (密度={line_density:.1%}, 平均行距={avg_gap:.1f}px)"
        
        return False, ""
    
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
        
        # 判断条件 - 使用更激进的策略
        high_text_density = ink_density > self.text_density_threshold
        no_table_structure = structure_score < self.min_table_structure_score
        
        # 策略1: 高文字密度 + 无表格结构 = 肯定是文本
        if high_text_density and no_table_structure:
            reason = f"高文字密度({ink_density:.1%})且无表格结构(score={structure_score:.0f})"
            return True, reason
        
        # 策略2: 极高文字密度（>10%）即使有一些结构也可能是文本
        if ink_density > 0.10:
            # 检查结构分数是否远低于典型表格
            if structure_score < self.min_table_structure_score * 2:  # 2倍阈值
                reason = f"极高文字密度({ink_density:.1%})，结构分数不足(score={structure_score:.0f})"
                return True, reason
        
        # 策略3: 低结构分数（<阈值的50%）很可能不是表格
        if structure_score < self.min_table_structure_score * 0.5:
            if ink_density > 0.02:  # 有一定内容
                reason = f"结构分数极低(score={structure_score:.0f})，不像表格"
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

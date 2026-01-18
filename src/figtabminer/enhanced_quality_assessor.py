#!/usr/bin/env python3
"""
Enhanced Quality Assessor for v1.7.

Provides comprehensive quality assessment with:
- Multi-dimensional quality scoring (5 dimensions)
- Anomaly detection (oversized, undersized, extreme aspect ratios)
- Quality-based filtering
- Detailed issue reporting and recommendations
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from . import bbox_utils
from . import utils

logger = utils.setup_logging(__name__)


@dataclass
class Anomaly:
    """异常检测结果"""
    detection_index: int  # 检测结果索引
    anomaly_type: str  # 异常类型
    severity: str  # 严重程度: 'low', 'medium', 'high'
    description: str  # 描述
    metrics: Dict = field(default_factory=dict)  # 相关指标


class EnhancedQualityAssessor:
    """
    增强型质量评估器 v1.7
    
    提供多维度质量评分、异常检测和质量过滤
    """
    
    # 质量维度权重
    DEFAULT_WEIGHTS = {
        'detection_confidence': 0.30,  # 检测置信度
        'content_completeness': 0.25,  # 内容完整性
        'boundary_precision': 0.20,    # 边界精确度
        'caption_match': 0.15,         # 标题匹配度
        'position_reasonableness': 0.10  # 位置合理性
    }
    
    # 异常检测阈值
    ANOMALY_THRESHOLDS = {
        'max_page_ratio': 0.90,      # 最大页面占比
        'min_page_ratio': 0.001,     # 最小页面占比
        'max_aspect_ratio': 10.0,    # 最大纵横比
        'min_aspect_ratio': 0.1,     # 最小纵横比
        'min_ink_ratio': 0.005,      # 最小墨水比例
        'max_text_overlap': 0.7      # 最大文字重叠比例
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化增强型质量评估器
        
        Args:
            config: 配置字典，可包含：
                - weights: 各维度权重
                - anomaly_thresholds: 异常检测阈值
                - min_quality_score: 最小质量分数
        """
        self.config = config or {}
        
        # 质量维度权重
        self.weights = self.config.get('weights', self.DEFAULT_WEIGHTS.copy())
        
        # 异常检测阈值
        self.anomaly_thresholds = self.config.get(
            'anomaly_thresholds',
            self.ANOMALY_THRESHOLDS.copy()
        )
        
        # 最小质量分数
        self.min_quality_score = self.config.get('min_quality_score', 0.5)
        
        logger.info("EnhancedQualityAssessor initialized")
    
    def assess_comprehensive(
        self,
        detection: Dict,
        page_image: np.ndarray,
        captions: Optional[List[Dict]] = None,
        page_layout: Optional[Dict] = None
    ) -> Dict:
        """
        综合质量评估
        
        Args:
            detection: 检测结果字典，包含bbox, type, score等
            page_image: 页面图像
            captions: 标题列表（可选）
            page_layout: 页面布局信息（可选）
            
        Returns:
            QualityScore字典包含：
                - overall_score: 总体质量分数 (0-1)
                - dimension_scores: 各维度分数
                - issues: 识别的问题列表
                - recommendations: 改进建议列表
        """
        dimension_scores = {}
        issues = []
        recommendations = []
        
        # 1. 检测置信度
        dimension_scores['detection_confidence'] = detection.get('score', 0.5)
        if dimension_scores['detection_confidence'] < 0.5:
            issues.append("Low detection confidence")
            recommendations.append("Consider using ensemble detection or adjusting thresholds")
        
        # 2. 内容完整性
        dimension_scores['content_completeness'] = self._assess_content_completeness(
            detection['bbox'], page_image
        )
        if dimension_scores['content_completeness'] < 0.5:
            issues.append("Low content density - bbox may be too large or mostly empty")
            recommendations.append("Refine bbox boundaries to remove excess whitespace")
        
        # 3. 边界精确度
        dimension_scores['boundary_precision'] = self._assess_boundary_precision(
            detection['bbox'], page_image
        )
        if dimension_scores['boundary_precision'] < 0.5:
            issues.append("Poor boundary precision - bbox may not tightly fit content")
            recommendations.append("Apply boundary refinement algorithm")
        
        # 4. 标题匹配度
        if captions:
            dimension_scores['caption_match'] = self._assess_caption_match(
                detection['bbox'], captions
            )
            if dimension_scores['caption_match'] < 0.3:
                issues.append("No matching caption found nearby")
                recommendations.append("Verify this is a valid figure/table")
        else:
            dimension_scores['caption_match'] = 0.5  # 中性分数
        
        # 5. 位置合理性
        dimension_scores['position_reasonableness'] = self._assess_position_reasonableness(
            detection['bbox'], page_image.shape, page_layout
        )
        if dimension_scores['position_reasonableness'] < 0.5:
            issues.append("Unusual position - may be at page edge or in header/footer")
            recommendations.append("Check if this is a valid figure or page artifact")
        
        # 计算基础总体分数
        overall_score = sum(
            dimension_scores[dim] * self.weights[dim]
            for dim in dimension_scores
        )
        
        # 6. 文字区域检测（假阳性过滤）
        text_density = self._detect_text_region(detection['bbox'], page_image)
        if text_density > 0.7:
            issues.append(f"High text density ({text_density:.2f}) - likely a text region, not a figure/table")
            recommendations.append("Filter this detection as false positive")
            overall_score *= 0.2  # 大幅降低分数
            logger.warning(f"Detected text region with density {text_density:.2f}, reducing quality score")
        elif text_density > 0.5:
            issues.append(f"Moderate text density ({text_density:.2f}) - may contain significant text")
            overall_score *= 0.6  # 适度降低分数
        
        # 7. arXiv编号检测（特殊假阳性过滤）
        is_arxiv = self._detect_arxiv_id(detection['bbox'], page_image.shape, page_layout)
        if is_arxiv:
            issues.append("Detected arXiv ID region - common false positive on first page")
            recommendations.append("Filter this detection as arXiv ID false positive")
            overall_score *= 0.1  # 极大降低分数
            logger.warning("Detected arXiv ID region, strongly reducing quality score")
        
        quality_score = {
            'overall_score': overall_score,
            'dimension_scores': dimension_scores,
            'issues': issues,
            'recommendations': recommendations
        }
        
        logger.debug(f"Quality assessment: overall={overall_score:.3f}, "
                    f"issues={len(issues)}")
        
        return quality_score
    
    def _assess_content_completeness(self, bbox: List[float], page_image: np.ndarray) -> float:
        """
        评估内容完整性
        
        检查边界框是否包含有意义的内容（不是大部分空白）
        """
        try:
            x0, y0, x1, y1 = [int(c) for c in bbox]
            h, w = page_image.shape[:2]
            
            # 限制在图像边界内
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            
            if x1 <= x0 or y1 <= y0:
                return 0.0
            
            crop = page_image[y0:y1, x0:x1]
            if crop.size == 0:
                return 0.0
            
            # 转换为灰度
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            
            # 计算墨水比例（非白色像素）
            _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
            ink_ratio = np.count_nonzero(binary) / binary.size
            
            # 基于墨水比例评分
            min_ink = self.anomaly_thresholds['min_ink_ratio']
            if ink_ratio < min_ink:
                return 0.0  # 太空
            elif ink_ratio > 0.5:
                return 1.0  # 内容密集
            else:
                # 在min_ink和0.5之间线性缩放
                return (ink_ratio - min_ink) / (0.5 - min_ink)
        
        except Exception as e:
            logger.debug(f"Error assessing content completeness: {e}")
            return 0.5
    
    def _assess_boundary_precision(self, bbox: List[float], page_image: np.ndarray) -> float:
        """
        评估边界精确度
        
        检查边界框是否紧密贴合内容
        """
        try:
            x0, y0, x1, y1 = [int(c) for c in bbox]
            h, w = page_image.shape[:2]
            
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            
            if x1 <= x0 or y1 <= y0:
                return 0.0
            
            crop = page_image[y0:y1, x0:x1]
            if crop.size == 0:
                return 0.0
            
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
            
            # 找到内容边界
            coords = cv2.findNonZero(binary)
            if coords is None or len(coords) == 0:
                return 0.0
            
            content_x, content_y, content_w, content_h = cv2.boundingRect(coords)
            
            # 计算内容占边界框的比例
            content_area = content_w * content_h
            bbox_area = (x1 - x0) * (y1 - y0)
            
            if bbox_area == 0:
                return 0.0
            
            fill_ratio = content_area / bbox_area
            
            # 填充比例越高，边界越精确
            return min(1.0, fill_ratio)
        
        except Exception as e:
            logger.debug(f"Error assessing boundary precision: {e}")
            return 0.5
    
    def _assess_caption_match(self, bbox: List[float], captions: List[Dict]) -> float:
        """
        评估标题匹配度
        
        检查是否有匹配的标题
        """
        if not captions:
            return 0.5
        
        # 找到最近的标题
        min_distance = float('inf')
        
        for caption in captions:
            if 'bbox' not in caption:
                continue
            
            distance = bbox_utils.bbox_distance(bbox, caption['bbox'])
            min_distance = min(min_distance, distance)
        
        # 距离越近，匹配度越高
        if min_distance < 50:
            return 1.0
        elif min_distance < 100:
            return 0.8
        elif min_distance < 200:
            return 0.5
        else:
            return 0.2
    
    def _assess_position_reasonableness(
        self,
        bbox: List[float],
        page_shape: Tuple[int, int],
        page_layout: Optional[Dict]
    ) -> float:
        """
        评估位置合理性
        
        检查位置是否合理（不在页边缘等）
        """
        x0, y0, x1, y1 = bbox
        h, w = page_shape[:2]
        
        score = 1.0
        
        # 检查是否太靠近页面边缘
        margin = 20
        if x0 < margin or y0 < margin or x1 > w - margin or y1 > h - margin:
            score -= 0.3
        
        # 检查是否在页眉或页脚区域
        header_region = h * 0.1
        footer_region = h * 0.9
        
        if y0 < header_region or y1 > footer_region:
            score -= 0.2
        
        # 检查是否在页面中心区域（更合理）
        center_y = h / 2
        bbox_center_y = (y0 + y1) / 2
        distance_from_center = abs(bbox_center_y - center_y) / (h / 2)
        
        # 越靠近中心越好
        if distance_from_center < 0.5:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _detect_text_region(self, bbox: List[float], page_image: np.ndarray) -> float:
        """
        检测是否为文字区域（假阳性过滤）
        
        Args:
            bbox: 边界框
            page_image: 页面图像
            
        Returns:
            文字密度分数 (0-1)，越高越可能是纯文字区域
        """
        try:
            x0, y0, x1, y1 = [int(c) for c in bbox]
            h, w = page_image.shape[:2]
            
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            
            if x1 <= x0 or y1 <= y0:
                return 0.0
            
            crop = page_image[y0:y1, x0:x1]
            if crop.size == 0:
                return 0.0
            
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            
            text_score = 0.0
            
            # 1. 检测水平线密度（文字有很多水平笔画）
            edges = cv2.Canny(gray, 50, 150)
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
            h_line_density = np.count_nonzero(h_lines) / h_lines.size
            
            if h_line_density > 0.01:  # 有明显的水平线
                text_score += 0.3
            
            # 2. 检测规则间距（文字行间距规则）
            # 计算垂直方向的投影
            v_projection = np.sum(gray < 200, axis=1)  # 非白色像素的垂直投影
            
            # 查找峰值（文字行）
            try:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(v_projection, distance=10, height=crop.shape[1] * 0.1)
                if len(peaks) >= 3:  # 至少3行文字
                    # 检查间距是否规则
                    if len(peaks) > 1:
                        intervals = np.diff(peaks)
                        interval_std = np.std(intervals)
                        interval_mean = np.mean(intervals)
                        if interval_mean > 0 and interval_std / interval_mean < 0.3:  # 间距规则
                            text_score += 0.4
            except ImportError:
                # scipy不可用，使用简单的峰值检测
                # 查找投影中的局部最大值
                for i in range(10, len(v_projection) - 10):
                    if v_projection[i] > crop.shape[1] * 0.1:  # 足够的像素
                        if v_projection[i] > v_projection[i-5] and v_projection[i] > v_projection[i+5]:
                            # 简单的局部最大值
                            pass
            except Exception:
                pass
            
            # 3. 检测小连通组件（文字是小块）
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            total_area = crop.shape[0] * crop.shape[1]
            small_components = 0
            large_components = 0
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 10 < area < total_area * 0.01:  # 小组件（文字大小）
                    small_components += 1
                elif area > total_area * 0.1:  # 大组件（图表元素）
                    large_components += 1
            
            # 文字区域应该有很多小组件，很少大组件
            if small_components > 20 and large_components < 3:
                text_score += 0.3
            
            logger.debug(f"Text detection: h_line_density={h_line_density:.3f}, "
                        f"small_comp={small_components}, large_comp={large_components}, "
                        f"text_score={text_score:.2f}")
            
            return min(1.0, text_score)
        
        except Exception as e:
            logger.debug(f"Error detecting text region: {e}")
            return 0.0
    
    def _detect_arxiv_id(
        self,
        bbox: List[float],
        page_shape: Tuple[int, int],
        page_layout: Optional[Dict]
    ) -> bool:
        """
        检测是否为arXiv编号区域（第一页左上角的常见假阳性）
        
        arXiv编号特征：
        - 位于页面左上角（通常在前20%高度，左侧30%宽度）
        - 小尺寸（通常<5%页面面积）
        - 包含"arXiv"文本（如果有OCR）
        
        Args:
            bbox: 边界框
            page_shape: 页面形状
            page_layout: 页面布局信息
            
        Returns:
            是否为arXiv编号区域
        """
        x0, y0, x1, y1 = bbox
        h, w = page_shape[:2]
        
        # 1. 位置检查：左上角
        bbox_center_x = (x0 + x1) / 2
        bbox_center_y = (y0 + y1) / 2
        
        # 必须在页面左上角区域
        in_top_left = (
            bbox_center_x < w * 0.3 and  # 左侧30%
            bbox_center_y < h * 0.2       # 顶部20%
        )
        
        if not in_top_left:
            return False
        
        # 2. 尺寸检查：小尺寸
        bbox_area = (x1 - x0) * (y1 - y0)
        page_area = h * w
        area_ratio = bbox_area / page_area
        
        # arXiv编号通常很小
        if area_ratio > 0.05:  # 大于5%页面面积，不太可能是arXiv编号
            return False
        
        # 3. 纵横比检查：arXiv编号通常是横向的小矩形
        width = x1 - x0
        height = y1 - y0
        if height > 0:
            aspect_ratio = width / height
            # arXiv编号通常宽度>高度，但不会太极端
            if not (1.5 < aspect_ratio < 8.0):
                return False
        
        # 4. 如果有页面布局信息，检查是否在页眉区域
        if page_layout and 'header_region' in page_layout:
            # 在页眉区域的小框更可能是arXiv编号
            pass
        
        # 满足所有条件，很可能是arXiv编号
        logger.debug(f"Detected potential arXiv ID: position=({bbox_center_x/w:.2f}, {bbox_center_y/h:.2f}), "
                    f"area_ratio={area_ratio:.4f}, aspect_ratio={aspect_ratio:.2f}")
        
        return True
    
    def detect_anomalies(
        self,
        detections: List[Dict],
        page_image: np.ndarray
    ) -> List[Anomaly]:
        """
        检测异常检测结果
        
        Args:
            detections: 检测结果列表
            page_image: 页面图像
            
        Returns:
            异常列表
        """
        anomalies = []
        h, w = page_image.shape[:2]
        page_area = h * w
        
        for idx, detection in enumerate(detections):
            bbox = detection['bbox']
            x0, y0, x1, y1 = bbox
            bbox_area = (x1 - x0) * (y1 - y0)
            
            # 1. 检测过大的边界框
            page_ratio = bbox_area / page_area
            if page_ratio > self.anomaly_thresholds['max_page_ratio']:
                anomalies.append(Anomaly(
                    detection_index=idx,
                    anomaly_type='oversized',
                    severity='high',
                    description=f"Bbox covers {page_ratio*100:.1f}% of page (>{self.anomaly_thresholds['max_page_ratio']*100}%)",
                    metrics={'page_ratio': page_ratio}
                ))
            
            # 2. 检测过小的边界框
            if page_ratio < self.anomaly_thresholds['min_page_ratio']:
                anomalies.append(Anomaly(
                    detection_index=idx,
                    anomaly_type='undersized',
                    severity='medium',
                    description=f"Bbox covers only {page_ratio*100:.3f}% of page (<{self.anomaly_thresholds['min_page_ratio']*100}%)",
                    metrics={'page_ratio': page_ratio}
                ))
            
            # 3. 检测极端纵横比
            aspect_ratio = (x1 - x0) / (y1 - y0) if (y1 - y0) > 0 else 0
            if aspect_ratio > self.anomaly_thresholds['max_aspect_ratio']:
                anomalies.append(Anomaly(
                    detection_index=idx,
                    anomaly_type='extreme_aspect_ratio',
                    severity='medium',
                    description=f"Extreme aspect ratio {aspect_ratio:.2f} (>{self.anomaly_thresholds['max_aspect_ratio']})",
                    metrics={'aspect_ratio': aspect_ratio}
                ))
            elif aspect_ratio < self.anomaly_thresholds['min_aspect_ratio'] and aspect_ratio > 0:
                anomalies.append(Anomaly(
                    detection_index=idx,
                    anomaly_type='extreme_aspect_ratio',
                    severity='medium',
                    description=f"Extreme aspect ratio {aspect_ratio:.2f} (<{self.anomaly_thresholds['min_aspect_ratio']})",
                    metrics={'aspect_ratio': aspect_ratio}
                ))
            
            # 4. 检测内容过于稀疏
            try:
                x0_int, y0_int = max(0, int(x0)), max(0, int(y0))
                x1_int, y1_int = min(w, int(x1)), min(h, int(y1))
                
                if x1_int > x0_int and y1_int > y0_int:
                    crop = page_image[y0_int:y1_int, x0_int:x1_int]
                    if crop.size > 0:
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
                        _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
                        ink_ratio = np.count_nonzero(binary) / binary.size
                        
                        if ink_ratio < self.anomaly_thresholds['min_ink_ratio']:
                            anomalies.append(Anomaly(
                                detection_index=idx,
                                anomaly_type='sparse_content',
                                severity='high',
                                description=f"Very sparse content (ink ratio: {ink_ratio*100:.2f}% <{self.anomaly_thresholds['min_ink_ratio']*100}%)",
                                metrics={'ink_ratio': ink_ratio}
                            ))
            except Exception as e:
                logger.debug(f"Error checking content density for detection {idx}: {e}")
        
        logger.info(f"Detected {len(anomalies)} anomalies in {len(detections)} detections")
        
        return anomalies
    
    def filter_by_quality(
        self,
        detections: List[Dict],
        page_image: Optional[np.ndarray] = None,
        captions: Optional[List[Dict]] = None,
        min_score: Optional[float] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        按质量过滤检测结果
        
        Args:
            detections: 检测结果列表
            page_image: 页面图像（可选）
            captions: 标题列表（可选）
            min_score: 最小质量分数（可选，默认使用配置值）
            
        Returns:
            (通过的检测列表, 被过滤的检测列表)
        """
        if min_score is None:
            min_score = self.min_quality_score
        
        passed = []
        filtered = []
        
        for detection in detections:
            # 如果已有质量分数，使用它
            if 'quality_score' in detection and isinstance(detection['quality_score'], dict):
                quality_score = detection['quality_score']['overall_score']
            elif page_image is not None:
                # 否则进行评估
                quality_result = self.assess_comprehensive(
                    detection, page_image, captions
                )
                detection['quality_score'] = quality_result
                quality_score = quality_result['overall_score']
            else:
                # 无法评估，使用检测置信度
                quality_score = detection.get('score', 0.5)
            
            # 过滤
            if quality_score >= min_score:
                passed.append(detection)
            else:
                filtered.append(detection)
                logger.debug(f"Filtered detection with quality score {quality_score:.3f} < {min_score}")
        
        logger.info(f"Quality filtering: {len(detections)} -> {len(passed)} "
                   f"(filtered {len(filtered)})")
        
        return passed, filtered
    
    def detect_table_features(
        self,
        bbox: List[float],
        page_image: np.ndarray
    ) -> Dict[str, float]:
        """
        检测表格特征（用于fig/table区分和三线表检测）
        
        Args:
            bbox: 边界框
            page_image: 页面图像
            
        Returns:
            表格特征字典：
                - has_grid: 是否有网格线
                - has_horizontal_lines: 是否有水平线
                - has_vertical_lines: 是否有垂直线
                - line_count: 线条数量
                - cell_structure: 是否有单元格结构
                - is_simple_table: 是否为简单表格（三线表）
        """
        try:
            x0, y0, x1, y1 = [int(c) for c in bbox]
            h, w = page_image.shape[:2]
            
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            
            if x1 <= x0 or y1 <= y0:
                return {}
            
            crop = page_image[y0:y1, x0:x1]
            if crop.size == 0:
                return {}
            
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            
            features = {}
            
            # 1. 检测线条
            edges = cv2.Canny(gray, 50, 150)
            
            # 检测水平线
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
            h_line_pixels = np.count_nonzero(h_lines)
            features['has_horizontal_lines'] = h_line_pixels > 100
            
            # 检测垂直线
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
            v_line_pixels = np.count_nonzero(v_lines)
            features['has_vertical_lines'] = v_line_pixels > 100
            
            # 网格检测
            features['has_grid'] = features['has_horizontal_lines'] and features['has_vertical_lines']
            
            # 2. 使用HoughLinesP检测线条数量
            # 降低阈值以提高敏感度
            min_line_length = int(crop.shape[1] * 0.2)  # 从0.3降到0.2
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=min_line_length, maxLineGap=10)
            
            if lines is not None:
                # 分类水平线和垂直线
                h_lines_count = 0
                v_lines_count = 0
                
                for line in lines:
                    x1_l, y1_l, x2_l, y2_l = line[0]
                    angle = np.abs(np.arctan2(y2_l - y1_l, x2_l - x1_l))
                    
                    if angle < 0.1 or angle > 3.04:  # 接近水平
                        h_lines_count += 1
                    elif 1.47 < angle < 1.67:  # 接近垂直
                        v_lines_count += 1
                
                features['horizontal_line_count'] = h_lines_count
                features['vertical_line_count'] = v_lines_count
                features['line_count'] = len(lines)
                
                logger.debug(f"detect_table_features: Found {h_lines_count} h-lines, {v_lines_count} v-lines, "
                            f"h_pixels={h_line_pixels}, v_pixels={v_line_pixels}")
                
                # 3. 三线表检测
                # 三线表特征：3条水平线（顶线、中线、底线），很少或没有垂直线
                # 注意：HoughLinesP可能会检测到多条线（线条的边缘等），所以放宽条件
                features['is_simple_table'] = (
                    2 <= h_lines_count <= 10 and  # 2-10条水平线（放宽从4到10）
                    v_lines_count <= 3 and         # 很少垂直线（放宽从2到3）
                    h_line_pixels > 100 and        # 足够的水平线像素
                    features['has_horizontal_lines']  # 确保检测到水平线
                )
                
                if features['is_simple_table']:
                    logger.debug(f"Detected simple table (three-line): h_lines={h_lines_count}, "
                                f"v_lines={v_lines_count}")
            else:
                logger.debug(f"detect_table_features: No lines detected by HoughLinesP")
                features['line_count'] = 0
                features['horizontal_line_count'] = 0
                features['vertical_line_count'] = 0
                features['is_simple_table'] = False
            
            # 4. 单元格结构检测
            # 如果有网格，尝试检测单元格
            if features['has_grid']:
                # 简单的单元格检测：查找矩形轮廓
                contours, _ = cv2.findContours(
                    cv2.bitwise_or(h_lines, v_lines),
                    cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                rect_count = 0
                for cnt in contours:
                    peri = cv2.arcLength(cnt, True)
                    if peri > 0:
                        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                        if len(approx) == 4:  # 矩形
                            rect_count += 1
                
                features['cell_structure'] = rect_count >= 4  # 至少4个单元格
            else:
                features['cell_structure'] = False
            
            return features
        
        except Exception as e:
            logger.debug(f"Error detecting table features: {e}")
            return {}


# 便捷函数
def assess_detection_quality(
    detection: Dict,
    page_image: np.ndarray,
    captions: Optional[List[Dict]] = None,
    config: Optional[Dict] = None
) -> Dict:
    """
    评估单个检测结果的质量
    
    Args:
        detection: 检测结果
        page_image: 页面图像
        captions: 标题列表
        config: 配置字典
        
    Returns:
        质量评分字典
    """
    assessor = EnhancedQualityAssessor(config)
    return assessor.assess_comprehensive(detection, page_image, captions)


def filter_detections_by_quality(
    detections: List[Dict],
    page_image: np.ndarray,
    min_score: float = 0.5,
    config: Optional[Dict] = None
) -> List[Dict]:
    """
    过滤低质量检测结果
    
    Args:
        detections: 检测结果列表
        page_image: 页面图像
        min_score: 最小质量分数
        config: 配置字典
        
    Returns:
        过滤后的检测结果列表
    """
    assessor = EnhancedQualityAssessor(config)
    passed, _ = assessor.filter_by_quality(detections, page_image, min_score=min_score)
    return passed


# 便捷函数：检测简单表格
def detect_simple_tables(
    page_image: np.ndarray,
    existing_detections: Optional[List[Dict]] = None
) -> List[Dict]:
    """
    检测简单表格（三线表）
    
    这是一个补充检测函数，用于发现被主检测器遗漏的简单表格
    
    Args:
        page_image: 页面图像
        existing_detections: 已有的检测结果（用于避免重复）
        
    Returns:
        简单表格检测结果列表
    """
    simple_tables = []
    
    # 在页面上搜索可能的表格区域
    # 策略：查找有3条水平线的区域
    
    gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY) if len(page_image.shape) == 3 else page_image
    edges = cv2.Canny(gray, 50, 150)
    
    # 检测长水平线 - 降低阈值以提高敏感度
    # minLineLength设为页面宽度的20%（之前是30%）
    min_line_length = int(page_image.shape[1] * 0.2)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=min_line_length, maxLineGap=20)
    
    logger.debug(f"detect_simple_tables: HoughLinesP detected {len(lines) if lines is not None else 0} lines "
                f"(min_length={min_line_length})")
    
    if lines is None or len(lines) < 3:
        logger.debug(f"detect_simple_tables: Not enough lines detected (found {len(lines) if lines is not None else 0})")
        return simple_tables
    
    # 筛选水平线
    h_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
        if angle < 0.1 or angle > 3.04:  # 接近水平
            h_lines.append((min(y1, y2), min(x1, x2), max(x1, x2)))  # (y, x_start, x_end)
    
    logger.debug(f"detect_simple_tables: Found {len(h_lines)} horizontal lines")
    
    if len(h_lines) < 3:
        return simple_tables
    
    # 按y坐标排序
    h_lines.sort(key=lambda l: l[0])
    
    # 合并相近的水平线（距离<10像素的认为是同一条线）
    merged_h_lines = []
    i = 0
    while i < len(h_lines):
        y, x_start, x_end = h_lines[i]
        # 查找所有y坐标相近的线
        j = i + 1
        while j < len(h_lines) and h_lines[j][0] - y < 10:
            # 合并：取平均y坐标，扩展x范围
            y_next, x_start_next, x_end_next = h_lines[j]
            y = (y + y_next) // 2
            x_start = min(x_start, x_start_next)
            x_end = max(x_end, x_end_next)
            j += 1
        merged_h_lines.append((y, x_start, x_end))
        i = j
    
    logger.debug(f"detect_simple_tables: After merging, {len(merged_h_lines)} horizontal lines")
    
    if len(merged_h_lines) < 3:
        return simple_tables
    
    # 查找三线表模式：3条线，间距合理
    for i in range(len(merged_h_lines) - 2):
        y1, x1_start, x1_end = merged_h_lines[i]
        y2, x2_start, x2_end = merged_h_lines[i + 1]
        y3, x3_start, x3_end = merged_h_lines[i + 2]
        
        # 检查间距是否合理（20-300像素）
        gap1 = y2 - y1
        gap2 = y3 - y2
        
        logger.debug(f"detect_simple_tables: Checking line group at y={y1},{y2},{y3}, gaps={gap1},{gap2}")
        
        if 20 < gap1 < 300 and 20 < gap2 < 300:
            # 可能是三线表
            x_min = min(x1_start, x2_start, x3_start)
            x_max = max(x1_end, x2_end, x3_end)
            y_min = y1 - 10  # 留一点边距
            y_max = y3 + 10
            
            # 确保边界在图像范围内
            y_min = max(0, y_min)
            y_max = min(page_image.shape[0], y_max)
            x_min = max(0, x_min)
            x_max = min(page_image.shape[1], x_max)
            
            # 创建检测结果
            bbox = [float(x_min), float(y_min), float(x_max), float(y_max)]
            
            # 检查是否与已有检测重叠
            is_duplicate = False
            if existing_detections:
                for det in existing_detections:
                    existing_bbox = det.get('bbox', [])
                    if existing_bbox:
                        iou = bbox_utils.bbox_iou(bbox, existing_bbox)
                        if iou > 0.5:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                simple_tables.append({
                    'bbox': bbox,
                    'type': 'table',
                    'score': 0.7,  # 中等置信度
                    'detector': 'simple_table_detector',
                    'method': 'three_line_detection'
                })
                
                logger.info(f"Detected simple table at y={y_min:.0f}-{y_max:.0f}, x={x_min:.0f}-{x_max:.0f}")
    
    logger.info(f"detect_simple_tables: Found {len(simple_tables)} simple tables")
    return simple_tables

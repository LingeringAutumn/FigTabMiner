#!/usr/bin/env python3
"""
Enhanced Chart Classifier for v1.7.

Implements hierarchical classification with 15+ chart types:
- Charts: bar_chart, pie_chart, line_plot, scatter_plot, histogram, box_plot, violin_plot
- Microscopy: sem, tem, optical
- Diagrams: flowchart, schematic
- Others: heatmap, spectrum, photo, unknown

Features:
- Hierarchical classification (main category → sub category)
- Enhanced visual feature extraction
- Confidence calibration using Platt scaling
- Multi-modal fusion (keywords + visual + context)
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
from . import utils
from .chart_classifier import ChartClassifier

logger = utils.setup_logging(__name__)


@dataclass
class HierarchicalClassification:
    """层次化分类结果"""
    main_category: str  # 主类别: 'chart', 'microscopy', 'diagram', 'photo'
    sub_category: str  # 子类别
    confidence_by_level: Dict[str, float]  # 各层级置信度
    all_scores: Dict[str, float] = field(default_factory=dict)  # 所有类型的分数
    debug_info: Dict = field(default_factory=dict)  # 调试信息


class EnhancedChartClassifier:
    """
    增强型图表分类器 v1.7
    
    支持15+种图表类型的层次化分类
    """
    
    # 扩展的图表类型（15+种）
    CHART_TYPES = [
        # Charts (7种)
        'bar_chart', 'pie_chart', 'line_plot', 'scatter_plot',
        'histogram', 'box_plot', 'violin_plot',
        # Microscopy (3种)
        'microscopy_sem', 'microscopy_tem', 'microscopy_optical',
        # Diagrams (2种)
        'diagram_flowchart', 'diagram_schematic',
        # Others (3种)
        'heatmap', 'spectrum', 'photo',
        # Fallback
        'unknown'
    ]
    
    # 层次化分类结构
    HIERARCHY = {
        'chart': ['bar_chart', 'pie_chart', 'line_plot', 'scatter_plot',
                 'histogram', 'box_plot', 'violin_plot', 'heatmap'],
        'microscopy': ['microscopy_sem', 'microscopy_tem', 'microscopy_optical'],
        'diagram': ['diagram_flowchart', 'diagram_schematic'],
        'photo': ['photo'],
        'spectrum': ['spectrum']
    }
    
    # 扩展的关键词模式
    CHART_KEYWORDS = {
        'bar_chart': ['bar chart', 'bar graph', 'column chart', 'bar plot'],
        'pie_chart': ['pie chart', 'pie graph', 'donut chart', 'circular chart'],
        'line_plot': ['line chart', 'line graph', 'line plot', 'curve', 'time series'],
        'scatter_plot': ['scatter plot', 'scatter chart', 'scatter diagram'],
        'histogram': ['histogram', 'distribution', 'frequency'],
        'box_plot': ['box plot', 'box chart', 'box-and-whisker', 'boxplot'],
        'violin_plot': ['violin plot', 'violin chart'],
        'heatmap': ['heat map', 'heatmap', 'color map', 'intensity map'],
        'microscopy_sem': ['sem', 'scanning electron', 'electron microscope'],
        'microscopy_tem': ['tem', 'transmission electron'],
        'microscopy_optical': ['optical microscope', 'light microscope', 'micrograph'],
        'diagram_flowchart': ['flowchart', 'flow chart', 'flow diagram', 'process diagram'],
        'diagram_schematic': ['schematic', 'circuit', 'wiring diagram', 'block diagram'],
        'spectrum': ['spectrum', 'spectroscopy', 'spectra', 'wavelength'],
        'photo': ['photograph', 'photo', 'image', 'picture']
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化增强型图表分类器
        
        Args:
            config: 配置字典，包含：
                - enable_visual_analysis: 是否启用视觉分析
                - enable_hierarchical: 是否启用层次化分类
                - enable_calibration: 是否启用置信度校准
                - visual_weight: 视觉特征权重
                - keyword_weight: 关键词权重
                - context_weight: 上下文权重
        """
        self.config = config or {}
        self.enable_visual = self.config.get('enable_visual_analysis', True)
        self.enable_hierarchical = self.config.get('enable_hierarchical', True)
        self.enable_calibration = self.config.get('enable_calibration', True)
        
        # 权重配置
        self.visual_weight = self.config.get('visual_weight', 0.5)
        self.keyword_weight = self.config.get('keyword_weight', 0.3)
        self.context_weight = self.config.get('context_weight', 0.2)
        
        # 置信度校准参数（Platt scaling）
        # 这些参数应该通过训练数据学习，这里使用默认值
        self.calibration_a = self.config.get('calibration_a', 1.0)
        self.calibration_b = self.config.get('calibration_b', 0.0)
        
        logger.info(f"EnhancedChartClassifier initialized with {len(self.CHART_TYPES)} types")
    
    def classify(
        self,
        image_path: str,
        caption: str = "",
        snippet: str = "",
        ocr_text: str = "",
        detection_metadata: Optional[Dict] = None
    ) -> Tuple[str, float, List[str], Dict]:
        """
        分类图表类型
        
        Args:
            image_path: 图表图像路径
            caption: 标题文本
            snippet: 片段文本
            ocr_text: OCR文本
            detection_metadata: 检测元数据（可选）
            
        Returns:
            (chart_type, confidence, matched_keywords, debug_info)
        """
        if self.enable_hierarchical:
            # 使用层次化分类
            result = self.classify_hierarchical(
                image_path, caption, snippet, ocr_text, detection_metadata
            )
            return (
                result.sub_category,
                result.confidence_by_level.get('sub', 0.5),
                self._get_matched_keywords(result.sub_category, caption, snippet, ocr_text),
                result.debug_info
            )
        else:
            # 使用扁平分类
            return self._classify_flat(image_path, caption, snippet, ocr_text, detection_metadata)
    
    def classify_hierarchical(
        self,
        image_path: str,
        caption: str = "",
        snippet: str = "",
        ocr_text: str = "",
        detection_metadata: Optional[Dict] = None
    ) -> HierarchicalClassification:
        """
        层次化分类（先分大类，再分子类）
        
        Args:
            image_path: 图表图像路径
            caption: 标题文本
            snippet: 片段文本
            ocr_text: OCR文本
            detection_metadata: 检测元数据
            
        Returns:
            HierarchicalClassification对象
        """
        # 合并所有文本
        text_combined = f"{caption} {snippet} {ocr_text}".lower()
        
        # 第一层：分主类别
        main_scores = self._classify_main_category(image_path, text_combined, detection_metadata)
        main_category = max(main_scores, key=main_scores.get)
        main_confidence = main_scores[main_category]
        
        # 第二层：在主类别内分子类别
        sub_scores = self._classify_sub_category(
            image_path, text_combined, main_category, detection_metadata
        )
        sub_category = max(sub_scores, key=sub_scores.get) if sub_scores else 'unknown'
        sub_confidence = sub_scores.get(sub_category, 0.0)
        
        # 校准置信度
        if self.enable_calibration:
            main_confidence = self._calibrate_confidence(main_confidence)
            sub_confidence = self._calibrate_confidence(sub_confidence)
        
        # 构建结果
        result = HierarchicalClassification(
            main_category=main_category,
            sub_category=sub_category,
            confidence_by_level={
                'main': main_confidence,
                'sub': sub_confidence
            },
            all_scores={**main_scores, **sub_scores},
            debug_info={
                'method': 'hierarchical_v1.7',
                'main_scores': main_scores,
                'sub_scores': sub_scores
            }
        )
        
        logger.debug(f"Hierarchical classification: {main_category} -> {sub_category} "
                    f"(main_conf={main_confidence:.2f}, sub_conf={sub_confidence:.2f})")
        
        return result
    
    def _classify_main_category(
        self,
        image_path: str,
        text: str,
        metadata: Optional[Dict]
    ) -> Dict[str, float]:
        """
        第一层分类：区分chart、microscopy、diagram、photo、spectrum
        
        Returns:
            各主类别的分数字典
        """
        scores = {
            'chart': 0.0,
            'microscopy': 0.0,
            'diagram': 0.0,
            'photo': 0.0,
            'spectrum': 0.0
        }
        
        # 1. 关键词特征
        keyword_scores = self._get_main_category_keyword_scores(text)
        
        # 2. 视觉特征
        visual_scores = {}
        if self.enable_visual:
            try:
                visual_scores = self._get_main_category_visual_scores(image_path)
            except Exception as e:
                logger.warning(f"Visual analysis failed: {e}")
                visual_scores = {k: 0.0 for k in scores.keys()}
        else:
            visual_scores = {k: 0.0 for k in scores.keys()}
        
        # 3. 上下文特征
        context_scores = self._get_context_scores(metadata) if metadata else {k: 0.0 for k in scores.keys()}
        
        # 4. 融合分数
        for category in scores.keys():
            scores[category] = (
                self.keyword_weight * keyword_scores.get(category, 0.0) +
                self.visual_weight * visual_scores.get(category, 0.0) +
                self.context_weight * context_scores.get(category, 0.0)
            )
        
        # 归一化
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def _classify_sub_category(
        self,
        image_path: str,
        text: str,
        main_category: str,
        metadata: Optional[Dict]
    ) -> Dict[str, float]:
        """
        第二层分类：在主类别内细分子类型
        
        Returns:
            各子类别的分数字典
        """
        # 获取该主类别下的所有子类型
        sub_types = self.HIERARCHY.get(main_category, [])
        if not sub_types:
            return {'unknown': 1.0}
        
        scores = {st: 0.0 for st in sub_types}
        
        # 1. 关键词匹配
        for sub_type in sub_types:
            keywords = self.CHART_KEYWORDS.get(sub_type, [])
            for kw in keywords:
                if kw in text:
                    scores[sub_type] += 1.0
        
        # 2. 视觉特征（针对子类型）
        if self.enable_visual:
            try:
                visual_scores = self._get_sub_category_visual_scores(image_path, main_category)
                for sub_type in sub_types:
                    if sub_type in visual_scores:
                        scores[sub_type] += visual_scores[sub_type] * 2.0  # 视觉特征权重更高
            except Exception as e:
                logger.warning(f"Sub-category visual analysis failed: {e}")
        
        # 归一化
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        else:
            # 如果没有任何匹配，返回均匀分布
            scores = {k: 1.0 / len(scores) for k in scores.keys()}
        
        return scores
    
    def _get_main_category_keyword_scores(self, text: str) -> Dict[str, float]:
        """基于关键词获取主类别分数"""
        scores = {
            'chart': 0.0,
            'microscopy': 0.0,
            'diagram': 0.0,
            'photo': 0.0,
            'spectrum': 0.0
        }
        
        # Diagram关键词（优先检查，权重更高）
        diagram_kws = ['flowchart', 'flow chart', 'schematic', 'diagram', 'circuit', 'illustration', 'block diagram']
        for kw in diagram_kws:
            if kw in text:
                scores['diagram'] += 2.0  # 更高权重
        
        # Chart关键词（排除已被diagram匹配的）
        if scores['diagram'] == 0:
            chart_kws = ['chart', 'graph', 'plot', 'histogram', 'distribution']
            for kw in chart_kws:
                if kw in text:
                    scores['chart'] += 1.0
        
        # Microscopy关键词
        micro_kws = ['microscopy', 'micrograph', 'sem', 'tem', 'afm', 'electron', 'optical microscope']
        for kw in micro_kws:
            if kw in text:
                scores['microscopy'] += 1.0
        
        # Photo关键词
        photo_kws = ['photograph', 'photo', 'image', 'picture', 'snapshot']
        for kw in photo_kws:
            if kw in text:
                scores['photo'] += 1.0
        
        # Spectrum关键词
        spectrum_kws = ['spectrum', 'spectroscopy', 'spectra', 'wavelength', 'absorption', 'emission']
        for kw in spectrum_kws:
            if kw in text:
                scores['spectrum'] += 1.0
        
        # 归一化
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def _get_main_category_visual_scores(self, image_path: str) -> Dict[str, float]:
        """基于视觉特征获取主类别分数"""
        img = cv2.imread(image_path)
        if img is None:
            return {k: 0.0 for k in ['chart', 'microscopy', 'diagram', 'photo', 'spectrum']}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        scores = {
            'chart': self._detect_chart_features(gray),
            'microscopy': self._detect_microscopy_features(gray, img),
            'diagram': self._detect_diagram_features(gray),
            'photo': self._detect_photo_features(gray, img),
            'spectrum': self._detect_spectrum_features(gray)
        }
        
        return scores
    
    def _detect_chart_features(self, gray: np.ndarray) -> float:
        """检测图表特征：坐标轴、网格、数据点"""
        score = 0.0
        
        # 检测直线（坐标轴）
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        if lines is not None and len(lines) >= 2:
            score += 0.3
        
        # 检测规则结构（网格）
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
        if np.count_nonzero(h_lines) > 100 and np.count_nonzero(v_lines) > 100:
            score += 0.3
        
        # 检测数据区域（非白色区域）
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        data_ratio = np.count_nonzero(thresh) / thresh.size
        if 0.1 < data_ratio < 0.7:
            score += 0.4
        
        return min(1.0, score)
    
    def _detect_microscopy_features(self, gray: np.ndarray, img: np.ndarray) -> float:
        """检测显微镜图像特征：高纹理、低白色比例、灰度分布"""
        score = 0.0
        
        # 高纹理（Laplacian方差）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        if variance > 300:
            score += 0.4
        
        # 低白色比例
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        white_ratio = np.sum(thresh == 255) / thresh.size
        if white_ratio < 0.2:
            score += 0.3
        
        # 灰度分布（显微镜图像通常有丰富的灰度层次）
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        if entropy > 5.0:
            score += 0.3
        
        return min(1.0, score)
    
    def _detect_diagram_features(self, gray: np.ndarray) -> float:
        """检测示意图特征：多个形状、连接线、适度白空间"""
        score = 0.0
        
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 多个形状
        shape_count = len([c for c in contours if cv2.contourArea(c) > 100])
        if 3 <= shape_count <= 20:
            score += 0.4
        
        # 连接线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=10)
        if lines is not None and len(lines) >= 5:
            score += 0.3
        
        # 适度白空间
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        white_ratio = np.sum(thresh == 255) / thresh.size
        if 0.3 < white_ratio < 0.7:
            score += 0.3
        
        return min(1.0, score)
    
    def _detect_photo_features(self, gray: np.ndarray, img: np.ndarray) -> float:
        """检测照片特征：丰富色彩、自然纹理、无规则结构"""
        score = 0.0
        
        # 色彩丰富度
        if len(img.shape) == 3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            mean_sat = np.mean(saturation)
            if mean_sat > 50:
                score += 0.3
        
        # 无明显规则结构（少量直线）
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0
        if line_count < 5:
            score += 0.4
        
        # 自然纹理（中等方差）
        variance = gray.var()
        if 500 < variance < 3000:
            score += 0.3
        
        return min(1.0, score)
    
    def _detect_spectrum_features(self, gray: np.ndarray) -> float:
        """检测光谱图特征：连续曲线、峰值、横轴"""
        score = 0.0
        
        # 检测连续曲线
        edges = cv2.Canny(gray, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=5)
        if lines is not None and len(lines) >= 10:
            score += 0.4
        
        # 检测峰值（局部最大值）
        # 简化：检测垂直方向的变化
        dy = np.diff(gray, axis=0)
        peaks = np.sum(np.abs(dy) > 20)
        if peaks > 100:
            score += 0.3
        
        # 检测横轴（底部的水平线）
        bottom_region = gray[-int(gray.shape[0] * 0.2):, :]
        h_edges = cv2.Canny(bottom_region, 50, 150)
        h_lines = cv2.HoughLinesP(h_edges, 1, np.pi/180, 30, minLineLength=50, maxLineGap=10)
        if h_lines is not None and len(h_lines) >= 1:
            score += 0.3
        
        return min(1.0, score)
    
    def _get_sub_category_visual_scores(
        self,
        image_path: str,
        main_category: str
    ) -> Dict[str, float]:
        """获取子类别的视觉特征分数"""
        img = cv2.imread(image_path)
        if img is None:
            return {}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if main_category == 'chart':
            return self._classify_chart_subtypes(gray)
        elif main_category == 'microscopy':
            return self._classify_microscopy_subtypes(gray, img)
        elif main_category == 'diagram':
            return self._classify_diagram_subtypes(gray)
        else:
            return {}
    
    def _classify_chart_subtypes(self, gray: np.ndarray) -> Dict[str, float]:
        """细分图表子类型"""
        scores = {}
        
        # Bar chart: 矩形检测
        scores['bar_chart'] = self._detect_rectangles(gray)
        
        # Pie chart: 圆形检测
        scores['pie_chart'] = self._detect_circles(gray)
        
        # Line plot: 连续曲线
        scores['line_plot'] = self._detect_continuous_lines(gray)
        
        # Scatter plot: 散点
        scores['scatter_plot'] = self._detect_scattered_points(gray)
        
        # Histogram: 类似bar chart但更密集
        scores['histogram'] = self._detect_histogram(gray)
        
        # Box plot: 特殊的箱型结构
        scores['box_plot'] = self._detect_box_plot(gray)
        
        # Violin plot: 类似box plot但有曲线轮廓
        scores['violin_plot'] = self._detect_violin_plot(gray)
        
        # Heatmap: 网格+颜色
        scores['heatmap'] = self._detect_heatmap(gray)
        
        return scores
    
    def _classify_microscopy_subtypes(self, gray: np.ndarray, img: np.ndarray) -> Dict[str, float]:
        """细分显微镜子类型"""
        scores = {}
        
        # SEM: 高对比度、3D感、灰度图
        scores['microscopy_sem'] = self._detect_sem(gray)
        
        # TEM: 高分辨率、黑白、晶格结构
        scores['microscopy_tem'] = self._detect_tem(gray)
        
        # Optical: 彩色、细胞结构
        scores['microscopy_optical'] = self._detect_optical(gray, img)
        
        return scores
    
    def _classify_diagram_subtypes(self, gray: np.ndarray) -> Dict[str, float]:
        """细分示意图子类型"""
        scores = {}
        
        # Flowchart: 多个框+箭头
        scores['diagram_flowchart'] = self._detect_flowchart(gray)
        
        # Schematic: 电路符号、连接线
        scores['diagram_schematic'] = self._detect_schematic(gray)
        
        return scores
    
    # 复用ChartClassifier的检测方法
    def _detect_rectangles(self, gray: np.ndarray) -> float:
        """检测矩形（bar chart）"""
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rect_count = 0
        total_area = gray.shape[0] * gray.shape[1]
        
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
            if len(approx) == 4:
                area = cv2.contourArea(cnt)
                if 0.001 * total_area < area < 0.3 * total_area:
                    rect_count += 1
        
        return min(1.0, rect_count / 10.0) if rect_count >= 3 else 0.0
    
    def _detect_circles(self, gray: np.ndarray) -> float:
        """检测圆形（pie chart）"""
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=100, param2=30, minRadius=20, maxRadius=min(gray.shape) // 2
        )
        return min(1.0, len(circles[0]) / 2.0) if circles is not None else 0.0
    
    def _detect_continuous_lines(self, gray: np.ndarray) -> float:
        """检测连续曲线（line plot）"""
        edges = cv2.Canny(gray, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        if lines is None:
            return 0.0
        
        curve_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
            if 0.1 < angle < 1.47 or 1.67 < angle < 3.04:
                curve_lines += 1
        
        return min(1.0, curve_lines / 20.0)
    
    def _detect_scattered_points(self, gray: np.ndarray) -> float:
        """检测散点（scatter plot）"""
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        point_count = 0
        total_area = gray.shape[0] * gray.shape[1]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 0.00001 * total_area < area < 0.001 * total_area:
                peri = cv2.arcLength(cnt, True)
                if peri > 0:
                    circularity = 4 * np.pi * area / (peri * peri)
                    if circularity > 0.5:
                        point_count += 1
        
        return min(1.0, point_count / 50.0) if point_count >= 10 else 0.0
    
    def _detect_histogram(self, gray: np.ndarray) -> float:
        """检测直方图（类似bar chart但更密集）"""
        rect_score = self._detect_rectangles(gray)
        
        # 直方图通常有更多的柱子
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect_count = sum(1 for cnt in contours if len(cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)) == 4)
        
        if rect_count > 10:
            return min(1.0, rect_score * 1.2)
        return rect_score * 0.5
    
    def _detect_box_plot(self, gray: np.ndarray) -> float:
        """检测箱线图"""
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=5)
        
        if lines is None:
            return 0.0
        
        # 箱线图有垂直线和水平线
        h_lines = sum(1 for line in lines if abs(line[0][1] - line[0][3]) < 10)
        v_lines = sum(1 for line in lines if abs(line[0][0] - line[0][2]) < 10)
        
        if h_lines >= 3 and v_lines >= 2:
            return min(1.0, (h_lines + v_lines) / 15.0)
        return 0.0
    
    def _detect_violin_plot(self, gray: np.ndarray) -> float:
        """检测小提琴图"""
        # 小提琴图有曲线轮廓
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        smooth_contours = 0
        for cnt in contours:
            if len(cnt) > 20:  # 足够多的点形成平滑曲线
                smooth_contours += 1
        
        return min(1.0, smooth_contours / 5.0) if smooth_contours >= 2 else 0.0
    
    def _detect_heatmap(self, gray: np.ndarray) -> float:
        """检测热图"""
        edges = cv2.Canny(gray, 50, 150)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
        
        grid = cv2.bitwise_and(h_lines, v_lines)
        grid_pixels = np.count_nonzero(grid)
        total_pixels = gray.shape[0] * gray.shape[1]
        grid_ratio = grid_pixels / total_pixels
        
        if 0.001 < grid_ratio < 0.1:
            return min(1.0, grid_ratio * 50)
        return 0.0
    
    def _detect_sem(self, gray: np.ndarray) -> float:
        """检测SEM图像：高对比度、3D感"""
        score = 0.0
        
        # 高对比度
        contrast = gray.max() - gray.min()
        if contrast > 150:
            score += 0.5
        
        # 3D感（边缘丰富）
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size
        if 0.05 < edge_ratio < 0.3:
            score += 0.5
        
        return min(1.0, score)
    
    def _detect_tem(self, gray: np.ndarray) -> float:
        """检测TEM图像：高分辨率、晶格结构"""
        score = 0.0
        
        # 高纹理细节
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        if variance > 500:
            score += 0.5
        
        # 周期性结构（FFT）
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        # 检测是否有明显的频率峰
        if magnitude.max() > magnitude.mean() * 10:
            score += 0.5
        
        return min(1.0, score)
    
    def _detect_optical(self, gray: np.ndarray, img: np.ndarray) -> float:
        """检测光学显微镜图像：彩色、细胞结构"""
        score = 0.0
        
        # 彩色图像
        if len(img.shape) == 3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            if np.mean(saturation) > 30:
                score += 0.5
        
        # 圆形结构（细胞）
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=5, maxRadius=100
        )
        if circles is not None and len(circles[0]) >= 3:
            score += 0.5
        
        return min(1.0, score)
    
    def _detect_flowchart(self, gray: np.ndarray) -> float:
        """检测流程图：多个框+连接线"""
        score = 0.0
        
        edges = cv2.Canny(gray, 50, 150)
        
        # 检测矩形框
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect_count = 0
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 4:
                    rect_count += 1
        
        if rect_count >= 3:
            score += 0.5
        
        # 检测连接线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=10)
        if lines is not None and len(lines) >= 5:
            score += 0.5
        
        return min(1.0, score)
    
    def _detect_schematic(self, gray: np.ndarray) -> float:
        """检测电路图：复杂连接、符号"""
        score = 0.0
        
        edges = cv2.Canny(gray, 50, 150)
        
        # 大量细线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=10, maxLineGap=5)
        if lines is not None and len(lines) > 20:
            score += 0.5
        
        # 多个小符号
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_shapes = sum(1 for cnt in contours if 50 < cv2.contourArea(cnt) < 500)
        if small_shapes > 5:
            score += 0.5
        
        return min(1.0, score)
    
    def _calibrate_confidence(self, raw_confidence: float) -> float:
        """
        使用Platt scaling校准置信度
        
        Platt scaling: P(y=1|f) = 1 / (1 + exp(A*f + B))
        其中f是原始分数，A和B是通过训练数据学习的参数
        
        Args:
            raw_confidence: 原始置信度
            
        Returns:
            校准后的置信度
        """
        # 应用sigmoid函数进行校准
        z = self.calibration_a * raw_confidence + self.calibration_b
        calibrated = 1.0 / (1.0 + np.exp(-z))
        
        return float(calibrated)
    
    def _get_context_scores(self, metadata: Dict) -> Dict[str, float]:
        """
        基于上下文元数据获取分数
        
        Args:
            metadata: 检测元数据，可能包含：
                - detector: 检测器名称
                - page_position: 页面位置
                - surrounding_text: 周围文本
                
        Returns:
            各主类别的上下文分数
        """
        scores = {k: 0.0 for k in ['chart', 'microscopy', 'diagram', 'photo', 'spectrum']}
        
        if not metadata:
            return scores
        
        # 基于检测器类型
        detector = metadata.get('detector', '')
        if 'table' in detector.lower():
            scores['chart'] += 0.3
        
        # 基于页面位置
        position = metadata.get('page_position', '')
        if position == 'top' or position == 'bottom':
            scores['chart'] += 0.2
        
        # 基于周围文本
        surrounding = metadata.get('surrounding_text', '').lower()
        if 'figure' in surrounding or 'fig.' in surrounding:
            scores['chart'] += 0.3
        if 'image' in surrounding or 'photo' in surrounding:
            scores['photo'] += 0.3
        
        return scores
    
    def _get_matched_keywords(
        self,
        chart_type: str,
        caption: str,
        snippet: str,
        ocr_text: str
    ) -> List[str]:
        """获取匹配的关键词列表"""
        text_combined = f"{caption} {snippet} {ocr_text}".lower()
        matched = []
        
        keywords = self.CHART_KEYWORDS.get(chart_type, [])
        for kw in keywords:
            if kw in text_combined:
                matched.append(kw)
        
        return matched
    
    def _classify_flat(
        self,
        image_path: str,
        caption: str,
        snippet: str,
        ocr_text: str,
        metadata: Optional[Dict]
    ) -> Tuple[str, float, List[str], Dict]:
        """
        扁平分类（不使用层次化）
        
        Returns:
            (chart_type, confidence, matched_keywords, debug_info)
        """
        text_combined = f"{caption} {snippet} {ocr_text}".lower()
        
        # 计算所有类型的分数
        scores = {}
        
        # 关键词分数
        for chart_type in self.CHART_TYPES:
            keywords = self.CHART_KEYWORDS.get(chart_type, [])
            kw_score = sum(1 for kw in keywords if kw in text_combined)
            scores[chart_type] = kw_score
        
        # 视觉分数
        if self.enable_visual:
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # 为每种类型计算视觉分数
                    visual_scores = self._get_all_visual_scores(gray, img)
                    for chart_type, v_score in visual_scores.items():
                        scores[chart_type] = scores.get(chart_type, 0.0) + v_score * 2.0
            except Exception as e:
                logger.warning(f"Visual analysis failed: {e}")
        
        # 找到最高分
        best_type = max(scores, key=scores.get) if scores else 'unknown'
        max_score = scores.get(best_type, 0.0)
        
        # 归一化置信度
        total = sum(scores.values())
        confidence = max_score / total if total > 0 else 0.0
        
        # 校准
        if self.enable_calibration:
            confidence = self._calibrate_confidence(confidence)
        
        # 获取匹配的关键词
        matched_kws = self._get_matched_keywords(best_type, caption, snippet, ocr_text)
        
        debug = {
            'method': 'flat_v1.7',
            'scores': scores,
            'raw_confidence': max_score / total if total > 0 else 0.0
        }
        
        return best_type, confidence, matched_kws, debug
    
    def _get_all_visual_scores(self, gray: np.ndarray, img: np.ndarray) -> Dict[str, float]:
        """获取所有类型的视觉分数"""
        scores = {}
        
        # Chart类型
        scores.update(self._classify_chart_subtypes(gray))
        
        # Microscopy类型
        scores.update(self._classify_microscopy_subtypes(gray, img))
        
        # Diagram类型
        scores.update(self._classify_diagram_subtypes(gray))
        
        # 其他类型
        scores['photo'] = self._detect_photo_features(gray, img)
        scores['spectrum'] = self._detect_spectrum_features(gray)
        
        return scores


# 便捷函数
def classify_chart_enhanced(
    image_path: str,
    caption: str = "",
    snippet: str = "",
    ocr_text: str = "",
    config: Optional[Dict] = None
) -> Tuple[str, float, List[str], Dict]:
    """
    使用增强型分类器分类图表
    
    Args:
        image_path: 图表图像路径
        caption: 标题文本
        snippet: 片段文本
        ocr_text: OCR文本
        config: 配置字典
        
    Returns:
        (chart_type, confidence, matched_keywords, debug_info)
    """
    classifier = EnhancedChartClassifier(config)
    return classifier.classify(image_path, caption, snippet, ocr_text)

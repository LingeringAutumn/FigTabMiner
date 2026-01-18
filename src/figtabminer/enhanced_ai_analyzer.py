"""
Enhanced AI Analyzer for FigTabMiner v1.7

This module provides robust AI analysis for charts and figures, including:
- Input validation
- Scientific metadata extraction
- Improved subtype recognition with confidence thresholds
"""

import re
import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import cv2
import numpy as np

from . import config
from . import utils

logger = utils.setup_logging(__name__)


@dataclass
class ValidationResult:
    """输入验证结果"""
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ScientificMetadata:
    """科学元数据"""
    experimental_conditions: List[Dict] = field(default_factory=list)
    materials: List[str] = field(default_factory=list)
    measurements: List[Dict] = field(default_factory=list)
    units: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """AI分析结果"""
    subtype: str
    subtype_confidence: float
    conditions: List[Dict] = field(default_factory=list)
    materials: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    method: str = "enhanced_analyzer_v1.7"
    debug: Optional[Dict] = None


class EnhancedAIAnalyzer:
    """
    增强型AI分析器
    
    改进：
    1. 输入验证 - 检查图像质量和数据完整性
    2. 鲁棒的子类型识别 - 最小置信度阈值0.6
    3. 科学元数据提取 - 实验条件、材料、测量值
    """
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """
        初始化增强型AI分析器
        
        Args:
            config_dict: 配置字典
        """
        self.config = config_dict or {}
        self.min_confidence = self.config.get('min_subtype_confidence', 0.6)
        self.enable_validation = self.config.get('enable_input_validation', True)
        self.enable_metadata_extraction = self.config.get('enable_metadata_extraction', True)
        
        logger.info(f"EnhancedAIAnalyzer initialized (min_confidence={self.min_confidence})")
    
    def validate_input(
        self,
        image_path: str,
        chart_type: str
    ) -> ValidationResult:
        """
        验证输入数据质量
        
        Args:
            image_path: 图表图像路径
            chart_type: 图表类型
            
        Returns:
            ValidationResult包含验证结果和建议
        """
        issues = []
        suggestions = []
        
        # 1. 检查图像文件是否存在
        if not os.path.exists(image_path):
            issues.append(f"Image file not found: {image_path}")
            suggestions.append("Check if the image path is correct")
            return ValidationResult(is_valid=False, issues=issues, suggestions=suggestions)
        
        # 2. 检查图像是否可读
        try:
            img = cv2.imread(image_path)
            if img is None:
                issues.append("Image file is corrupted or unreadable")
                suggestions.append("Try re-extracting the image from PDF")
                return ValidationResult(is_valid=False, issues=issues, suggestions=suggestions)
        except Exception as e:
            issues.append(f"Failed to load image: {e}")
            suggestions.append("Check if the image format is supported")
            return ValidationResult(is_valid=False, issues=issues, suggestions=suggestions)
        
        # 3. 检查图像质量
        h, w = img.shape[:2]
        
        if w < 50 or h < 50:
            issues.append(f"Image too small: {w}x{h}")
            suggestions.append("Image may be too small for reliable analysis")
        
        if w > 5000 or h > 5000:
            issues.append(f"Image very large: {w}x{h}")
            suggestions.append("Consider resizing for faster processing")
        
        # 4. 检查图表类型
        valid_types = ['figure', 'table', 'chart', 'unknown']
        if chart_type not in valid_types:
            issues.append(f"Invalid chart type: {chart_type}")
            suggestions.append(f"Use one of: {valid_types}")
        
        # 5. 检查图像清晰度（简单的模糊检测）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 10:
            issues.append(f"Image may be blurry (variance={laplacian_var:.2f})")
            suggestions.append("Low image quality may affect analysis accuracy")
        
        is_valid = len([i for i in issues if 'not found' in i or 'corrupted' in i]) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            suggestions=suggestions
        )
    
    def extract_scientific_metadata(
        self,
        image_path: str,
        chart_type: str,
        ocr_text: str
    ) -> ScientificMetadata:
        """
        提取科学元数据
        
        Args:
            image_path: 图表图像路径
            chart_type: 图表类型
            ocr_text: OCR文本
            
        Returns:
            ScientificMetadata包含实验条件、材料等
        """
        metadata = ScientificMetadata()
        
        if not self.enable_metadata_extraction:
            return metadata
        
        # 1. 提取实验条件
        for cond_type, pattern in config.CONDITION_PATTERNS.items():
            matches = re.finditer(pattern, ocr_text)
            for m in matches:
                metadata.experimental_conditions.append({
                    "type": cond_type,
                    "value": m.group(1),
                    "source": "ocr_regex",
                    "confidence": 0.8
                })
        
        # 2. 提取材料（简单的化学式识别）
        # 使用更严格的过滤避免误识别
        for pattern in config.MATERIAL_PATTERNS:
            matches = re.finditer(pattern, ocr_text)
            for m in matches:
                material = m.group(0)
                # 过滤常见单词
                if len(material) > 1 and material not in ['The', 'In', 'On', 'At', 'To', 'For']:
                    if material not in metadata.materials:
                        metadata.materials.append(material)
        
        # 限制材料数量避免噪音
        metadata.materials = metadata.materials[:10]
        
        # 3. 提取测量值和单位
        measurement_pattern = r'(\d+(\.\d+)?)\s*([a-zA-Zμ°]+)'
        matches = re.finditer(measurement_pattern, ocr_text)
        for m in matches:
            value = m.group(1)
            unit = m.group(3)
            metadata.measurements.append({
                "value": value,
                "unit": unit,
                "source": "ocr_regex"
            })
            if unit not in metadata.units:
                metadata.units.append(unit)
        
        # 限制数量
        metadata.measurements = metadata.measurements[:20]
        metadata.units = metadata.units[:10]
        
        return metadata
    
    def _classify_subtype_robust(
        self,
        image_path: str,
        caption: str,
        snippet: str,
        ocr_text: str
    ) -> Tuple[str, float, List[str], Dict]:
        """
        鲁棒的子类型识别
        
        Returns:
            (subtype, confidence, keywords, debug_info)
        """
        text_combined = (caption + " " + snippet + " " + ocr_text).lower()
        
        # 1. 关键词匹配
        scores = {k: 0.0 for k in config.SUBTYPE_KEYWORDS.keys()}
        matched_kws = []
        
        for subtype, kws in config.SUBTYPE_KEYWORDS.items():
            for kw in kws:
                if kw.lower() in text_combined:
                    scores[subtype] += 1.0
                    matched_kws.append(kw)
        
        # 2. 图像启发式分析
        img = cv2.imread(image_path)
        white_ratio = 0.0
        edge_density = 0.0
        
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 白色像素比例
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            white_ratio = np.sum(thresh == 255) / thresh.size
            
            # 边缘密度
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 基于视觉特征调整分数
            if white_ratio < 0.3:
                # 暗图像 -> 可能是显微镜图像
                scores["microscopy"] += 0.5
            elif white_ratio > 0.7:
                # 白色背景 -> 可能是线图或光谱图
                scores["line_plot"] += 0.5
                scores["spectrum"] += 0.3
            
            # 高边缘密度 -> 可能是线图
            if edge_density > 0.05:
                scores["line_plot"] += 0.3
        
        # 3. 确定最佳子类型
        best_subtype = "unknown"
        max_score = 0.0
        
        for st, score in scores.items():
            if score > max_score:
                max_score = score
                best_subtype = st
        
        # 4. 计算置信度
        # 归一化分数到0-1范围
        confidence = min(1.0, max_score / 3.0)
        
        # 如果没有关键词但图像特征明显，给予中等置信度
        if best_subtype == "unknown" and white_ratio > 0.7:
            best_subtype = "line_plot"
            confidence = 0.3
        
        # 5. 应用最小置信度阈值
        if confidence < self.min_confidence:
            logger.warning(
                f"Subtype confidence {confidence:.2f} below threshold {self.min_confidence}, "
                f"marking as low_confidence"
            )
            best_subtype = f"{best_subtype}_low_confidence"
        
        debug_info = {
            "scores": scores,
            "white_ratio": white_ratio,
            "edge_density": edge_density,
            "matched_keywords": matched_kws,
            "confidence_threshold": self.min_confidence
        }
        
        return best_subtype, confidence, list(set(matched_kws)), debug_info
    
    def analyze_robust(
        self,
        image_path: str,
        chart_type: str,
        caption: str = "",
        snippet: str = "",
        ocr_text: str = ""
    ) -> AnalysisResult:
        """
        鲁棒的图表分析
        
        Args:
            image_path: 图表图像路径
            chart_type: 图表类型
            caption: 标题
            snippet: 片段
            ocr_text: OCR文本
            
        Returns:
            AnalysisResult包含分析结果
        """
        debug_info = {}
        
        # 1. 输入验证
        if self.enable_validation:
            validation = self.validate_input(image_path, chart_type)
            debug_info['validation'] = {
                'is_valid': validation.is_valid,
                'issues': validation.issues,
                'suggestions': validation.suggestions
            }
            
            if not validation.is_valid:
                logger.error(f"Input validation failed: {validation.issues}")
                return AnalysisResult(
                    subtype="unknown",
                    subtype_confidence=0.0,
                    method="enhanced_analyzer_v1.7_validation_failed",
                    debug=debug_info
                )
        
        # 2. 子类型识别
        try:
            subtype, confidence, keywords, subtype_debug = self._classify_subtype_robust(
                image_path, caption, snippet, ocr_text
            )
            debug_info['subtype_classification'] = subtype_debug
        except Exception as e:
            logger.error(f"Subtype classification failed: {e}", exc_info=True)
            subtype = "unknown"
            confidence = 0.0
            keywords = []
            debug_info['subtype_error'] = str(e)
        
        # 3. 科学元数据提取
        conditions = []
        materials = []
        
        if self.enable_metadata_extraction:
            try:
                full_text = f"{caption} {snippet} {ocr_text}"
                metadata = self.extract_scientific_metadata(
                    image_path, chart_type, full_text
                )
                conditions = metadata.experimental_conditions
                materials = metadata.materials
                debug_info['metadata'] = {
                    'conditions_count': len(conditions),
                    'materials_count': len(materials),
                    'measurements_count': len(metadata.measurements)
                }
            except Exception as e:
                logger.warning(f"Metadata extraction failed: {e}")
                debug_info['metadata_error'] = str(e)
        
        return AnalysisResult(
            subtype=subtype,
            subtype_confidence=confidence,
            conditions=conditions,
            materials=materials,
            keywords=keywords,
            method="enhanced_analyzer_v1.7",
            debug=debug_info
        )


# 便捷函数
def analyze_chart(
    image_path: str,
    chart_type: str,
    caption: str = "",
    snippet: str = "",
    ocr_text: str = "",
    config_dict: Optional[Dict] = None
) -> AnalysisResult:
    """
    分析图表的便捷函数
    
    Args:
        image_path: 图表图像路径
        chart_type: 图表类型
        caption: 标题
        snippet: 片段
        ocr_text: OCR文本
        config_dict: 配置字典
        
    Returns:
        AnalysisResult
    """
    analyzer = EnhancedAIAnalyzer(config_dict)
    return analyzer.analyze_robust(image_path, chart_type, caption, snippet, ocr_text)

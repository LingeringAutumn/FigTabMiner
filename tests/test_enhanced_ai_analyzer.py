"""
Tests for Enhanced AI Analyzer (v1.7)
"""

import os
import sys
import tempfile
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from figtabminer.enhanced_ai_analyzer import (
    EnhancedAIAnalyzer,
    ValidationResult,
    ScientificMetadata,
    AnalysisResult,
    analyze_chart
)


def create_test_image(width=200, height=200, white_ratio=0.5):
    """创建测试图像"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # 添加一些内容
    if white_ratio < 0.5:
        # 暗图像（显微镜风格）
        img = np.ones((height, width, 3), dtype=np.uint8) * 50
        cv2.circle(img, (width//2, height//2), 50, (200, 200, 200), -1)
    else:
        # 亮图像（线图风格）
        cv2.line(img, (20, 180), (180, 20), (0, 0, 0), 2)
        cv2.line(img, (20, 180), (180, 180), (0, 0, 0), 2)
        cv2.line(img, (20, 20), (20, 180), (0, 0, 0), 2)
    
    return img


class TestInputValidation:
    """测试输入验证"""
    
    def test_validate_valid_input(self):
        """测试有效输入"""
        analyzer = EnhancedAIAnalyzer()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = create_test_image()
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            result = analyzer.validate_input(temp_path, 'figure')
            assert result.is_valid, f"Should be valid, issues: {result.issues}"
            print("✓ test_validate_valid_input")
        finally:
            os.unlink(temp_path)
    
    def test_validate_missing_file(self):
        """测试缺失文件"""
        analyzer = EnhancedAIAnalyzer()
        
        result = analyzer.validate_input('/nonexistent/file.png', 'figure')
        assert not result.is_valid
        assert len(result.issues) > 0
        assert 'not found' in result.issues[0].lower()
        print("✓ test_validate_missing_file")
    
    def test_validate_small_image(self):
        """测试过小图像"""
        analyzer = EnhancedAIAnalyzer()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = create_test_image(width=30, height=30)
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            result = analyzer.validate_input(temp_path, 'figure')
            # 图像过小会有警告，但仍然valid
            assert result.is_valid
            assert any('small' in issue.lower() for issue in result.issues)
            print("✓ test_validate_small_image")
        finally:
            os.unlink(temp_path)
    
    def test_validate_invalid_chart_type(self):
        """测试无效图表类型"""
        analyzer = EnhancedAIAnalyzer()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = create_test_image()
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            result = analyzer.validate_input(temp_path, 'invalid_type')
            assert any('invalid chart type' in issue.lower() for issue in result.issues)
            print("✓ test_validate_invalid_chart_type")
        finally:
            os.unlink(temp_path)


class TestScientificMetadataExtraction:
    """测试科学元数据提取"""
    
    def test_extract_temperature(self):
        """测试提取温度"""
        analyzer = EnhancedAIAnalyzer()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = create_test_image()
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            ocr_text = "The sample was heated to 300°C for 2 hours"
            metadata = analyzer.extract_scientific_metadata(temp_path, 'figure', ocr_text)
            
            # 应该提取到温度
            temp_conditions = [c for c in metadata.experimental_conditions if c['type'] == 'temperature']
            assert len(temp_conditions) > 0, "Should extract temperature"
            assert '300°C' in temp_conditions[0]['value']
            print("✓ test_extract_temperature")
        finally:
            os.unlink(temp_path)
    
    def test_extract_multiple_conditions(self):
        """测试提取多个条件"""
        analyzer = EnhancedAIAnalyzer()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = create_test_image()
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            ocr_text = "Measured at 500 nm wavelength, 25°C, pH = 7.4, 1 bar pressure"
            metadata = analyzer.extract_scientific_metadata(temp_path, 'figure', ocr_text)
            
            # 应该提取到多个条件
            assert len(metadata.experimental_conditions) >= 3, \
                f"Should extract multiple conditions, got {len(metadata.experimental_conditions)}"
            
            condition_types = [c['type'] for c in metadata.experimental_conditions]
            assert 'temperature' in condition_types
            assert 'wavelength' in condition_types
            assert 'ph' in condition_types
            print("✓ test_extract_multiple_conditions")
        finally:
            os.unlink(temp_path)
    
    def test_extract_measurements(self):
        """测试提取测量值"""
        analyzer = EnhancedAIAnalyzer()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = create_test_image()
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            ocr_text = "Peak at 1650 cm-1, intensity 0.85 a.u."
            metadata = analyzer.extract_scientific_metadata(temp_path, 'figure', ocr_text)
            
            # 应该提取到测量值
            assert len(metadata.measurements) > 0, "Should extract measurements"
            assert len(metadata.units) > 0, "Should extract units"
            print("✓ test_extract_measurements")
        finally:
            os.unlink(temp_path)


class TestSubtypeRecognition:
    """测试子类型识别"""
    
    def test_microscopy_recognition(self):
        """测试显微镜图像识别"""
        analyzer = EnhancedAIAnalyzer()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            # 创建暗图像（显微镜风格）
            img = create_test_image(white_ratio=0.2)
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            caption = "SEM image of the sample"
            result = analyzer.analyze_robust(temp_path, 'figure', caption=caption)
            
            assert 'microscopy' in result.subtype.lower(), \
                f"Should recognize microscopy, got {result.subtype}"
            assert result.subtype_confidence > 0, "Should have confidence > 0"
            print("✓ test_microscopy_recognition")
        finally:
            os.unlink(temp_path)
    
    def test_spectrum_recognition(self):
        """测试光谱图识别"""
        analyzer = EnhancedAIAnalyzer()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = create_test_image(white_ratio=0.8)
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            caption = "FTIR spectrum showing characteristic peaks"
            result = analyzer.analyze_robust(temp_path, 'figure', caption=caption)
            
            assert 'spectrum' in result.subtype.lower(), \
                f"Should recognize spectrum, got {result.subtype}"
            print("✓ test_spectrum_recognition")
        finally:
            os.unlink(temp_path)
    
    def test_line_plot_recognition(self):
        """测试线图识别"""
        analyzer = EnhancedAIAnalyzer()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = create_test_image(white_ratio=0.8)
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            caption = "Plot of temperature vs time"
            result = analyzer.analyze_robust(temp_path, 'figure', caption=caption)
            
            assert 'line_plot' in result.subtype.lower(), \
                f"Should recognize line_plot, got {result.subtype}"
            print("✓ test_line_plot_recognition")
        finally:
            os.unlink(temp_path)
    
    def test_low_confidence_marking(self):
        """测试低置信度标记"""
        analyzer = EnhancedAIAnalyzer({'min_subtype_confidence': 0.8})
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = create_test_image()
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            # 没有明确关键词，置信度会较低
            result = analyzer.analyze_robust(temp_path, 'figure', caption="")
            
            # 如果置信度低于阈值，应该标记为low_confidence
            if result.subtype_confidence < 0.8:
                assert 'low_confidence' in result.subtype, \
                    f"Low confidence should be marked, got {result.subtype}"
            print("✓ test_low_confidence_marking")
        finally:
            os.unlink(temp_path)


class TestRobustAnalysis:
    """测试鲁棒分析"""
    
    def test_analyze_with_all_inputs(self):
        """测试完整输入分析"""
        analyzer = EnhancedAIAnalyzer()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = create_test_image()
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            result = analyzer.analyze_robust(
                temp_path,
                'figure',
                caption="XRD spectrum of TiO2",
                snippet="The sample was measured at 300K",
                ocr_text="Peak at 25.3° corresponds to anatase phase"
            )
            
            assert result.subtype != "", "Should have subtype"
            assert result.subtype_confidence >= 0, "Should have confidence"
            assert result.method == "enhanced_analyzer_v1.7"
            assert result.debug is not None, "Should have debug info"
            print("✓ test_analyze_with_all_inputs")
        finally:
            os.unlink(temp_path)
    
    def test_analyze_with_validation_failure(self):
        """测试验证失败的分析"""
        analyzer = EnhancedAIAnalyzer({'enable_input_validation': True})
        
        result = analyzer.analyze_robust('/nonexistent/file.png', 'figure')
        
        assert result.subtype == "unknown"
        assert result.subtype_confidence == 0.0
        assert 'validation_failed' in result.method
        assert 'validation' in result.debug
        print("✓ test_analyze_with_validation_failure")
    
    def test_analyze_without_validation(self):
        """测试禁用验证的分析"""
        analyzer = EnhancedAIAnalyzer({'enable_input_validation': False})
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = create_test_image()
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            result = analyzer.analyze_robust(temp_path, 'figure')
            
            # 应该没有验证信息
            assert 'validation' not in result.debug or result.debug['validation'] is None
            print("✓ test_analyze_without_validation")
        finally:
            os.unlink(temp_path)
    
    def test_analyze_with_metadata_extraction(self):
        """测试带元数据提取的分析"""
        analyzer = EnhancedAIAnalyzer({'enable_metadata_extraction': True})
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = create_test_image()
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            result = analyzer.analyze_robust(
                temp_path,
                'figure',
                ocr_text="Measured at 500 nm, 25°C"
            )
            
            # 应该提取到条件
            assert len(result.conditions) > 0, "Should extract conditions"
            assert 'metadata' in result.debug
            print("✓ test_analyze_with_metadata_extraction")
        finally:
            os.unlink(temp_path)


class TestConvenienceFunctions:
    """测试便捷函数"""
    
    def test_analyze_chart_function(self):
        """测试analyze_chart便捷函数"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = create_test_image()
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            result = analyze_chart(
                temp_path,
                'figure',
                caption="Test chart"
            )
            
            assert isinstance(result, AnalysisResult)
            assert result.subtype != ""
            print("✓ test_analyze_chart_function")
        finally:
            os.unlink(temp_path)


def run_tests():
    """运行所有测试"""
    print("=" * 70)
    print("Enhanced AI Analyzer Test Suite (v1.7)")
    print("=" * 70)
    
    test_classes = [
        ("TestInputValidation", TestInputValidation),
        ("TestScientificMetadataExtraction", TestScientificMetadataExtraction),
        ("TestSubtypeRecognition", TestSubtypeRecognition),
        ("TestRobustAnalysis", TestRobustAnalysis),
        ("TestConvenienceFunctions", TestConvenienceFunctions),
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for class_name, test_class in test_classes:
        print(f"\n{class_name}:")
        print("-" * 70)
        
        instance = test_class()
        test_methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                passed_tests += 1
            except AssertionError as e:
                failed_tests += 1
                print(f"✗ {method_name}: {e}")
            except Exception as e:
                failed_tests += 1
                print(f"✗ {method_name}: Unexpected error: {e}")
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    
    if failed_tests == 0:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {failed_tests} test(s) failed")
    
    return failed_tests == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

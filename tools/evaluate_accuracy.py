#!/usr/bin/env python3
"""
准确率评估工具

使用标注数据集评估系统性能：
- 计算 Precision, Recall, F1-score
- 计算边界框 IoU
- 计算过度分割率和错误合并率
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from figtabminer import utils, bbox_utils

logger = utils.setup_logging(__name__)


class AccuracyEvaluator:
    """准确率评估器"""
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.results = {
            "total_docs": 0,
            "total_gt_items": 0,
            "total_pred_items": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "iou_scores": [],
            "per_doc_results": []
        }
    
    def load_ground_truth(self, annotation_path: Path) -> Dict:
        """加载标注数据（ground truth）"""
        return utils.read_json(annotation_path)
    
    def load_predictions(self, doc_id: str, output_dir: Path) -> Dict:
        """加载系统预测结果"""
        manifest_path = output_dir / doc_id / "manifest.json"
        if not manifest_path.exists():
            return None
        return utils.read_json(manifest_path)
    
    def compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """计算两个边界框的 IoU"""
        return bbox_utils.bbox_iou(bbox1, bbox2)
    
    def match_predictions_to_ground_truth(
        self,
        gt_items: List[Dict],
        pred_items: List[Dict]
    ) -> Tuple[List, List, List]:
        """
        匹配预测结果和标注数据
        
        Returns:
            (matched_pairs, unmatched_gt, unmatched_pred)
        """
        matched_pairs = []
        unmatched_gt = list(range(len(gt_items)))
        unmatched_pred = list(range(len(pred_items)))
        
        # 计算所有配对的 IoU
        iou_matrix = np.zeros((len(gt_items), len(pred_items)))
        
        for i, gt_item in enumerate(gt_items):
            for j, pred_item in enumerate(pred_items):
                # 只匹配同一页面的项目
                if gt_item["page_index"] != pred_item["page_index"]:
                    continue
                
                # 只匹配同类型的项目
                if gt_item["type"] != pred_item["type"]:
                    continue
                
                iou = self.compute_iou(gt_item["bbox"], pred_item["bbox"])
                iou_matrix[i, j] = iou
        
        # 贪心匹配：优先匹配 IoU 最高的
        while True:
            # 找到最大 IoU
            max_iou = iou_matrix.max()
            
            if max_iou < self.iou_threshold:
                break
            
            # 找到最大 IoU 的位置
            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            
            # 记录匹配
            matched_pairs.append({
                "gt_idx": i,
                "pred_idx": j,
                "iou": max_iou,
                "gt_item": gt_items[i],
                "pred_item": pred_items[j]
            })
            
            # 移除已匹配的项目
            if i in unmatched_gt:
                unmatched_gt.remove(i)
            if j in unmatched_pred:
                unmatched_pred.remove(j)
            
            # 清除该行和列
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0
        
        return matched_pairs, unmatched_gt, unmatched_pred
    
    def evaluate_document(
        self,
        doc_id: str,
        annotation_path: Path,
        output_dir: Path
    ) -> Dict:
        """评估单个文档"""
        # 加载数据
        gt_data = self.load_ground_truth(annotation_path)
        pred_data = self.load_predictions(doc_id, output_dir)
        
        if pred_data is None:
            logger.warning(f"No predictions found for {doc_id}")
            return None
        
        gt_items = gt_data["items"]
        pred_items = pred_data["items"]
        
        # 匹配
        matched, unmatched_gt, unmatched_pred = self.match_predictions_to_ground_truth(
            gt_items, pred_items
        )
        
        # 计算指标
        tp = len(matched)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 平均 IoU
        avg_iou = np.mean([m["iou"] for m in matched]) if matched else 0
        
        result = {
            "doc_id": doc_id,
            "gt_count": len(gt_items),
            "pred_count": len(pred_items),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "avg_iou": avg_iou,
            "matched_pairs": len(matched),
            "unmatched_gt": len(unmatched_gt),
            "unmatched_pred": len(unmatched_pred)
        }
        
        # 更新总体结果
        self.results["total_docs"] += 1
        self.results["total_gt_items"] += len(gt_items)
        self.results["total_pred_items"] += len(pred_items)
        self.results["true_positives"] += tp
        self.results["false_positives"] += fp
        self.results["false_negatives"] += fn
        self.results["iou_scores"].extend([m["iou"] for m in matched])
        self.results["per_doc_results"].append(result)
        
        return result
    
    def compute_overall_metrics(self) -> Dict:
        """计算总体指标"""
        tp = self.results["true_positives"]
        fp = self.results["false_positives"]
        fn = self.results["false_negatives"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_iou = np.mean(self.results["iou_scores"]) if self.results["iou_scores"] else 0
        
        return {
            "total_documents": self.results["total_docs"],
            "total_gt_items": self.results["total_gt_items"],
            "total_pred_items": self.results["total_pred_items"],
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "average_iou": avg_iou
        }
    
    def print_report(self):
        """打印评估报告"""
        overall = self.compute_overall_metrics()
        
        print("\n" + "="*60)
        print("FigTabMiner 准确率评估报告")
        print("="*60)
        
        print(f"\n总体统计:")
        print(f"  评估文档数: {overall['total_documents']}")
        print(f"  标注项目数: {overall['total_gt_items']}")
        print(f"  预测项目数: {overall['total_pred_items']}")
        
        print(f"\n检测性能:")
        print(f"  True Positives:  {overall['true_positives']}")
        print(f"  False Positives: {overall['false_positives']}")
        print(f"  False Negatives: {overall['false_negatives']}")
        
        print(f"\n准确率指标:")
        print(f"  Precision: {overall['precision']:.3f}")
        print(f"  Recall:    {overall['recall']:.3f}")
        print(f"  F1-Score:  {overall['f1_score']:.3f}")
        print(f"  Avg IoU:   {overall['average_iou']:.3f}")
        
        # 按文档显示
        print(f"\n{'='*60}")
        print("各文档详细结果:")
        print(f"{'='*60}")
        
        for result in self.results["per_doc_results"]:
            print(f"\n{result['doc_id']}:")
            print(f"  GT: {result['gt_count']}, Pred: {result['pred_count']}")
            print(f"  P: {result['precision']:.3f}, R: {result['recall']:.3f}, F1: {result['f1_score']:.3f}")
            print(f"  IoU: {result['avg_iou']:.3f}")
        
        print(f"\n{'='*60}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="准确率评估工具")
    parser.add_argument("--annotation-dir", default="data/annotations", help="标注目录")
    parser.add_argument("--output-dir", default="data/outputs", help="输出目录")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU 阈值")
    parser.add_argument("--save-report", help="保存报告到文件")
    
    args = parser.parse_args()
    
    annotation_dir = Path(args.annotation_dir)
    output_dir = Path(args.output_dir)
    
    if not annotation_dir.exists():
        print(f"错误：标注目录不存在: {annotation_dir}")
        return 1
    
    # 创建评估器
    evaluator = AccuracyEvaluator(iou_threshold=args.iou_threshold)
    
    # 评估所有标注文档
    annotation_files = list(annotation_dir.glob("*.json"))
    
    if not annotation_files:
        print(f"错误：未找到标注文件: {annotation_dir}")
        return 1
    
    print(f"找到 {len(annotation_files)} 个标注文件")
    
    for annotation_file in annotation_files:
        doc_id = annotation_file.stem
        print(f"\n评估: {doc_id}")
        
        result = evaluator.evaluate_document(doc_id, annotation_file, output_dir)
        
        if result:
            print(f"  ✓ F1: {result['f1_score']:.3f}, IoU: {result['avg_iou']:.3f}")
        else:
            print(f"  ✗ 评估失败")
    
    # 打印总体报告
    evaluator.print_report()
    
    # 保存报告
    if args.save_report:
        overall = evaluator.compute_overall_metrics()
        overall["per_document"] = evaluator.results["per_doc_results"]
        
        report_path = Path(args.save_report)
        utils.write_json(overall, report_path)
        print(f"\n报告已保存: {report_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


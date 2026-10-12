"""
evaluate_unknown_v3.py —— 第三阶段：开放集三分类模型评估
=========================================================
评估 train_openset.py 训练出来的“猫/狗/未知”三分类模型：

    unknown_score = softmax 输出的 P(unknown)
    score >= threshold  -> Unknown
    score <  threshold  -> Cat / Dog（取猫狗两类里概率更大的）

评估数据与第二阶段保持一致，便于直接对比：
- 已知类：data/test_set/test_set 中的猫狗各 300 张
- 未知类：data/unknown_set 中按同样比例留出的 30% 未知图片

用法：
    python evaluate_unknown_v3.py --weights models/openset-vgg16.pth
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.model import vgg16_openset
from src.openset_data import CLASS_UNKNOWN, collect_unknown_items, split_unknown_items
from evaluate_unknown import (average_precision, build_transform,
                              collect_known_items, roc_auc_score)


def parse_args():
    parser = argparse.ArgumentParser(description='第三阶段开放集模型评估')
    parser.add_argument('--weights', type=str, default='models/openset-vgg16.pth')
    parser.add_argument('--known_dir', type=str, default='data/test_set/test_set')
    parser.add_argument('--unknown_dir', type=str, default='data/unknown_set')
    parser.add_argument('--out_dir', type=str, default='reports/unknown_eval_v3')
    parser.add_argument('--max_known_per_class', type=int, default=300)
    parser.add_argument('--unknown_train_ratio', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_fpr', type=float, default=0.05)
    parser.add_argument('--holdout_unknown_classes', type=str, default='',
                        help='只评估这些完全留出的未知类别（逗号分隔，例如 rabbit）')
    return parser.parse_args()


@torch.inference_mode()
def predict_unknown_scores(model, items, transform, device, batch_size):
    """返回每张图片的 P(unknown) 以及猫狗分类预测。"""
    unknown_scores = []
    known_predictions = []

    for start in range(0, len(items), batch_size):
        batch_items = items[start:start + batch_size]
        batch = torch.stack([
            transform(Image.open(path).convert('RGB')) for path, _ in batch_items
        ]).to(device)
        probabilities = torch.softmax(model(batch), dim=1)
        unknown_scores.extend(probabilities[:, CLASS_UNKNOWN].cpu().numpy().tolist())
        known_predictions.extend(probabilities[:, :2].argmax(dim=1).cpu().numpy().tolist())

    return np.asarray(unknown_scores), np.asarray(known_predictions)


def sweep_thresholds(known_scores, unknown_scores, max_fpr):
    scores = np.concatenate([known_scores, unknown_scores])
    thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 201)))

    rows = []
    for threshold in thresholds:
        known_flagged = known_scores >= threshold
        unknown_detected = unknown_scores >= threshold
        known_fpr = float(known_flagged.mean())
        unknown_recall = float(unknown_detected.mean())
        tp = int(unknown_detected.sum())
        fp = int(known_flagged.sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * unknown_recall / (precision + unknown_recall)
              if (precision + unknown_recall) > 0 else 0.0)
        rows.append({
            'threshold': round(float(threshold), 6),
            'known_false_unknown_rate': round(known_fpr, 4),
            'unknown_recall': round(unknown_recall, 4),
            'unknown_precision': round(precision, 4),
            'f1': round(f1, 4),
        })

    candidates = [r for r in rows if r['known_false_unknown_rate'] <= max_fpr]
    if candidates:
        best = max(candidates, key=lambda r: (r['f1'], r['unknown_recall']))
    else:
        best = min(rows, key=lambda r: (r['known_false_unknown_rate'], -r['f1']))
    return rows, best


def main():
    args = parse_args()
    base_dir = Path('.')

    holdout_classes = [name.strip() for name in
                       args.holdout_unknown_classes.split(',') if name.strip()]
    suffix = f"_holdout_{'-'.join(holdout_classes)}" if holdout_classes else ''
    out_dir = base_dir / (args.out_dir + suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = vgg16_openset(num_classes=3)
    model.load_state_dict(torch.load(base_dir / args.weights, map_location='cpu'))
    model.to(device).eval()

    transform = build_transform()

    known_items = collect_known_items(base_dir / args.known_dir,
                                      args.max_known_per_class)
    unknown_items = collect_unknown_items(base_dir / args.unknown_dir)
    if holdout_classes:
        # 留一类实验：只评估完全没参与训练的类别
        unknown_val = [item for item in unknown_items
                       if item[0].parent.name in holdout_classes]
    else:
        _, unknown_val = split_unknown_items(unknown_items,
                                             train_ratio=args.unknown_train_ratio,
                                             seed=args.seed)

    known_labels = np.asarray([label for _, label in known_items])
    unknown_names = [path.parent.name for path, _ in unknown_val]
    print(f'已知类图片: {len(known_items)} 张')
    print(f'留出未知图片: {len(unknown_val)} 张')

    known_scores, known_predictions = predict_unknown_scores(
        model, known_items, transform, device, args.batch_size)
    unknown_scores, _ = predict_unknown_scores(
        model, unknown_val, transform, device, args.batch_size)

    known_accuracy = float((known_predictions == known_labels).mean())
    scores = np.concatenate([known_scores, unknown_scores])
    is_unknown = np.concatenate([np.zeros(len(known_scores)),
                                 np.ones(len(unknown_scores))])
    auroc = roc_auc_score(scores, is_unknown)
    ap = average_precision(scores, is_unknown)
    rows, best = sweep_thresholds(known_scores, unknown_scores, args.max_fpr)

    per_class = {}
    unknown_names_array = np.asarray(unknown_names)
    for class_name in sorted(set(unknown_names)):
        mask = unknown_names_array == class_name
        per_class[class_name] = {
            'count': int(mask.sum()),
            'detected_at_best_threshold':
                round(float((unknown_scores[mask] >= best['threshold']).mean()), 4),
        }

    summary = {
        'weights': str(base_dir / args.weights),
        'holdout_unknown_classes': holdout_classes,
        'known_images': len(known_items),
        'unknown_images': len(unknown_val),
        'known_closed_set_accuracy': round(known_accuracy, 4),
        'auroc': round(float(auroc), 4),
        'average_precision': round(float(ap), 4),
        'best_threshold': best['threshold'],
        'best_metrics': best,
        'per_unknown_class': per_class,
    }

    with open(out_dir / 'threshold_sweep.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    plt.figure(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 40)
    plt.hist(known_scores, bins=bins, alpha=0.6, label='Known (Cat/Dog)')
    plt.hist(unknown_scores, bins=bins, alpha=0.6, label='Unknown')
    plt.xlabel('P(unknown)')
    plt.ylabel('Number of images')
    plt.title('Unknown score distribution (three-class model)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'score_histogram.png', dpi=150)
    plt.close()

    print('\n===== 第三阶段评估结果 =====')
    print(f'已知类闭集准确率: {known_accuracy:.4f}')
    print(f'AUROC: {auroc:.4f}    AP: {ap:.4f}')
    print(f"推荐阈值: {best['threshold']:.4f} "
          f"(猫狗误报率 {best['known_false_unknown_rate']:.4f}, "
          f"未知检出率 {best['unknown_recall']:.4f}, F1 {best['f1']:.4f})")
    for class_name, stats in per_class.items():
        print(f"  {class_name}: {stats['count']} 张, "
              f"检出率 {stats['detected_at_best_threshold']:.4f}")
    print(f'\n结果已保存到: {out_dir.resolve()}')


if __name__ == '__main__':
    main()

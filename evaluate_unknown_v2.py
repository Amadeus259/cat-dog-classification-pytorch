"""
evaluate_unknown_v2.py —— 第二阶段：特征距离法未知检测评估
============================================================
相比第一阶段的“最大概率阈值”，本脚本改用 VGG16 特征空间的距离：

- 用训练集猫狗特征拟合类中心与共享协方差；
- 马氏距离 mahalanobis：考虑特征分布形状，通常效果更好；
- 余弦距离 cosine：只看特征方向，实现简单；
- 用同一批已知/未知图片计算 AUROC、AP、未知检出率、猫狗误报率；
- 结果保存到 reports/unknown_eval_v2/。

用法：
    python evaluate_unknown_v2.py --max_fit_per_class 400 --max_known_per_class 300
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

from src.ood import (FeatureExtractor, cosine_scores, fit_cosine,
                     fit_mahalanobis, mahalanobis_scores, save_ood_stats)
from evaluate_unknown import (average_precision, build_transform,
                              collect_known_items, collect_unknown_items,
                              load_model, roc_auc_score)


def parse_args():
    parser = argparse.ArgumentParser(description='第二阶段：特征距离法未知检测评估')
    parser.add_argument('--base_dir', type=str, default='.')
    parser.add_argument('--weights', type=str, default='models/01-trained-model.pth')
    parser.add_argument('--fit_dir', type=str, default='data/training_set/training_set',
                        help='用于拟合类中心/协方差的训练集目录')
    parser.add_argument('--known_dir', type=str, default='data/test_set/test_set')
    parser.add_argument('--unknown_dir', type=str, default='data/unknown_set')
    parser.add_argument('--out_dir', type=str, default='reports/unknown_eval_v2')
    parser.add_argument('--max_fit_per_class', type=int, default=400)
    parser.add_argument('--max_known_per_class', type=int, default=300)
    parser.add_argument('--max_unknown_per_class', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_fpr', type=float, default=0.05)
    return parser.parse_args()


def predict_dog_probabilities(model, paths, transform, device, batch_size):
    """得到模型输出的 P(dog)，用于计算闭集准确率。"""
    from PIL import Image
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            batch = torch.stack([
                transform(Image.open(p).convert('RGB')) for p in batch_paths
            ]).to(device)
            outputs = model(batch).squeeze(1)
            probabilities.extend(outputs.detach().cpu().numpy().tolist())
    return np.asarray(probabilities, dtype=np.float64)


def sweep_distance_thresholds(known_scores, unknown_scores, max_fpr):
    """扫描距离阈值：distance >= threshold 判为 Unknown。"""
    all_scores = np.concatenate([known_scores, unknown_scores])
    thresholds = np.unique(np.quantile(all_scores, np.linspace(0.0, 1.0, 201)))

    rows = []
    total_known = len(known_scores)
    total_unknown = len(unknown_scores)

    for threshold in thresholds:
        known_flagged = known_scores >= threshold
        unknown_detected = unknown_scores >= threshold

        known_fpr = float(known_flagged.sum()) / total_known
        unknown_recall = float(unknown_detected.sum()) / total_unknown
        tp = int(unknown_detected.sum())
        fp = int(known_flagged.sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * unknown_recall / (precision + unknown_recall)
              if (precision + unknown_recall) > 0 else 0.0)
        balanced_accuracy = 0.5 * (unknown_recall + (1.0 - known_fpr))

        rows.append({
            'threshold': round(float(threshold), 6),
            'known_false_unknown_rate': round(known_fpr, 4),
            'unknown_recall': round(unknown_recall, 4),
            'unknown_precision': round(precision, 4),
            'f1': round(f1, 4),
            'balanced_accuracy': round(balanced_accuracy, 4),
        })

    candidates = [r for r in rows if r['known_false_unknown_rate'] <= max_fpr]
    if candidates:
        best = max(candidates, key=lambda r: (r['f1'], r['unknown_recall']))
    else:
        best = min(rows, key=lambda r: (r['known_false_unknown_rate'], -r['f1']))
    return rows, best


def plot_score_histogram(known_scores, unknown_scores, method, out_path):
    plt.figure(figsize=(8, 5))
    bins = np.linspace(min(known_scores.min(), unknown_scores.min()),
                       np.quantile(np.concatenate([known_scores, unknown_scores]), 0.99),
                       40)
    plt.hist(known_scores, bins=bins, alpha=0.6, label='Known (Cat/Dog)')
    plt.hist(unknown_scores, bins=bins, alpha=0.6, label='Unknown (Rabbit/Cow/Horse)')
    plt.xlabel(f'{method} distance score (higher = more unknown)')
    plt.ylabel('Number of images')
    plt.title(f'Score distribution ({method})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_threshold_curves(rows, best, method, out_path):
    plt.figure(figsize=(8, 5))
    thresholds = [r['threshold'] for r in rows]
    plt.plot(thresholds, [r['unknown_recall'] for r in rows], label='Unknown recall')
    plt.plot(thresholds, [r['known_false_unknown_rate'] for r in rows],
             label='Known false-unknown rate')
    plt.plot(thresholds, [r['f1'] for r in rows], label='F1')
    plt.axvline(best['threshold'], color='red', linestyle='--',
                label=f"Best threshold = {best['threshold']:.4f}")
    plt.xlabel(f'{method} distance threshold')
    plt.ylabel('Score')
    plt.title(f'Threshold sweep ({method})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate_method(name, known_scores, unknown_scores, unknown_names,
                    max_fpr, out_dir):
    """对一种距离分数做完整评估，返回摘要字典。"""
    is_unknown = np.concatenate([np.zeros(len(known_scores)),
                                 np.ones(len(unknown_scores))])
    scores = np.concatenate([known_scores, unknown_scores])
    auroc = roc_auc_score(scores, is_unknown)
    ap = average_precision(scores, is_unknown)

    rows, best = sweep_distance_thresholds(known_scores, unknown_scores, max_fpr)

    per_class = {}
    unknown_names_array = np.asarray(unknown_names)
    for class_name in sorted(set(unknown_names)):
        mask = unknown_names_array == class_name
        detected = (unknown_scores[mask] >= best['threshold']).mean()
        per_class[class_name] = {
            'count': int(mask.sum()),
            'detected_at_best_threshold': round(float(detected), 4),
        }

    csv_path = out_dir / f'threshold_sweep_{name}.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_score_histogram(known_scores, unknown_scores, name,
                         out_dir / f'score_histogram_{name}.png')
    plot_threshold_curves(rows, best, name,
                          out_dir / f'threshold_curves_{name}.png')

    return {
        'method': name,
        'auroc': round(float(auroc), 4),
        'average_precision': round(float(ap), 4),
        'best_threshold': best['threshold'],
        'best_metrics': best,
        'per_unknown_class': per_class,
    }


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = load_model(base_dir / args.weights, device)
    transform = build_transform()
    extractor = FeatureExtractor(model, device)

    # 1) 用训练集特征拟合类中心与协方差
    fit_items = []
    for class_name, label in (('cats', 0), ('dogs', 1)):
        files = sorted((base_dir / args.fit_dir / class_name).glob('*.jpg'))
        if args.max_fit_per_class > 0:
            files = files[:args.max_fit_per_class]
        fit_items.extend((path, label) for path in files)
    fit_paths = [item[0] for item in fit_items]
    fit_labels = np.asarray([item[1] for item in fit_items])
    print(f'拟合用训练图片: {len(fit_items)} 张')

    fit_features = extractor.extract(fit_paths, transform, args.batch_size)
    means, precision = fit_mahalanobis(fit_features, fit_labels)
    centroids = fit_cosine(fit_features, fit_labels)
    save_ood_stats(out_dir / 'ood_stats.npz', means, precision, centroids)

    # 2) 提取已知类测试图片特征
    known_items = collect_known_items(base_dir / args.known_dir,
                                      args.max_known_per_class)
    unknown_items = collect_unknown_items(base_dir / args.unknown_dir,
                                          args.max_unknown_per_class)
    known_paths = [item[0] for item in known_items]
    known_labels = np.asarray([item[1] for item in known_items])
    unknown_paths = [item[0] for item in unknown_items]
    unknown_names = [item[1] for item in unknown_items]

    print(f'已知类图片: {len(known_items)} 张')
    print(f'未知类图片: {len(unknown_items)} 张')

    known_features = extractor.extract(known_paths, transform, args.batch_size)
    unknown_features = extractor.extract(unknown_paths, transform, args.batch_size)

    # 闭集准确率（只作为参考，与未知检测无关）
    known_probs = predict_dog_probabilities(model, known_paths, transform,
                                            device, args.batch_size)
    known_accuracy = float(((known_probs >= 0.5).astype(int) == known_labels).mean())

    # 3) 两种距离分别评估
    known_maha = mahalanobis_scores(known_features, means, precision)
    unknown_maha = mahalanobis_scores(unknown_features, means, precision)
    known_cos = cosine_scores(known_features, centroids)
    unknown_cos = cosine_scores(unknown_features, centroids)

    results = {
        'device': str(device),
        'known_images': len(known_items),
        'unknown_images': len(unknown_items),
        'known_closed_set_accuracy': round(known_accuracy, 4),
        'mahalanobis': evaluate_method('mahalanobis', known_maha, unknown_maha,
                                       unknown_names, args.max_fpr, out_dir),
        'cosine': evaluate_method('cosine', known_cos, unknown_cos,
                                  unknown_names, args.max_fpr, out_dir),
    }

    (out_dir / 'summary.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\n===== 第二阶段评估结果 =====')
    print(f'已知类闭集准确率: {known_accuracy:.4f}')
    for method in ('mahalanobis', 'cosine'):
        info = results[method]
        metrics = info['best_metrics']
        print(f"\n[{method}]")
        print(f"  AUROC: {info['auroc']:.4f}    AP: {info['average_precision']:.4f}")
        print(f"  推荐阈值: {info['best_threshold']:.4f}")
        print(f"  猫狗误报率: {metrics['known_false_unknown_rate']:.4f}")
        print(f"  未知检出率: {metrics['unknown_recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        for class_name, stats in info['per_unknown_class'].items():
            print(f"    {class_name}: {stats['count']} 张, "
                  f"检出率 {stats['detected_at_best_threshold']:.4f}")

    print(f'\n结果已保存到: {out_dir.resolve()}')


if __name__ == '__main__':
    main()

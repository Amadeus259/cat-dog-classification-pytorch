"""
evaluate_ensemble.py —— 融合马氏距离与 P(unknown) 的未知检测
============================================================
背景：
- 第三阶段（三分类）在“同类别未知图片”上表现很好（AUROC 0.999）；
- 但在“完全没见过的新类别（兔子）”上 AUROC 只有 0.81；
- 第二阶段的马氏距离是通用的特征距离，对新类别更稳健。

因此本脚本把两种信号融合：
    ensemble_score = 0.5 * z(马氏距离) + 0.5 * z(P(unknown))
其中 z 表示用已知类样本的均值和标准差做标准化。

评估对象是“训练时完全没见过的类别”，例如 rabbit。

用法：
    python evaluate_ensemble.py --holdout_unknown_classes rabbit \
        --openset_weights models/openset-vgg16-holdout-rabbit.pth
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.model import vgg16_openset, vgg16_transfer_learning
from src.ood import FeatureExtractor, load_ood_stats, mahalanobis_scores
from src.openset_data import collect_unknown_items
from evaluate_unknown import (average_precision, build_transform,
                              collect_known_items, roc_auc_score)


def parse_args():
    parser = argparse.ArgumentParser(description='融合马氏距离与 P(unknown) 的评估')
    parser.add_argument('--known_dir', type=str, default='data/test_set/test_set')
    parser.add_argument('--unknown_dir', type=str, default='data/unknown_set')
    parser.add_argument('--holdout_unknown_classes', type=str, default='rabbit')
    parser.add_argument('--binary_weights', type=str,
                        default='models/01-trained-model.pth')
    parser.add_argument('--ood_stats', type=str,
                        default='reports/unknown_eval_v2/ood_stats.npz')
    parser.add_argument('--openset_weights', type=str,
                        default='models/openset-vgg16-holdout-rabbit.pth')
    parser.add_argument('--out_dir', type=str, default='reports/unknown_ensemble')
    parser.add_argument('--max_known_per_class', type=int, default=300)
    parser.add_argument('--max_unknown_per_class', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_fpr', type=float, default=0.05)
    return parser.parse_args()


def batch_tensors(paths, transform, device, batch_size):
    """把图片路径批量转换成张量。"""
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        batch = torch.stack([
            transform(Image.open(path).convert('RGB')) for path in batch_paths
        ]).to(device)
        yield batch


@torch.inference_mode()
def openset_unknown_probabilities(model, paths, transform, device, batch_size):
    """计算三分类模型的 P(unknown)。"""
    scores = []
    for batch in batch_tensors(paths, transform, device, batch_size):
        probabilities = torch.softmax(model(batch), dim=1)
        scores.extend(probabilities[:, 2].detach().cpu().numpy().tolist())
    return np.asarray(scores)


def standardize(scores_known, scores_unknown):
    """用已知类分数的均值/标准差做 z-score 标准化。"""
    mean = scores_known.mean()
    std = scores_known.std() + 1e-12
    return (scores_known - mean) / std, (scores_unknown - mean) / std


def sweep(scores_known, scores_unknown, max_fpr):
    scores = np.concatenate([scores_known, scores_unknown])
    thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 201)))
    rows = []
    for threshold in thresholds:
        known_flagged = scores_known >= threshold
        unknown_detected = scores_unknown >= threshold
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


def evaluate_signal(name, known_scores, unknown_scores, unknown_names, max_fpr):
    labels = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    scores = np.concatenate([known_scores, unknown_scores])
    auroc = roc_auc_score(scores, labels)
    ap = average_precision(scores, labels)
    _, best = sweep(known_scores, unknown_scores, max_fpr)
    return {
        'name': name,
        'auroc': round(float(auroc), 4),
        'average_precision': round(float(ap), 4),
        'best_threshold': best['threshold'],
        'best_metrics': best,
    }


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
    transform = build_transform()

    known_items = collect_known_items(base_dir / args.known_dir,
                                      args.max_known_per_class)
    unknown_items = collect_unknown_items(base_dir / args.unknown_dir,
                                          args.max_unknown_per_class)
    unknown_items = [item for item in unknown_items
                     if item[0].parent.name in holdout_classes]
    known_paths = [path for path, _ in known_items]
    unknown_paths = [path for path, _ in unknown_items]
    print(f'已知类图片: {len(known_paths)} 张')
    print(f'完全留出类别 {holdout_classes}: {len(unknown_paths)} 张')

    # 1) 马氏距离（第二阶段）
    binary_model = vgg16_transfer_learning()
    binary_model.load_state_dict(torch.load(base_dir / args.binary_weights,
                                            map_location='cpu'))
    binary_model.to(device).eval()
    extractor = FeatureExtractor(binary_model, device)
    means, precision, _ = load_ood_stats(base_dir / args.ood_stats)

    known_features = extractor.extract(known_paths, transform, args.batch_size)
    unknown_features = extractor.extract(unknown_paths, transform, args.batch_size)
    known_maha = mahalanobis_scores(known_features, means, precision)
    unknown_maha = mahalanobis_scores(unknown_features, means, precision)

    # 2) P(unknown)（第三阶段，留一类模型）
    open_model = vgg16_openset(num_classes=3)
    open_model.load_state_dict(torch.load(base_dir / args.openset_weights,
                                          map_location='cpu'))
    open_model.to(device).eval()
    known_punknown = openset_unknown_probabilities(
        open_model, known_paths, transform, device, args.batch_size)
    unknown_punknown = openset_unknown_probabilities(
        open_model, unknown_paths, transform, device, args.batch_size)

    results = {}
    results['mahalanobis'] = evaluate_signal(
        'mahalanobis', known_maha, unknown_maha, holdout_classes, args.max_fpr)
    results['p_unknown'] = evaluate_signal(
        'p_unknown', known_punknown, unknown_punknown, holdout_classes, args.max_fpr)

    # 3) 融合：两个分数都转成 z-score 后取平均
    known_maha_z, unknown_maha_z = standardize(known_maha, unknown_maha)
    known_pu_z, unknown_pu_z = standardize(known_punknown, unknown_punknown)
    known_ensemble = 0.5 * known_maha_z + 0.5 * known_pu_z
    unknown_ensemble = 0.5 * unknown_maha_z + 0.5 * unknown_pu_z
    results['ensemble'] = evaluate_signal(
        'ensemble', known_ensemble, unknown_ensemble, holdout_classes, args.max_fpr)

    summary = {
        'holdout_unknown_classes': holdout_classes,
        'known_images': len(known_paths),
        'unknown_images': len(unknown_paths),
        'results': results,
    }
    summary_name = f'summary_fpr{int(round(args.max_fpr * 100)):02d}.json'
    (out_dir / summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    plt.figure(figsize=(8, 5))
    bins = np.linspace(-3, 6, 40)
    plt.hist(known_ensemble, bins=bins, alpha=0.6, label='Known (Cat/Dog)')
    plt.hist(unknown_ensemble, bins=bins, alpha=0.6, label='Unknown (unseen class)')
    plt.xlabel('Ensemble score (z-scored)')
    plt.ylabel('Number of images')
    plt.title('Ensemble score distribution on unseen class')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'ensemble_histogram.png', dpi=150)
    plt.close()

    print('\n===== 融合评估结果（完全没见过的类别） =====')
    for name in ('mahalanobis', 'p_unknown', 'ensemble'):
        info = results[name]
        metrics = info['best_metrics']
        print(f"[{name}] AUROC={info['auroc']:.4f} AP={info['average_precision']:.4f} "
              f"阈值={info['best_threshold']:.4f} "
              f"误报={metrics['known_false_unknown_rate']:.4f} "
              f"检出率={metrics['unknown_recall']:.4f} F1={metrics['f1']:.4f}")
    print(f'\n结果已保存到: {out_dir.resolve()}')


if __name__ == '__main__':
    main()

"""
evaluate_unknown.py —— 第一阶段“未知检测”评估脚本
====================================================
第一阶段的未知检测逻辑是：
    confidence = max(P(dog), P(cat))
    unknown_score = 1 - confidence
    confidence < unknown_threshold  -> 判为 Unknown

本脚本会批量评估这套逻辑的效果：

1. 已知类：从 data/test_set/test_set/{cats,dogs} 取猫狗图片；
2. 未知类：从 data/unknown_set/* 取兔子、牛、马等图片；
3. 扫描多个阈值，统计未知检出率、猫狗误报率、F1 等指标；
4. 用 unknown_score 计算 AUROC / AP，衡量“未知分数”本身的质量；
5. 输出 CSV 指标表、摘要 JSON/TXT，以及置信度分布图和阈值曲线图。

用法示例：
    python evaluate_unknown.py --weights models/01-trained-model.pth
    python evaluate_unknown.py --max_known_per_class 300 --max_fpr 0.05
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')          # 用非交互后端，避免评估时弹窗阻塞
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.model import vgg16_transfer_learning


def parse_args():
    parser = argparse.ArgumentParser(description='评估第一阶段的未知检测效果')
    parser.add_argument('--base_dir', type=str, default='.',
                        help='项目根目录，默认当前目录')
    parser.add_argument('--weights', type=str, default='models/01-trained-model.pth',
                        help='已训练模型权重路径')
    parser.add_argument('--known_dir', type=str,
                        default='data/test_set/test_set',
                        help='已知类目录（包含 cats/ 与 dogs/）')
    parser.add_argument('--unknown_dir', type=str, default='data/unknown_set',
                        help='未知类目录（每个子目录是一个未知类别）')
    parser.add_argument('--out_dir', type=str, default='reports/unknown_eval',
                        help='评估结果输出目录')
    parser.add_argument('--max_known_per_class', type=int, default=300,
                        help='每个已知类最多取多少张；0 表示全部使用')
    parser.add_argument('--max_unknown_per_class', type=int, default=0,
                        help='每个未知类最多取多少张；0 表示全部使用')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='推理时的 batch size')
    parser.add_argument('--max_fpr', type=float, default=0.05,
                        help='选择最佳阈值时允许的猫狗误报率上限')
    return parser.parse_args()


def build_transform():
    """与训练、推理阶段保持一致的数据预处理。"""
    return transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_model(weights_path: Path, device: torch.device):
    """加载与训练时结构一致的 VGG16 迁移学习模型，并读入权重。"""
    model = vgg16_transfer_learning()
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def collect_known_items(known_dir: Path, max_per_class: int):
    """收集已知类（猫、狗）图片路径与标签：猫=0，狗=1。"""
    items = []
    for class_name, label in (('cats', 0), ('dogs', 1)):
        files = sorted((known_dir / class_name).glob('*.jpg'))
        if max_per_class > 0:
            files = files[:max_per_class]
        items.extend((path, label) for path in files)
    return items


def collect_unknown_items(unknown_dir: Path, max_per_class: int):
    """收集全部未知类图片路径与类别名。"""
    items = []
    for class_dir in sorted(p for p in unknown_dir.iterdir() if p.is_dir()):
        files = sorted(class_dir.glob('*.jpg'))
        if max_per_class > 0:
            files = files[:max_per_class]
        items.extend((path, class_dir.name) for path in files)
    return items


def predict_dog_probabilities(model, paths, transform, device, batch_size):
    """批量推理，返回每张图片“是狗”的概率。"""
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            tensors = []
            for path in batch_paths:
                image = Image.open(path).convert('RGB')
                tensors.append(transform(image))
            batch = torch.stack(tensors).to(device)
            outputs = model(batch).squeeze(1)
            probabilities.extend(outputs.detach().cpu().numpy().tolist())
    return np.asarray(probabilities, dtype=np.float64)


def roc_auc_score(unknown_scores, is_unknown):
    """
    计算 AUROC：未知类为正样本，已知类为负样本。
    使用成对比较法，避免依赖 scikit-learn。
    """
    positives = unknown_scores[is_unknown == 1]
    negatives = unknown_scores[is_unknown == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float('nan')

    wins = 0.0
    for pos in positives:
        wins += float(np.sum(pos > negatives)) + 0.5 * float(np.sum(pos == negatives))
    return wins / (len(positives) * len(negatives))


def average_precision(unknown_scores, is_unknown):
    """计算 PR 曲线下的平均精度 AP（未知类为正样本）。"""
    order = np.argsort(-unknown_scores)
    labels = is_unknown[order]
    total_positives = int(labels.sum())
    if total_positives == 0:
        return float('nan')

    tp = np.cumsum(labels)
    precision = tp / (np.arange(len(labels)) + 1)
    recall = tp / total_positives
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.sum(np.diff(recall) * precision[1:]))


def sweep_thresholds(known_confidence, unknown_confidence, max_fpr):
    """扫描阈值，返回每个阈值下的指标以及推荐阈值。"""
    rows = []
    total_known = len(known_confidence)
    total_unknown = len(unknown_confidence)

    for threshold in np.arange(0.50, 1.00, 0.01):
        known_flagged = known_confidence < threshold
        unknown_detected = unknown_confidence < threshold

        known_fpr = float(known_flagged.sum()) / total_known
        unknown_recall = float(unknown_detected.sum()) / total_unknown

        tp = int(unknown_detected.sum())
        fp = int(known_flagged.sum())
        fn = total_unknown - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * unknown_recall / (precision + unknown_recall)
              if (precision + unknown_recall) > 0 else 0.0)
        balanced_accuracy = 0.5 * (unknown_recall + (1.0 - known_fpr))

        rows.append({
            'threshold': round(float(threshold), 4),
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


def plot_confidence_histogram(known_confidence, unknown_confidence, out_path):
    """画已知类与未知类的置信度分布直方图。"""
    plt.figure(figsize=(8, 5))
    bins = np.linspace(0.5, 1.0, 26)
    plt.hist(known_confidence, bins=bins, alpha=0.6, label='Known (Cat/Dog)')
    plt.hist(unknown_confidence, bins=bins, alpha=0.6, label='Unknown (Rabbit/Cow/Horse)')
    plt.xlabel('Confidence = max(P(dog), P(cat))')
    plt.ylabel('Number of images')
    plt.title('Confidence distribution: known vs unknown')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_threshold_curves(rows, best, out_path):
    """画阈值与未知检出率、猫狗误报率、F1 的关系曲线。"""
    thresholds = [r['threshold'] for r in rows]
    unknown_recall = [r['unknown_recall'] for r in rows]
    known_fpr = [r['known_false_unknown_rate'] for r in rows]
    f1 = [r['f1'] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, unknown_recall, label='Unknown recall')
    plt.plot(thresholds, known_fpr, label='Known false-unknown rate')
    plt.plot(thresholds, f1, label='F1 (unknown as positive)')
    plt.axvline(best['threshold'], color='red', linestyle='--',
                label=f"Best threshold = {best['threshold']:.2f}")
    plt.xlabel('Confidence threshold')
    plt.ylabel('Score')
    plt.title('Threshold sweep for unknown detection')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = load_model(base_dir / args.weights, device)
    transform = build_transform()

    known_items = collect_known_items(base_dir / args.known_dir, args.max_known_per_class)
    unknown_items = collect_unknown_items(base_dir / args.unknown_dir,
                                          args.max_unknown_per_class)
    if not known_items or not unknown_items:
        raise SystemExit('没有收集到图片，请检查 known_dir / unknown_dir 路径。')

    print(f'已知类图片: {len(known_items)} 张')
    print(f'未知类图片: {len(unknown_items)} 张')

    known_paths = [item[0] for item in known_items]
    known_labels = np.asarray([item[1] for item in known_items])
    unknown_paths = [item[0] for item in unknown_items]
    unknown_names = [item[1] for item in unknown_items]

    known_probs = predict_dog_probabilities(model, known_paths, transform,
                                            device, args.batch_size)
    unknown_probs = predict_dog_probabilities(model, unknown_paths, transform,
                                              device, args.batch_size)

    # 置信度与未知分数
    known_confidence = np.maximum(known_probs, 1.0 - known_probs)
    unknown_confidence = np.maximum(unknown_probs, 1.0 - unknown_probs)

    # 已知类闭集准确率（不考虑 Unknown 分支）
    known_predictions = (known_probs >= 0.5).astype(int)
    known_accuracy = float((known_predictions == known_labels).mean())

    unknown_scores = np.concatenate([1.0 - known_confidence, 1.0 - unknown_confidence])
    is_unknown = np.concatenate([np.zeros(len(known_confidence)),
                                 np.ones(len(unknown_confidence))])
    auroc = roc_auc_score(unknown_scores, is_unknown)
    ap = average_precision(unknown_scores, is_unknown)

    rows, best = sweep_thresholds(known_confidence, unknown_confidence, args.max_fpr)

    # 每个未知类在推荐阈值下的检出率
    per_class = {}
    unknown_names_array = np.asarray(unknown_names)
    for class_name in sorted(set(unknown_names)):
        mask = unknown_names_array == class_name
        detected = (unknown_confidence[mask] < best['threshold']).mean()
        per_class[class_name] = {
            'count': int(mask.sum()),
            'detected_at_best_threshold': round(float(detected), 4),
        }

    summary = {
        'device': str(device),
        'known_images': len(known_items),
        'unknown_images': len(unknown_items),
        'known_closed_set_accuracy': round(known_accuracy, 4),
        'auroc': round(float(auroc), 4),
        'average_precision': round(float(ap), 4),
        'best_threshold': best['threshold'],
        'best_metrics': best,
        'per_unknown_class': per_class,
    }

    # 保存阈值扫描表
    csv_path = out_dir / 'threshold_sweep.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 保存摘要
    (out_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    plot_confidence_histogram(known_confidence, unknown_confidence,
                             out_dir / 'confidence_histogram.png')
    plot_threshold_curves(rows, best, out_dir / 'threshold_curves.png')

    print('\n===== 评估结果 =====')
    print(f"已知类闭集准确率: {known_accuracy:.4f}")
    print(f"AUROC: {auroc:.4f}    AP: {ap:.4f}")
    print(f"推荐阈值: {best['threshold']:.2f} "
          f"(猫狗误报率 {best['known_false_unknown_rate']:.4f}, "
          f"未知检出率 {best['unknown_recall']:.4f}, F1 {best['f1']:.4f})")
    for class_name, stats in per_class.items():
        print(f"  {class_name}: {stats['count']} 张, "
              f"检出率 {stats['detected_at_best_threshold']:.4f}")
    print(f'\n结果已保存到: {out_dir.resolve()}')


if __name__ == '__main__':
    main()

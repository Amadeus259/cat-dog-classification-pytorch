"""
train_openset.py —— 第三阶段：带“未知类”的开放集训练
=====================================================
思路：把模型从“猫狗二分类”改成“猫 / 狗 / 未知 三分类”。

- 猫、狗来自 data/training_set/training_set
- 未知类来自 data/unknown_set（兔子、牛、马）
- 未知图片按比例切分：70% 参与训练，30% 留作验证
- VGG16 骨干冻结，只训练新的三分类头
- 训练结束后保存模型与指标，供 evaluate_unknown_v3.py 与 inference.py 使用

用法：
    python src/train_openset.py --epochs 6 --batch_size 32
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

_SRC_DIRECTORY = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIRECTORY.parent
for _path in (_PROJECT_ROOT, _SRC_DIRECTORY):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

from src.model import vgg16_openset
from src.openset_data import (CLASS_UNKNOWN, ImagePathDataset,
                              build_balanced_sampler, collect_known_items,
                              collect_unknown_items, split_unknown_items)
from evaluate_unknown import roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(description='第三阶段：开放集三分类训练')
    parser.add_argument('--known_dir', type=str,
                        default='data/training_set/training_set')
    parser.add_argument('--unknown_dir', type=str, default='data/unknown_set')
    parser.add_argument('--val_known_dir', type=str, default='data/test_set/test_set')
    parser.add_argument('--val_known_per_class', type=int, default=500)
    parser.add_argument('--unknown_train_ratio', type=float, default=0.7)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='openset-vgg16.pth')
    parser.add_argument('--out_dir', type=str, default='reports/openset_v3')
    parser.add_argument('--holdout_unknown_classes', type=str, default='',
                        help='完全留出的未知类别（逗号分隔，例如 rabbit）；'
                             '这些类别不参与训练，只用于测试全新类别')
    return parser.parse_args()


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


@torch.inference_mode()
def evaluate(model, known_loader, unknown_loader, device, max_fpr=0.05):
    """计算验证损失、闭集准确率、未知检测 AUROC 与最佳阈值。"""
    criterion = nn.CrossEntropyLoss()
    model.eval()

    known_loss, known_correct, known_total = 0.0, 0, 0
    known_scores, unknown_scores = [], []

    for images, labels in known_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        known_loss += criterion(logits, labels).item()
        probabilities = torch.softmax(logits, dim=1)
        known_scores.extend(probabilities[:, CLASS_UNKNOWN].cpu().numpy().tolist())
        # 闭集准确率只统计猫狗两类
        known_predictions = probabilities[:, :2].argmax(dim=1)
        known_correct += int((known_predictions == labels).sum().item())
        known_total += len(labels)

    for images, labels in unknown_loader:
        images = images.to(device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        unknown_scores.extend(probabilities[:, CLASS_UNKNOWN].cpu().numpy().tolist())

    known_scores = np.asarray(known_scores)
    unknown_scores = np.asarray(unknown_scores)
    scores = np.concatenate([known_scores, unknown_scores])
    is_unknown = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    auroc = roc_auc_score(scores, is_unknown)

    best = None
    for threshold in np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 201))):
        known_flagged = known_scores >= threshold
        unknown_detected = unknown_scores >= threshold
        known_fpr = float(known_flagged.mean())
        if known_fpr > max_fpr:
            continue
        unknown_recall = float(unknown_detected.mean())
        tp = int(unknown_detected.sum())
        fp = int(known_flagged.sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * unknown_recall / (precision + unknown_recall)
              if (precision + unknown_recall) > 0 else 0.0)
        candidate = {
            'threshold': round(float(threshold), 4),
            'known_false_unknown_rate': round(known_fpr, 4),
            'unknown_recall': round(unknown_recall, 4),
            'f1': round(f1, 4),
        }
        if best is None or (candidate['f1'], candidate['unknown_recall']) > \
                (best['f1'], best['unknown_recall']):
            best = candidate

    return {
        'known_loss': round(known_loss / max(len(known_loader), 1), 4),
        'known_closed_set_accuracy': round(known_correct / max(known_total, 1), 4),
        'unknown_auroc': round(float(auroc), 4),
        'best_threshold_metrics': best,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    base_dir = Path('.')

    holdout_classes = [name.strip() for name in
                       args.holdout_unknown_classes.split(',') if name.strip()]
    suffix = f"_holdout_{'-'.join(holdout_classes)}" if holdout_classes else ''
    out_dir = base_dir / (args.out_dir + suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model_name
    if holdout_classes and args.model_name == 'openset-vgg16.pth':
        model_name = f"openset-vgg16-holdout-{'-'.join(holdout_classes)}.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'训练设备: {device}')

    train_transform, eval_transform = build_transforms()

    known_items = collect_known_items(base_dir / args.known_dir)
    unknown_items = collect_unknown_items(base_dir / args.unknown_dir)

    # 留一类实验：把指定类别从训练/验证未知集中完全剔除
    if holdout_classes:
        holdout_items = [item for item in unknown_items
                         if item[0].parent.name in holdout_classes]
        unknown_items = [item for item in unknown_items
                         if item[0].parent.name not in holdout_classes]
    else:
        holdout_items = []

    unknown_train, unknown_val = split_unknown_items(
        unknown_items, train_ratio=args.unknown_train_ratio, seed=args.seed)

    train_items = known_items + unknown_train
    print(f'训练集: 猫狗 {len(known_items)} 张 + 未知 {len(unknown_train)} 张')
    print(f'留出未知验证集: {len(unknown_val)} 张')
    if holdout_classes:
        print(f'完全留出的类别 {holdout_classes}: {len(holdout_items)} 张'
              f'（不参与训练，仅用于测试）')

    val_known_items = collect_known_items(base_dir / args.val_known_dir,
                                          max_per_class=args.val_known_per_class)

    train_dataset = ImagePathDataset(train_items, train_transform)
    sampler = build_balanced_sampler(train_items)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=sampler, drop_last=True, num_workers=0)
    known_val_loader = DataLoader(ImagePathDataset(val_known_items, eval_transform),
                                  batch_size=args.batch_size, shuffle=False)
    unknown_val_loader = DataLoader(ImagePathDataset(unknown_val, eval_transform),
                                    batch_size=args.batch_size, shuffle=False)

    model = vgg16_openset(num_classes=3).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_auroc = -1.0
    model_path = base_dir / 'models' / model_name
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in tqdm(range(args.epochs), desc='epochs'):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += int((logits.argmax(dim=1) == labels).sum().item())
            train_total += len(labels)

        validation = evaluate(model, known_val_loader, unknown_val_loader, device)
        record = {
            'epoch': epoch + 1,
            'train_loss': round(train_loss / max(len(train_loader), 1), 4),
            'train_accuracy': round(train_correct / max(train_total, 1), 4),
            **validation,
        }
        history.append(record)
        print(f"Epoch {epoch + 1}: train_loss={record['train_loss']} "
              f"train_acc={record['train_accuracy']} "
              f"known_acc={record['known_closed_set_accuracy']} "
              f"AUROC={record['unknown_auroc']}")

        if record['unknown_auroc'] > best_auroc:
            best_auroc = record['unknown_auroc']
            torch.save(model.state_dict(), model_path)
            print(f'  保存最佳模型 -> {model_path}')

    summary = {
        'model_path': str(model_path),
        'epochs': args.epochs,
        'holdout_unknown_classes': holdout_classes,
        'holdout_items': len(holdout_items),
        'unknown_train': len(unknown_train),
        'unknown_val': len(unknown_val),
        'best_auroc': best_auroc,
        'history': history,
    }
    (out_dir / 'metrics.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if holdout_items:
        (out_dir / 'holdout_items.json').write_text(
            json.dumps([{'path': str(path), 'class': path.parent.name}
                        for path, _ in holdout_items], ensure_ascii=False, indent=2),
            encoding='utf-8')
    print(f'\n训练完成，指标已保存到 {out_dir / "metrics.json"}')


if __name__ == '__main__':
    main()

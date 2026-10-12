"""
openset_data.py —— 第三阶段：开放集（猫/狗/未知）数据准备
==========================================================
第三阶段的思路是让模型显式学习一个 Unknown 类别：

    0 -> cats
    1 -> dogs
    2 -> unknown（兔子、牛、马等）

本模块负责：
1. 收集猫、狗、未知类图片路径与标签；
2. 把未知类图片按比例切分成“训练用未知”和“留出验证用未知”；
3. 提供通用的 ImagePathDataset 与类别平衡采样器。
"""
import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


CLASS_CAT = 0
CLASS_DOG = 1
CLASS_UNKNOWN = 2
CLASS_NAMES = {CLASS_CAT: 'cats', CLASS_DOG: 'dogs', CLASS_UNKNOWN: 'unknown'}


class ImagePathDataset(Dataset):
    """最小图片数据集：items 是 (图片路径, 标签) 的列表。"""

    def __init__(self, items, transform):
        self.items = list(items)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        image = Image.open(path).convert('RGB')
        return self.transform(image), label


def collect_known_items(known_dir: Path, max_per_class: int = 0):
    """收集猫狗训练图片：cats -> 0，dogs -> 1。"""
    items = []
    for class_name, label in (('cats', CLASS_CAT), ('dogs', CLASS_DOG)):
        files = sorted((known_dir / class_name).glob('*.jpg'))
        if max_per_class > 0:
            files = files[:max_per_class]
        items.extend((path, label) for path in files)
    return items


def collect_unknown_items(unknown_dir: Path, max_per_class: int = 0):
    """收集未知类图片：每个子目录一个未知类别，标签统一为 2。"""
    items = []
    for class_dir in sorted(p for p in unknown_dir.iterdir() if p.is_dir()):
        files = sorted(class_dir.glob('*.jpg'))
        if max_per_class > 0:
            files = files[:max_per_class]
        items.extend((path, CLASS_UNKNOWN) for path in files)
    return items


def split_unknown_items(unknown_items, train_ratio: float = 0.7, seed: int = 42):
    """
    把未知类图片切分成训练用与验证用两部分。
    返回 (unknown_train, unknown_val)。
    """
    items = list(unknown_items)
    rng = random.Random(seed)
    rng.shuffle(items)
    split_index = int(len(items) * train_ratio)
    return items[:split_index], items[split_index:]


def build_balanced_sampler(items):
    """
    为类别不平衡的数据构建采样器：
    未知类样本少，会被更频繁地采样，避免模型只学会猫狗。
    """
    labels = [label for _, label in items]
    class_counts = {label: labels.count(label) for label in set(labels)}
    weights = [1.0 / class_counts[label] for _, label in items]
    return WeightedRandomSampler(weights, num_samples=len(items), replacement=True)

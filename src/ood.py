"""
ood.py —— 第二阶段：基于特征距离的未知检测
==========================================
第一阶段的“最大概率阈值”在评估中 AUC 只有 0.61，基本无法发现未知类。
原因是二分类器对所有输入都会给出接近 1 的置信度。

本模块改用特征空间的距离来衡量“像不像已知类”：

1. 用 VGG16 的卷积部分把图片编码成 512 维特征向量；
2. 用训练集的猫狗特征，为每个已知类计算特征均值（类中心）和共享协方差；
3. 对每张图片计算它到最近类中心的马氏距离 / 余弦距离；
4. 距离越大，说明越不像已知类，越可能是未知。

距离分数与阈值比较的方式是：
    score >= threshold  -> Unknown
    score <  threshold  -> Cat / Dog
"""
import numpy as np
import torch
from PIL import Image


class FeatureExtractor:
    """把 VGG16 迁移学习模型当作特征提取器使用。"""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def extract(self, paths, transform, batch_size: int = 32) -> np.ndarray:
        """
        批量提取特征向量。
        输入图片路径列表，返回形状为 (N, 512) 的 numpy 数组。
        """
        vectors = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            tensors = []
            for path in batch_paths:
                image = Image.open(path).convert('RGB')
                tensors.append(transform(image))
            batch = torch.stack(tensors).to(self.device)

            # VGG16: features -> avgpool -> flatten 得到 512 维特征
            features = self.model.features(batch)
            features = self.model.avgpool(features)
            features = torch.flatten(features, start_dim=1)
            vectors.append(features.detach().cpu().numpy())

        return np.concatenate(vectors, axis=0).astype(np.float64)


def fit_mahalanobis(features: np.ndarray, labels: np.ndarray,
                    num_classes: int = 2, reg: float = 1e-3):
    """
    用训练特征拟合每个类别的均值和共享协方差矩阵。
    返回 (means, precision)，precision 是协方差矩阵的伪逆。
    """
    means = np.stack([
        features[labels == class_id].mean(axis=0)
        for class_id in range(num_classes)
    ])

    dimension = features.shape[1]
    covariance = np.zeros((dimension, dimension), dtype=np.float64)
    for class_id in range(num_classes):
        diff = features[labels == class_id] - means[class_id]
        covariance += diff.T @ diff

    # 协方差矩阵的归一化自由度：样本数 - 类别数
    dof = max(len(features) - num_classes, 1)
    covariance /= dof

    # 加上一个很小的对角项，避免矩阵奇异导致求逆失败
    covariance += np.eye(dimension) * reg
    precision = np.linalg.pinv(covariance)
    return means, precision


def mahalanobis_scores(features: np.ndarray, means: np.ndarray,
                       precision: np.ndarray) -> np.ndarray:
    """
    计算每张图片到最近类中心的马氏距离。
    距离越大，越可能是未知类别。
    """
    best = np.full(len(features), np.inf, dtype=np.float64)
    for class_id in range(len(means)):
        diff = features - means[class_id]
        distance = np.einsum('ij,jk,ik->i', diff, precision, diff)
        best = np.minimum(best, distance)
    return np.sqrt(np.maximum(best, 0.0))


def fit_cosine(features: np.ndarray, labels: np.ndarray, num_classes: int = 2):
    """计算每个类别的归一化特征中心，用于余弦距离。"""
    centroids = []
    for class_id in range(num_classes):
        center = features[labels == class_id].mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-12)
        centroids.append(center)
    return np.stack(centroids)


def cosine_scores(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """计算每张图片到最近类中心的余弦距离（1 - 最大余弦相似度）。"""
    normalized = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    similarity = normalized @ centroids.T          # (N, num_classes)
    return 1.0 - similarity.max(axis=1)


def save_ood_stats(path, means: np.ndarray, precision: np.ndarray,
                   centroids: np.ndarray) -> None:
    """把拟合好的统计量保存成 npz，供推理脚本复用。"""
    np.savez(path, means=means, precision=precision, centroids=centroids)


def load_ood_stats(path):
    """读取 npz 里的均值、协方差伪逆与余弦中心。"""
    data = np.load(path)
    return data['means'], data['precision'], data['centroids']

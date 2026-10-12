"""
inference.py —— 推理脚本（用训练好的模型预测单张图片）
=======================================================
运行方式：在项目根目录执行  python inference.py

流程：
1. 定义与训练时一致的预处理流水线（224x224、转张量、ImageNet 归一化）；
2. 构建与训练时相同的模型结构（VGG16 迁移学习），并加载 models/ 下保存的权重；
3. 从 data/test_set/test_set/cats 目录选一张猫图；
4. 调用 src.engine.predict 得到预测标签（Cat / Dog），并弹窗显示结果。
"""
from src.utils import plot_accuracy_loss, plot_inference
from src.engine import predict
from pathlib import Path
from PIL import Image
from torchvision import transforms
from src.model import SimpleBinaryClassifier, vgg16_transfer_learning

import numpy as np 
import cv2
import torch
import random
import argparse

if __name__ == '__main__':
    
    # 以项目根目录为基准路径
    BASE_PATH = Path('./')

    # 支持从命令行传入图片路径和未知阈值
    parser = argparse.ArgumentParser(description='猫狗分类推理，并检测未知类别')
    parser.add_argument('--image', type=str, default=None,
                        help='待预测图片路径；不填则从测试集中随机选一张猫/狗图')
    parser.add_argument('--unknown_threshold', type=float, default=0.85,
                        help='置信度低于该值时判为 Unknown，默认 0.85')
    parser.add_argument('--method', type=str, default='confidence',
                        choices=['confidence', 'mahalanobis', 'openset'],
                        help='未知检测方法：confidence（第一阶段）、mahalanobis（第二阶段）'
                             '或 openset（第三阶段三分类模型）')
    parser.add_argument('--ood_threshold', type=float, default=None,
                        help='马氏距离阈值；不填则自动读取评估结果里的推荐阈值')
    parser.add_argument('--ood_stats', type=str,
                        default='reports/unknown_eval_v2/ood_stats.npz',
                        help='第二阶段拟合好的 OOD 统计量路径')
    parser.add_argument('--ood_summary', type=str,
                        default='reports/unknown_eval_v2/summary.json',
                        help='第二阶段评估摘要，用于读取推荐阈值')
    parser.add_argument('--openset_weights', type=str,
                        default='models/openset-vgg16.pth',
                        help='第三阶段三分类模型权重路径')
    parser.add_argument('--openset_threshold', type=float, default=None,
                        help='三分类模型的 P(unknown) 阈值；不填则自动读取评估结果')
    parser.add_argument('--openset_summary', type=str,
                        default='reports/unknown_eval_v3/summary.json',
                        help='第三阶段评估摘要，用于读取推荐阈值')
    args = parser.parse_args()

    if args.image:
        IMAGE = Path(args.image)
    else:
        cat_files = list((BASE_PATH / 'data/test_set/test_set/cats').glob('*.jpg'))
        dog_files = list((BASE_PATH / 'data/test_set/test_set/dogs').glob('*.jpg'))
        IMAGE = random.choice(cat_files + dog_files)
    
    # 预处理流水线：必须与训练时保持一致，否则预测效果会受影响
    # Create transforms
    data_transform = transforms.Compose([
        # Resize the images to 224x224
        transforms.Resize(size=(224, 224)),
        # Turn the image into a torch.Tensor
        transforms.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
        # Normalize
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    
    # 输出值到类别的映射：0 -> Cat（猫），1 -> Dog（狗）
    class_dict = {0: 'Cat', 1: 'Dog'}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.method in ('confidence', 'mahalanobis'):
        # 一、二阶段共用的二分类迁移学习模型
        SBC = vgg16_transfer_learning()
        state_dict = torch.load(BASE_PATH / 'models/01-trained-model.pth',
                                map_location='cpu')
        SBC.load_state_dict(state_dict)

    if args.method == 'confidence':
        # 第一阶段：基于最大概率阈值
        im, label, confidence, is_unknown = predict(
            SBC, IMAGE, data_transform, class_dict, True,
            unknown_threshold=args.unknown_threshold
        )
        print(f"Result: {label} | confidence={confidence:.4f} | is_unknown={is_unknown}")
    elif args.method == 'mahalanobis':
        # 第二阶段：基于 VGG16 特征空间的马氏距离
        import json
        from src.ood import FeatureExtractor, load_ood_stats, mahalanobis_scores

        SBC = SBC.to(device)

        stats_path = BASE_PATH / args.ood_stats
        if not stats_path.exists():
            raise SystemExit(
                f'找不到 {stats_path}，请先运行: python evaluate_unknown_v2.py')
        means, precision, _ = load_ood_stats(stats_path)

        threshold = args.ood_threshold
        if threshold is None:
            summary_path = BASE_PATH / args.ood_summary
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
                threshold = summary['mahalanobis']['best_threshold']
            else:
                threshold = 47.0
        print(f'使用马氏距离，阈值 = {threshold:.4f}')

        im = cv2.cvtColor(cv2.imread(str(IMAGE)), cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(im)
        tensor = data_transform(pil_image).unsqueeze(0).to(device)
        with torch.inference_mode():
            p_dog = float(SBC(tensor).item())

        extractor = FeatureExtractor(SBC, device)
        features = extractor.extract([IMAGE], data_transform, batch_size=1)
        score = float(mahalanobis_scores(features, means, precision)[0])

        is_unknown = score >= threshold
        label = 'Unknown' if is_unknown else class_dict[1 if p_dog >= 0.5 else 0]
        print(f'Result: {label} | p_dog={p_dog:.4f} | '
              f'mahalanobis={score:.4f} | is_unknown={is_unknown}')
        plot_inference(im, label)
    else:
        # 第三阶段：猫 / 狗 / 未知 三分类模型
        import json
        from src.model import vgg16_openset

        openset_path = BASE_PATH / args.openset_weights
        if not openset_path.exists():
            raise SystemExit(
                f'找不到 {openset_path}，请先运行: python src/train_openset.py')

        open_model = vgg16_openset(num_classes=3)
        open_model.load_state_dict(torch.load(openset_path, map_location='cpu'))
        open_model.to(device)
        open_model.eval()

        threshold = args.openset_threshold
        if threshold is None:
            summary_path = BASE_PATH / args.openset_summary
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
                threshold = summary['best_threshold']
            else:
                threshold = 0.85
        print(f'使用三分类开放集模型，P(unknown) 阈值 = {threshold:.4f}')

        im = cv2.cvtColor(cv2.imread(str(IMAGE)), cv2.COLOR_BGR2RGB)
        tensor = data_transform(Image.fromarray(im)).unsqueeze(0).to(device)
        with torch.inference_mode():
            probabilities = torch.softmax(open_model(tensor), dim=1)[0]

        p_cat, p_dog, p_unknown = (float(probabilities[0]), float(probabilities[1]),
                                   float(probabilities[2]))
        is_unknown = p_unknown >= threshold
        if is_unknown:
            label = 'Unknown'
        else:
            label = class_dict[1 if p_dog >= p_cat else 0]
        print(f'Result: {label} | p_cat={p_cat:.4f} | p_dog={p_dog:.4f} | '
              f'p_unknown={p_unknown:.4f} | is_unknown={is_unknown}')
        plot_inference(im, label)
    
    
    # ======== 以下是被注释掉的备用代码 ========
    # 可读取训练时保存的 .npz 指标文件，绘制损失/准确率曲线
    # data = np.load('imclfv1-0-metrics.npz')
    # plot_accuracy_loss(data['train_acc'], data['test_acc'], 'Accuracy')
    # plot_accuracy_loss(data['train_loss'], data['test_loss'], 'Loss')

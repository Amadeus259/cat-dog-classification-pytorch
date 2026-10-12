"""
train.py —— 训练入口脚本
=========================
本文件是整个项目的训练主入口，在项目根目录执行：

    python src/train.py

主要流程：
1. 解析命令行超参数（batch size、学习率、epoch 数、目录、模型名等）；
2. 定位 data/ 下的训练集与验证集目录，创建 DataLoader；
3. 默认使用 VGG16 迁移学习模型（代码中注释掉了自定义 CNN 的选项）；
4. 调用 engine.train 完成训练与验证，并按条件把权重保存到 models/；
5. 把每个 epoch 的指标保存为 .npz 文件，最后绘制训练/验证准确率曲线。
"""
# --- 让脚本无论用哪种方式运行都能正确找到模块 ---
# 支持两种运行方式：
#   1) 在项目根目录执行  python src/train.py
#   2) 在项目根目录执行  python -m src.train
# 原因是 train.py 里用的是 import data_setup/engine/...（需要 src 目录在搜索路径中），
# 而 engine.py 里用的是 from src.utils import ...（需要项目根目录在搜索路径中）。
import sys
from pathlib import Path

_SRC_DIRECTORY = Path(__file__).resolve().parent          # 项目里的 src 目录
_PROJECT_ROOT = _SRC_DIRECTORY.parent                     # 项目根目录
for _path in (_PROJECT_ROOT, _SRC_DIRECTORY):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

import torch 
import data_setup, engine, model, utils 

import argparse
import numpy as np

from torchvision import transforms
from pathlib import Path



# 固定随机种子，保证训练结果可复现
torch.manual_seed(42)


if __name__ == '__main__':
    
    # ===================== 1. 命令行参数 =====================
    # Create the parser
    parser = argparse.ArgumentParser(description='Input Parameters')

    # Add arguments
    # 全部超参数都可以通过命令行传入，例如 --batch_size 64 --num_epochs 30
    parser.add_argument('--batch_size', type=int, help='Batch Size')
    parser.add_argument('--learning_rate', type=float, help='Learning Rate')
    parser.add_argument('--num_epochs', type=int, help='Number of Epochs')
    parser.add_argument('--train_dir', type=str, help='Train Directory')
    parser.add_argument('--test_dir', type=str, help='Test Directory')
    parser.add_argument('--model_name', type=str, help='Model filename')
    parser.add_argument('--verbose', action='store_true', help='Output verbosity')

    # Parse the arguments
    args = parser.parse_args()
    # ===================== 2. 默认超参数 =====================
    # Setup hyperparameters
    NUM_EPOCHS = 20
    BATCH_SIZE = 32    
    LEARNING_RATE = 0.001
    
    # 命令行有传入时，用传入值覆盖默认值
    if args.num_epochs:
        NUM_EPOCHS = args.num_epochs
        
    if args.batch_size:
        BATCH_SIZE = args.batch_size 
        
    if args.learning_rate:
        LEARNING_RATE = args.learning_rate
        
    # ===================== 3. 数据目录 =====================
    BASE_DIRECTORY = Path('./')
    
    DATA_DIRECTORY = BASE_DIRECTORY / 'data'

    # Setup directories
    # Kaggle 数据解压后，训练/测试集目录里还嵌套了一层同名子目录
    train_dir = "training_set/training_set"
    test_dir = "test_set/test_set"
    
    # 也可用命令行参数指定自定义目录
    if args.train_dir:
        train_dir = args.train_dir
        
    if args.test_dir:
        test_dir = args.test_dir
        
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = '01-trained-model.pth'
        
    # 大于 0 表示每隔这么多 epoch 尝试保存一次模型
    save_model_after_epochs = 10
        
    TRAINING_DATA_DIRECTORY = DATA_DIRECTORY / train_dir
    TEST_DATA_DIRECTORY = DATA_DIRECTORY / test_dir

    # ===================== 4. 计算设备 =====================
    # Setup target device
    # 有 CUDA 显卡就用 GPU，否则退回 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Training device: {device}')

    # ===================== 5. 图像预处理 =====================
    # Create transforms
    # 预处理必须与模型输入尺寸一致：224x224、转张量、ImageNet 归一化
    data_transform = transforms.Compose([
        # Resize the images to 224x224
        transforms.Resize(size=(224, 224)),
        # Turn the image into a torch.Tensor
        transforms.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
        # Normalize
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    # ===================== 6. 创建 DataLoader =====================
    # Create DataLoaders with help from data_setup.py
    # 训练集用全部图片；验证集取测试目录中每类前 800 张
    train_dl = data_setup.get_data_loaders(TRAINING_DATA_DIRECTORY, -1, BATCH_SIZE)
    validation_dl = data_setup.get_data_loaders(TEST_DATA_DIRECTORY, 800, BATCH_SIZE)
        
    # ===================== 7. 模型 =====================
    # Create model with help from model_builder.py
    # 默认使用 VGG16 迁移学习；如需自定义 CNN，取消下一行注释并注释掉 SBC 行
    # SBC = model.SimpleBinaryClassifier().to(device)
    SBC = model.vgg16_transfer_learning().to(device)

    # ===================== 8. 损失函数与优化器 =====================
    # Set loss and optimizer
    # 二分类问题用 BCELoss（配合模型末端的 Sigmoid 输出）
    loss_fn = torch.nn.BCELoss()
    # Adam 优化器；weight_decay 是 L2 正则化，帮助抑制过拟合
    optimizer = torch.optim.Adam(SBC.parameters(),
                                lr=LEARNING_RATE, weight_decay=1e-5)

    # ===================== 9. 开始训练 =====================
    # Start training with help from engine.py
    # engine.train 内部按 epoch 循环训练 + 验证，并返回各指标
    accuracy_loss_metrics = engine.train(model=SBC,
                                         train_dataloader=train_dl,
                                         validation_dataloader=validation_dl,
                                         optimizer=optimizer,
                                         loss_fn=loss_fn,                
                                         epochs=NUM_EPOCHS,
                                         save_model_after_epochs=save_model_after_epochs, 
                                         model_name=model_name,
                                         device=device,
                                         BASE_PATH=BASE_DIRECTORY)

    # ===================== 10. 保存指标并绘图 =====================
    # 模型名去掉 .pth 后拼上 -metrics.npz，例如 01-trained-model-metrics.npz
    model_metrics_filename = model_name.split('.pth')[0] + '-metrics.npz'
    # 把训练/验证 loss 与 acc 序列打包保存成 numpy 压缩文件
    np.savez(model_metrics_filename, **accuracy_loss_metrics)
    
    # 弹出准确率曲线窗口（训练 vs 验证）
    utils.plot_accuracy_loss(accuracy_loss_metrics['train_acc'],
                             accuracy_loss_metrics['test_acc'],
                             'Accuracy')

"""
model.py —— 模型定义模块
========================
提供两种用于猫狗二分类的模型：

1. SimpleBinaryClassifier：从零训练的轻量 CNN
   （3 层卷积 + 最大池化 + Dropout，最后接 1 个神经元 + Sigmoid 输出 0~1 的概率）。
2. vgg16_transfer_learning()：基于预训练 VGG16 的迁移学习模型
   （冻结骨干网络权重，替换全局池化与分类头，使其适配二分类任务）。

本文件也可以单独运行（python src/model.py），用 torchinfo 打印模型结构摘要。
"""
import torch

# 把 torch 预训练权重的缓存目录固定到 D 盘，
# 避免默认下载到 C 盘的 .cache 目录。训练和推理脚本都会导入本模块，
# 因此都会自动使用这个缓存位置。
torch.hub.set_dir(r"D:\torch_cache")

import torch.nn as nn
from torchinfo import summary
from torchvision import models

class SimpleBinaryClassifier(nn.Module):
    '''
    Performs binary classification using convolutional layers
    '''

    # 输入约定为 224x224 的 RGB 图片，张量形状为 (B, 3, 224, 224)
    def __init__(self):
        super(SimpleBinaryClassifier, self).__init__()
        # starting with image size 3, 224, 224 (C, H, W)

        # 卷积块 1：3 -> 8 通道；padding=1 保持宽高为 224，池化后降为 112
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1
            ), # (8, 224, 224) <---- (w_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1 = (224 + 2 - 3)/1 + 1 = 224
            # ReLU 激活函数：给网络引入非线性
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  #(8, 112, 112) <----
            # Dropout：训练时随机丢弃部分神经元，防止过拟合
            nn.Dropout()
        )
        
        # 卷积块 2：8 -> 16 通道；池化后由 112 降为 56
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ), # (16, 112, 112) <--- (112 + 4 - 5)/1 + 1 = 112
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=2), # (16, 56, 56)
            nn.Dropout(0.5)
        )
        
        # 卷积块 3：16 -> 32 通道；池化后由 56 降为 28
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ), # (32, 56, 56) <--- (56 + 4 - 5)/1 + 1 = 56
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # (32, 28, 28)
            nn.Dropout(0.5)
        )
        
        # 分类头：把 32x28x28 的特征图展平后接全连接层，
        # 输出 1 个值，并用 Sigmoid 压缩到 0~1，表示“是狗”的概率
        self.fc1 = nn.Sequential(nn.Linear(32*28*28, 1), # binary classifier ----> 1 output
                                 nn.Sigmoid()
        )
        
    def forward(self, x):
        # 依次经过 3 个卷积块逐层提取图像特征
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # flatten the output of conv2 to (batch_size, C*H*W=64*28*28)
        # 将 (B, C, H, W) 展平为 (B, C*H*W)，才能输入全连接层
        x = x.view(x.size(0), -1)
        # 全连接 + Sigmoid，输出形状为 (B, 1) 的二分类概率
        x = self.fc1(x)
        return x
    
def vgg16_transfer_learning():
    # 加载在 ImageNet 上预训练好的 VGG16 权重
    # 说明：旧写法 models.vgg16(pretrained=True) 已被 torchvision 废弃，
    # 新版推荐用 weights=VGG16_Weights.IMAGENET1K_V1 来指定预训练权重
    # Get VGG16 model from torchvision models
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    # Freeze model parameters
    # 冻结骨干网络全部参数：迁移学习阶段只训练自定义分类头，可显著减少计算量
    for params in model.parameters():
        params.requires_grad = False
    # Update average pooling and classifier block
    # 把原来的平均池化替换为自适应池化，无论输入特征多大都输出 1x1
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    # 替换掉原 1000 类分类器，构造自己的二分类头：
    # 512 -> 128 -> 1，最后用 Sigmoid 输出“是狗”的概率
    model.classifier = nn.Sequential(nn.Flatten(),
                                    nn.Linear(512, 128),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(128, 1),
                                    nn.Sigmoid()) # as we are solving a binary classification (cats & dogs)
    return model


def vgg16_openset(num_classes: int = 3):
    """
    第三阶段使用的开放集模型：VGG16 骨干 + 猫/狗/未知多分类头。

    输出 num_classes 个 logits（不接 Sigmoid/Softmax），
    训练时配合 CrossEntropyLoss 使用。
    """
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for params in model.parameters():
        params.requires_grad = False

    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model
    
    
if __name__ == '__main__':
    # 单独运行时默认加载 VGG16 迁移学习模型，打印其结构摘要
    # model = SimpleBinaryClassifier()
    model = vgg16_transfer_learning()
    print(summary(model, (1, 3, 224, 224)))
        

"""
utils.py —— 通用工具函数模块
============================
包含三个与训练/推理配套的辅助函数：

1. save_model: 把模型权重保存为 .pth/.pt 文件到 models/ 目录；
2. plot_accuracy_loss: 绘制训练集与验证集的准确率或损失曲线；
3. plot_inference: 显示推理用的图片及其预测标签。
"""
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def save_model(model: torch.nn.Module, model_name: str, BASE_PATH: Path = Path('./')):
    '''
    Saves PyTorch model to models directory
    
    Inputs:
    
    model:      model to save
    model_name: model filename
    BASE_PATH:  directory where models subdirectory resides
    '''
    
    # 校验文件名必须以 .pt 或 .pth 结尾，避免保存成无法识别的格式
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    # 模型统一保存到 BASE_PATH/models/ 目录下
    model_save_to = BASE_PATH / f'models/{model_name}'
    
    # save the model's state_dict()
    # 只保存模型权重字典 state_dict（加载时需要先重建相同结构的模型）
    torch.save(model.state_dict(), model_save_to)
    print('Model saved successfully')
    
def plot_accuracy_loss(train_data, valid_data, title):
    '''
    Plots epochs versus accuracy/loss
    '''
    
    # train_data/valid_data 分别是训练与验证的指标序列；以较短的序列长度作为横轴
    training_data, validation_data = len(train_data), len(valid_data)
    print(training_data)
    num_epochs = training_data
    if num_epochs == 0:
        num_epochs = validation_data
        if num_epochs == 0:
            # 两组数据都为空时无法绘图，直接返回
            return
    # 横轴：从 0 到 epoch 数-1
    epochs = list(range(num_epochs))
    if training_data:
        # 传入了训练数据才绘制训练曲线（绿色标记）
        plt.plot(epochs, train_data, '-sk', markerfacecolor='g', alpha=0.3, label='Training')
    if validation_data:
        # 传入了验证数据才绘制验证曲线（蓝色标记）
        plt.plot(epochs, valid_data, '-sk', markerfacecolor='b', alpha=0.3, label='Validation')
    # 添加图例、网格、坐标轴标签后显示
    plt.legend()
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel(title)
    plt.show()
    
def plot_inference(image, label):
    # 显示推理图片，并隐藏坐标轴刻度
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    # 标题中显示预测标签（Cat 或 Dog）
    plt.title(f'Prediction: {label.capitalize()}')
    plt.show()

"""
engine.py —— 训练 / 验证 / 推理引擎
====================================
本文件包含训练流程中最核心的几个函数：

- train_step: 在训练集 DataLoader 上跑一个 epoch（前向 + 反向传播 + 更新参数）；
- test_step: 在验证/测试集 DataLoader 上评估一个 epoch（只前向，不更新参数）；
- predict: 用训练好的模型对单张图片做预测并给出标签；
- train: 循环执行 train_step 与 test_step，记录指标，并按条件保存模型权重。
"""
import torch 
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from typing import Dict, List, Tuple 
from pathlib import Path
from src.utils import save_model, plot_inference
from torchvision import transforms

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    
    '''
    Trains a PyTorch model for a single epoch.
    
    Inputs:
    
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").
    
    '''


    # 在训练集上执行一个 epoch 的完整训练流程
    # Put model in train mode
    # 切换到训练模式：启用 Dropout、BatchNorm 的训练时行为
    model.train()

    # Setup train loss and train accuracy values
    # 累计本 epoch 的总损失与总准确率，最后除以 batch 数得到平均
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    # 逐 batch 遍历整个训练集
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        # 把数据和标签搬到 GPU（或 CPU）上
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        # 前向传播：模型输出每张图“是狗”的概率，形状为 (batch, 1)
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        # 计算当前 batch 的损失（BCELoss）并累加到总损失中
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        # 清空上一次反向传播留下的梯度，防止梯度累加
        optimizer.zero_grad()

        # 4. Loss backward
        # 反向传播：根据损失计算每个参数的梯度
        loss.backward()

        # 5. Optimizer step
        # 用梯度更新模型参数
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        # 二分类阈值取 0.5：输出 > 0.5 判为狗(1)，否则判为猫(0)
        y_pred_class = y_pred > 0.5
        # 统计本 batch 预测正确的样本数 / 样本总数，累加到总准确率
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    # 除以 batch 数量，得到本 epoch 的平均损失和平均准确率
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    '''
        Trains a PyTorch model for a single epoch.
        
        Inputs:
        
            model: A PyTorch model to be trained.
            dataloader: A DataLoader instance for the model to be trained on.
            loss_fn: A PyTorch loss function to minimize.
            optimizer: A PyTorch optimizer to help minimize the loss function.
            device: A target device to compute on (e.g. "cuda" or "cpu").
        
    '''
    # Put model in eval mode
    # 注意：上方英文文档字符串疑似从 train_step 复制而来，本函数实际执行的是【验证/测试】，
    # 只做前向评估，不更新模型参数。
    # 在验证/测试集上评估模型
    # 切换到评估模式：关闭 Dropout，BatchNorm 使用全局统计量
    model.eval() 

    # Setup test loss and test accuracy values
    # 累计总损失与总准确率
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    # inference_mode：关闭梯度追踪，节省显存并加快推理
    with torch.inference_mode():
        # Loop through DataLoader batches
        # 逐 batch 遍历验证/测试集
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            # 把数据和标签搬到 GPU（或 CPU）上
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            # 前向传播得到预测概率
            y_preds = model(X)

            # 2. Calculate and accumulate loss
            # 计算验证损失并累加（用于监控模型在未见数据上的表现）
            loss = loss_fn(y_preds, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            # 阈值 0.5 转成 0/1 标签，与真实标签比较后统计准确率
            test_pred_labels = y_preds > 0.5
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    # 得到整个验证集上的平均损失与平均准确率
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def predict(model: torch.nn.Module,
            image: Path,
            transform: transforms,
            class_dict: Dict,
            plot: bool,
            unknown_threshold: float = 0.85) -> Tuple[Path, str, float, bool]:
    '''
        Makes inference using the trained model
        
        Inputs:
        
            model: A trained PyTorch model for making inference.
            image: Image path
            transform:  Data transforms to be applied to the image
            class_dict: Dictionary mapping the output to class label
            plot: Flag whether to plot the image and prediction result
            unknown_threshold: Minimum confidence required to accept the result as a known class.
                               If the maximum probability is below this value, return 'Unknown'.
        
    '''
    # 用训练好的模型对单张图片做推理预测，并判断是否为未知类别
    # Put model in eval mode
    # 切换到评估模式
    model.eval()   

    # Turn on inference context manager
    # 关闭梯度追踪，推理更省内存
    with torch.inference_mode():
        # OpenCV 默认按 BGR 读图，先转成 RGB 保证颜色正确
        im = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        # 转成 PIL Image 后执行与训练一致的预处理（缩放、转张量、归一化）
        im_transformed = Image.fromarray(im)
        im_transformed = transform(im_transformed)
        # Now the image has dimensions (C, H, W) - 
        # for the model we need to convert this to (B, C, H, W)
        # 模型输入需要 (B, C, H, W)，所以在第 0 维增加 batch 维度
        im_transformed = im_transformed.unsqueeze(0)
        # Make inference
        # 前向传播，得到“是狗”的概率
        y_pred = model(im_transformed)
        # Output label
        # 获取模型预测“是狗”的概率，并计算二分类的最大概率作为置信度。
        # 在二分类中，最大概率 = max(P(dog), P(cat))，最小可能值为 0.5。
        p_dog = float(y_pred.item())
        p_cat = 1.0 - p_dog
        confidence = max(p_dog, p_cat)

        # 第一阶段未知检测：如果模型对猫或狗都没有足够信心，就判为 Unknown。
        # unknown_threshold 可以由调用方调节，用于平衡“误判未知”和“漏判未知”。
        is_unknown = confidence < unknown_threshold

        if is_unknown:
            label = 'Unknown'
        else:
            predicted_class = 1 if p_dog >= 0.5 else 0
            label = class_dict.get(predicted_class)

        print(f'confidence={confidence:.4f} | unknown={is_unknown}')
        
    if plot:
        
        # 按需展示原图与预测标签
        plot_inference(im, label)
        
        
    return image, label, confidence, is_unknown

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          validation_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          save_model_after_epochs: int, 
          model_name: str,         
          device: torch.device,
          BASE_PATH: Path=Path('./')) -> Dict[str, List]:
    '''
        Trains and tests PyTorch model for the number of epochs.
        
        Inputs:
        
            model:                  A PyTorch model to be trained.
            train_dataloader:       A DataLoader instance for the model to be trained on.
            validation_dataloader:  A DataLoader instance for the model to be tested on.
            optimizer:              A PyTorch optimizer to help minimize the loss function.
            loss_fn:                A PyTorch loss function to minimize.
            epochs:                 Number of epochs used for training
            save_mode_after_epochs: Save model after number of specified epochs, default = -1 (save model only at the end of epochs)
            model_name:             Name of model filename
            device:                 A target device to compute on (e.g. "cuda" or "cpu").
            BASE_PATH:              Location containing models directory
        
    '''
  
    # 完整的训练流程：循环 epochs 次，交替执行训练与验证，并保存模型
    # Create empty results dictionary
    # 用字典记录每个 epoch 的训练/验证损失与准确率
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # Loop through training and testing steps for a number of epochs
    # tqdm 用于显示训练进度条
    for epoch in tqdm(range(epochs)):
        # 1) 在训练集上训练一个 epoch
        train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
        # 2) 在验证集上评估一个 epoch
        test_loss, test_acc = test_step(model=model,
            dataloader=validation_dataloader,
            loss_fn=loss_fn,
            device=device)

        
        # Update results dictionary
        # 把本 epoch 的指标追加到 results，训练结束后可保存或绘图
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        if save_model_after_epochs > 0:                
            
            # 每隔 save_model_after_epochs 个 epoch 尝试保存一次模型
            if epoch % save_model_after_epochs == 0:            
                # Print out what's happening
                # 打印当前 epoch 的训练/验证损失与准确率
                print(
                    f"Epoch: {epoch+1} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"train_acc: {train_acc:.4f} | "
                    f"test_loss: {test_loss:.4f} | "
                    f"test_acc: {test_acc:.4f}"
                )
                
                if test_loss <= train_loss:
                    # 验证损失不高于训练损失时才保存（作者自定义的保存策略）
                    save_model(model, model_name, BASE_PATH)
                    print(f'Model successfully saved at epoch {epoch + 1}')
                    
        else:
            # if save_model_after_epoch is not a positive number, only display loss/accuracy metrics after every 10 epochs
            # 未开启中途保存时，每 10 个 epoch 打印一次指标
            if epoch % 10 == 0:            
                # Print out what's happening
                print(
                    f"Epoch: {epoch+1} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"train_acc: {train_acc:.4f} | "
                    f"test_loss: {test_loss:.4f} | "
                    f"test_acc: {test_acc:.4f}"
                )
                
    # 若未开启中途保存，训练结束后保存最终模型
    # Save model at the end of training is save_model_after_epoch is not a positive number            
    if save_model_after_epochs <= 0:
        save_model(model, model_name, BASE_PATH)
        print(f'Model successfully saved at epoch {epoch + 1}')     
        

    # Return the filled results at the end of the epochs
    # 返回记录全部 epoch 指标的字典，供上层保存成 .npz 或绘图
    return results

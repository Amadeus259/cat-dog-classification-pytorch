"""
data_setup.py —— 数据准备模块
=============================
这个文件负责把“猫狗数据集”变成 PyTorch 可以训练的数据：

1. CatDogDataset：自定义 Dataset，从 cats/、dogs/ 子目录读取 .jpg 图片，
   做缩放、随机水平翻转、转张量、归一化等预处理，并生成标签
   （文件名以 dog 开头 -> 1，代表狗；否则 -> 0，代表猫）。
2. unzip_file / unzip_data：把 Kaggle 下载的 zip 数据集解压到指定目录。
3. get_data_loaders：把 CatDogDataset 包装成 DataLoader，供训练/验证循环使用。

本文件也可以单独运行（python src/data_setup.py），用于测试数据读取与预处理流程。
"""
import zipfile
import glob
import cv2
import torch

from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from PIL import Image

from random import shuffle, seed

seed(42)

class CatDogDataset(Dataset):
    """
    Create a custom Cat-Dog Dataset
    """

    # 数据集目录结构要求：
    #     directory/
    #     ├── cats/  (猫图片)
    #     └── dogs/  (狗图片)
    # 标签规则：文件名以 "dog" 开头 -> 1（狗），否则 -> 0（猫）。
    def __init__(self, directory: Path, num_files: int = -1):
        CATS_DIRECTORY = directory / 'cats'
        DOGS_DIRECTORY = directory / 'dogs'
        # 用 rglob('*.jpg') 递归找出 cats/ 与 dogs/ 目录下所有 jpg 图片
        cats = list(CATS_DIRECTORY.rglob('*.jpg'))
        dogs = list(DOGS_DIRECTORY.rglob('*.jpg'))
        print(f'Total number of Cat files in this set: {len(cats)}')
        print(f'Total number of Dog files in this set: {len(dogs)}')
        # num_files = -1 表示使用全部图片；否则每类只取前 num_files 张
        self.fpaths = cats[:num_files] + dogs[:num_files]
        
        # transforms.Compose：把多个图像预处理操作按顺序组合，逐张应用
        self.data_transform = transforms.Compose([
            # Resize the images to 224x224
            # 统一缩放到 224x224，匹配自定义 CNN / VGG16 的输入尺寸
            transforms.Resize(size=(224, 224)),
            # Horizontal flip
            # 随机水平翻转：数据增强，让模型对左右镜像更鲁棒
            transforms.RandomHorizontalFlip(),
            # Turn the image into a torch.Tensor
            # 转成 torch.Tensor，同时把像素值从 [0, 255] 缩放到 [0.0, 1.0]
            transforms.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
            # Normalize
            # 用 ImageNet 的均值/标准差做标准化，加速收敛
            transforms.Normalize(mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        # 随机打乱 (cat 图片 + dog 图片) 的排列顺序
        shuffle(self.fpaths)
        # 通过文件名是否以 "dog" 开头生成标签：True -> 1（狗），False -> 0（猫）
        self.targets =[fpath.as_posix().split('/')[-1].startswith('dog') for fpath in self.fpaths] 
        
    def __len__(self):
        # 返回数据集样本总数，DataLoader 依赖该方法知道一轮有多少样本
        return len(self.fpaths )
    
    def __getitem__(self, ix):
        # 根据索引 ix 取出第 ix 张图片及其标签
        f = self.fpaths[ix]
        target = self.targets[ix]
        # OpenCV 默认按 BGR 读取图片，这里先转成 RGB，避免颜色通道颠倒
        im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        # numpy 数组转成 PIL Image，便于交给 torchvision.transforms 处理
        im = Image.fromarray(im)
        im = self.data_transform(im)
        # 返回 (图像张量, 标签张量)：标签是形状 (1,) 的 float 张量，方便与 BCELoss 计算
        return im, torch.tensor([target]).float()

def unzip_file(zip_path, extract_to):
    """
    Unzips a zip file to the specified directory.
    
    Inputs:

    zip_path: Path to the zip file
    extract_to: Directory to extract files to
    """
    # 打开 zip 压缩包并把全部文件解压到 extract_to 目录
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted all files to {extract_to}")
        
def unzip_data(zip_path, extract_to):
    # 对 unzip_file 的简单封装：出现异常时不做处理，继续向上抛出
    try:
        unzip_file(zip_path, extract_to)
    except Exception as e:
        raise e
    
def get_data_loaders(directory: Path, num_files: int = -1, batch_size: int = 32):
    '''
    Get custom dataset and data loaders
    
    Inputs:
    
    directory:  Training, validation or test directory
    num_files:  Number of files to be read from the data directory
    batch_size: Batch size for the data loader
    
    '''
    data = CatDogDataset(directory, num_files)
    # shuffle=True 让每个 epoch 的训练顺序都不同；drop_last=True 丢弃不足一个 batch 的尾部
    return DataLoader(data, batch_size, shuffle=True, drop_last = True)

if __name__ == '__main__':
    
    # 两个功能开关：UNZIP_DATA=1 表示先解压 zip；CUSTOM_DATASET=1 表示测试自定义数据集
    UNZIP_DATA = 0
    CUSTOM_DATASET = 1    
    BASE_PATH = Path('./')
    # Kaggle 原始数据压缩包名，以及解压后的数据目录
    ZIP_FILENAME = 'cat-and-dog.zip'
    DATA_DIRECTORY = BASE_PATH / 'data'
    # 注意：压缩包内部还有一层同名子目录，所以路径要写两层
    TRAINING_DATA_DIRECTORY = DATA_DIRECTORY / 'training_set/training_set'
    TEST_DATA_DIRECTORY = DATA_DIRECTORY / 'test_set/test_set'
    BATCH_SIZE = 32
    # extract from zip to data directory
    if UNZIP_DATA:        
        # 打开压缩包并解压到 data/ 目录
        zip_path = BASE_PATH / ZIP_FILENAME
        extract_to = DATA_DIRECTORY
        unzip_file(zip_path, extract_to)
    # create custom dataset
    if CUSTOM_DATASET:
        # 生成训练集（使用全部图片）与验证集（测试集每类取 500 张）的 DataLoader
        # Data loaders
        train_dl = get_data_loaders(TRAINING_DATA_DIRECTORY, -1, BATCH_SIZE)
        validation_dl = get_data_loaders(TEST_DATA_DIRECTORY, 500, BATCH_SIZE)
        
        
        
        
        
        
        
        # 下面是一段调试演示代码：读一张图片并打印预处理前后的信息
        # dataset = CatDogDataset(DATA_DIRECTORY, 500)         
        cat_data = DATA_DIRECTORY / 'training_set/training_set/dogs'
        # 注意：变量名叫 cat_data，实际指向的是 dogs 目录（原代码遗留，仅作读取测试）
        cat_files = list(cat_data.rglob('*.jpg'))
        im = cv2.cvtColor(cv2.imread(cat_files[0]), cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)      
        # im = torch.from_numpy(im).permute(2, 0, 1).float()
        # im = torch.permute(im, (2, 0, 1))
        # print(len(list(cat_data.rglob('*.jpg'))))#.as_posix().split('/')[-1].startswith('cat'))
        print(im.size)
    
        data_transform = transforms.Compose([
            # Resize the images to 224x224
            transforms.Resize(size=(224, 224)),
            # Turn the image into a torch.Tensor
            transforms.ToTensor(), # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
            # Normalize
            transforms.Normalize(mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            
        ])
        
        im_transformed = data_transform(im)
        # 打印预处理后张量的类型、形状和数据类型，检查流水线是否正确
        print(type(im_transformed))
        print(im_transformed.shape)
        print(im_transformed.dtype)
    

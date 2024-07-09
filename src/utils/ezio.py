import torch
import numpy as np
import pandas as pd


# 构建数据集，划分训练集和测试集
def build_dataset(data, train_ratio=0.8):
    '''
        构建数据集，划分训练集和测试集
    Args:
        data: torch.Tensor, 待划分的数据集
        train_ratio: float, 训练集占比
    Returns:
        train_data: torch.Tensor, 训练集
        test_data: torch.Tensor, 测试集
    '''
    DATA = data[torch.randperm(len(data))]  #shuffle
    DATA = DATA.to('cuda:0')
    train_data = DATA[:int(train_ratio*len(DATA))]
    test_data = DATA[int(train_ratio*len(DATA)):]

    return train_data, test_data

# 保存Tensor为CSV文件
def save_tensor_to_csv(tensor, file_name):
    """
        将给定的Tensor保存为CSV文件。
    
    Args:
        tensor: 要保存的Tensor。
        file_name: 保存的CSV文件名。
    Returns:
        None
    """
    # 确保Tensor在CPU上
    tensor = tensor.cpu()
    numpy_array = tensor.numpy()
    df = pd.DataFrame(numpy_array)
    df.to_csv(file_name, index=False)

    return None
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# MNIST 数据集
class MNIST_DATA(Dataset): # 继承 Dataset 类

    def __init__(self, filepath):
        df = pd.read_csv(filepath) # 导入数据
        arr = df.values # 对象退化为数组
        # imgs = np.array([data[i][1:] for i in range(len(data))])
        # labels = np.zeros((len(data), 10))
        # labels[np.arange(len(data)), [data[i][0] for i in range(len(data))]] = 1
        # arr = np.concatenate((imgs, labels), axis=1)
        arr = arr.astype(np.float32) # 转为 float32 类型数组        
        ts = torch.tensor(arr) # 数组转为张量
        ts = ts.to('cuda:0') # 把训练集搬到 cuda 上
        self.X = ts[ : ,-784 : ]
        self.Y = ts[ : , 0].reshape((-1,1)) 
        self.len = ts.shape[0] # 样本的总数

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len
'''
最基础的全连接神经网络尝试
'''
import torch
import numpy as np
import pandas as pd

from modules import nets
from utils import nuts, ezio

# 外部参数
data_path = r'database\MNIST\mnist_test.csv'
train_ratio = 0.8
epochs = 500

# 导入数据
data = pd.read_csv(data_path, header=None).values
imgs = np.array([data[i][1:] for i in range(len(data))])
labels = np.zeros((len(data), 10))
labels[np.arange(len(data)), [data[i][0] for i in range(len(data))]] = 1
imgs = torch.tensor(imgs, dtype=torch.int)
labels = torch.tensor(labels, dtype=torch.float32)

# 构建数据集，划分训练集和测试集
DATA = torch.cat((imgs, labels), dim=1)
train_data, test_data = ezio.build_dataset(DATA, train_ratio)

# 定义模型
model = nets.SFDNN().to('cuda:0')

# 训练模型
X = train_data[:, :784] # 前 784 列为输入特征
Y = train_data[:, -10:] # 后 10 列为输出特征
losses = nuts.train_process(model, X, Y, epochs=epochs)
nuts.plot_loss(losses) # 绘制损失函数

# 测试网络
# 给测试集划分输入与输出
X = test_data[:, :784] # 前 784 列为输入特征
Y = test_data[:, -10:] # 后 10 列为输出特征
Pred = nuts.test_process(model, X, Y)

# ezio.save_tensor_to_csv(Y, 'database/MNIST/test-data.csv')
# ezio.save_tensor_to_csv(Pred, 'database/MNIST/Pred.csv')
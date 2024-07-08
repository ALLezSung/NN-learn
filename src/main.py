import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nets.simple as simple
import utils.nuts as nuts

# 外部参数
data_path = r'database\MNIST\mnist_test.csv'
train_ratio = 0.8
epochs = 1000

# 导入数据
data = pd.read_csv(data_path, header=None).values
imgs = np.array([data[i][1:] for i in range(len(data))])
labels = np.zeros((len(data), 10))
labels[np.arange(len(data)), [data[i][0] for i in range(len(data))]] = 1
imgs = torch.tensor(imgs, dtype=torch.int)
labels = torch.tensor(labels, dtype=torch.float32)

# 构建数据集，划分训练集和测试集
DATA = torch.cat((imgs, labels), dim=1)
train_data, test_data = nuts.build_dataset(DATA, train_ratio)

# 定义模型
model = simple.DNN().to('cuda:0')

# 训练模型
X = train_data[:, :784] # 前 784 列为输入特征
Y = train_data[:, -10:] # 后 10 列为输出特征
losses = nuts.train(model, X, Y, epochs=epochs)
# 绘图
Fig = plt.figure()
plt.plot(range(epochs), losses)
plt.ylabel('loss'), plt.xlabel('epoch')

# 测试网络
# 给测试集划分输入与输出
X = test_data[:, :784] # 前 784 列为输入特征
Y = test_data[:, -10:] # 后 10 列为输出特征
nuts.test(model, X, Y)

plt.show()
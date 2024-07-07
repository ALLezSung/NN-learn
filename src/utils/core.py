import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..nets import DNN_simple

#外部参数
data_path = r'database\MNIST\mnist_test.csv'
train_ratio = 0.8

#导入数据
data = pd.read_csv(data_path, header=None).values
imgs = np.array([data[i][1:] for i in range(len(data))])
labels = np.zeros((len(data), 10))
labels[np.arange(len(data)), [data[i][0] for i in range(len(data))]] = 1
imgs = torch.tensor(imgs, dtype=torch.int)
labels = torch.tensor(labels, dtype=torch.float32)

#构建数据集，划分训练集和测试集
DATA = torch.cat((imgs, labels), dim=1)
DATA = DATA[torch.randperm(len(DATA))]  #shuffle
DATA = DATA.to('cuda:0')
train_data = DATA[:int(train_ratio*len(DATA))]
test_data = DATA[int(train_ratio*len(DATA)):]

#定义模型
model = DNN_simple.DNN().to('cuda:0')

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练网络
epochs = 1000
losses = [] # 记录损失函数变化的列表
# 给训练集划分输入与输出
X = train_data[:, :784] # 前 784 列为输入特征
Y = train_data[:, -10:] # 后 10 列为输出特征
for epoch in range(epochs):
	Pred = model(X) # 一次前向传播（批量）
	loss = loss_fn(Pred, Y) # 计算损失函数
	losses.append(loss.item()) # 记录损失函数的变化
	optimizer.zero_grad() # 清理上一轮滞留的梯度
	loss.backward() # 一次反向传播
	optimizer.step() # 优化内部参数

Fig = plt.figure()
plt.plot(range(epochs), losses)
plt.ylabel('loss'), plt.xlabel('epoch')

# 测试网络
# 给测试集划分输入与输出
X = test_data[:, :784] # 前 784 列为输入特征
Y = test_data[:, -10:] # 后 10 列为输出特征
with torch.no_grad(): # 该局部关闭梯度计算功能
	Pred = model(X) # 一次前向传播（批量）
	Pred[:,torch.argmax(Pred, axis=1)] = 1
	Pred[Pred!=1] = 0
	correct = torch.sum( (Pred == Y).all(1) ) # 预测正确的样本
	total = Y.size(0) # 全部的样本数量
	print(f'测试集精准度: {100*correct/total} %')
	
plt.show()
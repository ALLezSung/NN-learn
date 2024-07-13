import torch
import torch.nn as nn

# 全连接神经网络
class SFDNN(nn.Module):

    def __init__(self):
        ''' 
        简单的全连接神经网络
        Simple-Fully-connecteD-Neural-Network
        784 -> 256 -> 128 -> 32 -> 10
        '''
        super(SFDNN,self).__init__()
        self.net = nn.Sequential(       # 按顺序搭建各层
            nn.Linear(784, 784), nn.ReLU(), # 第 1 层：全连接层
            nn.Linear(784, 512), nn.ReLU(), # 第 1 层：全连接层
            nn.Linear(512, 256), nn.ReLU(), # 第 2 层：全连接层
            nn.Linear(256, 64), nn.ReLU(),  # 第 3 层：全连接层
            nn.Linear(64, 10)  # 第 4 层：全连接层
        )
    
    def forward(self, x):
        ''' 前向传播 '''
        y = self.net(x) # x 即输入数据
        return y        # y 即输出数据

class MNIST_DNN(nn.Module):

    def __init__(self):
        '''
        针对 MNIST 数据集的神经网络
        784 -> 512 -> 256 -> 128 -> 64 -> 10
        '''
        super(MNIST_DNN,self).__init__()
        self.net = nn.Sequential( # 按顺序搭建各层
        # nn.Flatten(), # 把图像铺平成一维
        nn.Linear(784, 512), nn.ReLU(), # 第 1 层：全连接层
        nn.Linear(512, 256), nn.ReLU(), # 第 2 层：全连接层
        nn.Linear(256, 128), nn.ReLU(), # 第 3 层：全连接层
        nn.Linear(128, 64), nn.ReLU(), # 第 4 层：全连接层
        nn.Linear(64, 10) # 第 5 层：全连接层
        )
        self.to('cuda:0')
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    # 前向传播
    def forward(self, x):
        y = self.net(x) # x 即输入数据
        return y # y 即输出数据   

    # 训练方法
    def train_process(self, train_loader, epochs=50):
        losses = []
        for epoch in range(epochs):
            for (x, y) in train_loader: # 获取小批次的 x 与 y
                Pred = self.forward(x) # 一次前向传播（小批量）
                y = y.squeeze().long()  # 移除 y 的所有单维度，并确保数据类型为 long
                loss = self.loss_fn(Pred, y) # 计算损失函数
                losses.append(loss.item()) # 记录损失函数的变化
                self.optimizer.zero_grad() # 清理上一轮滞留的梯度
                loss.backward() # 一次反向传播
                self.optimizer.step() # 优化内部参数

        return losses

    # 测试方法
    def test_process(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad(): # 该局部关闭梯度计算功能
            for (x, y) in test_loader: # 获取小批次的 x 与 y
                Pred = self.forward(x) # 一次前向传播（小批量）
                _, predicted = torch.max(Pred.data, dim=1)
                y = y.squeeze()
                correct += torch.sum( (predicted == y) )
                total += y.size(0) 
        result = f'测试集精准度: {100*correct/total} %'
        
        return result
    
    def save_process(self, path):
        torch.save(self.state_dict(), path)


class MNIST_zh_CNN(nn.Module):

    def __init__(self):
        '''
        针对 MNIST_zh 数据集的类LeNet-5卷积神经网络

        IN层：  1  x64 x64
        C1层：  8  x64 x64
        S2层:   8  x30 x30
        C3层：  24 x24 x24
        S4层：  24 x10 x10
        C5层：  64 x6  x6
        S6层：  64 x4  x4
        C7层：  256x1  x1
        F8层：  128
        Out层： 15
        '''
        super(MNIST_zh_CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding=3), nn.ReLU(), # 第 1 层：卷积层 1x64x64 -> 8x64x64
            nn.AvgPool2d(kernel_size=5, stride=2), # 第 2 层：池化层 8x64x64 -> 8x30x30
            nn.Conv2d(8, 24, kernel_size=7), nn.ReLU(), # 第 3 层：卷积层 8x30x30 -> 24x24x24
            nn.MaxPool2d(kernel_size=5, stride=2), # 第 4 层：池化层 24x24x24 -> 24x10x10
            nn.Conv2d(24, 64, kernel_size=5), nn.ReLU(), # 第 5 层：卷积层 24x10x10 -> 64x6x6
            nn.AvgPool2d(kernel_size=3, stride=1), # 第 6 层：池化层 64x6x6 -> 64x4x4
            nn.Conv2d(64, 256, kernel_size=4), nn.ReLU(), # 第 7 层：卷积层 64x4x4 -> 256x1x1
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), # 第 8 层：全连接层
            nn.Linear(128, 15) # 第 9 层：全连接层
        )
        
        self.to('cuda:0')
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    # 前向传播
    def forward(self, x):
        y = self.net(x)
        return y    
    
    # 训练方法
    def train_process(self, train_loader, epochs=50):
        losses = []
        for epoch in range(epochs):
            for (x, y) in train_loader:
                Pred = self.forward(x)
                loss = self.loss_fn(Pred, y)
                losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return losses
    
    # 测试方法
    def test_procss(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for (x, y) in test_loader:
                Pred = self.forward(x)
                _, predicted = torch.max(Pred.data, dim=1)
                correct += torch.sum( (predicted == y) )
                total += y.size(0)
        result = f'测试集精准度: {100*correct/total} %'
        return result
    
    def save_process(self, path):
        torch.save(self.state_dict(), path)


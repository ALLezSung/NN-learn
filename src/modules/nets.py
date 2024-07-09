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

    # 训练方法
    def train(self, train_loader, epochs=50):
        losses = []
        for epoch in range(epochs):
            for (x, y) in train_loader: # 获取小批次的 x 与 y
                x, y = x.to('cuda:0'), y.to('cuda:0')
                Pred = self.forward(x) # 一次前向传播（小批量）
                y = y.squeeze().long()  # 移除 y 的所有单维度，并确保数据类型为 long
                loss = self.loss_fn(Pred, y) # 计算损失函数
                losses.append(loss.item()) # 记录损失函数的变化
                self.optimizer.zero_grad() # 清理上一轮滞留的梯度
                loss.backward() # 一次反向传播
                self.optimizer.step() # 优化内部参数

        return losses

    # 测试方法
    def test(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad(): # 该局部关闭梯度计算功能
            for (x, y) in test_loader: # 获取小批次的 x 与 y
                x, y = x.to('cuda:0'), y.to('cuda:0')
                Pred = self.forward(x) # 一次前向传播（小批量）
                _, predicted = torch.max(Pred.data, dim=1)
                y = y.squeeze()
                correct += torch.sum( (predicted == y) )
                total += y.size(0) 
        result = f'测试集精准度: {100*correct/total} %'
        
        return result
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def forward(self, x):
        ''' 前向传播 '''
        y = self.net(x) # x 即输入数据
        return y # y 即输出数据



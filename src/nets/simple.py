import torch
import torch.nn as nn

# 全连接神经网络
class DNN(nn.Module):

    def __init__(self):
        ''' 
        简单的全连接神经网络
        784 -> 256 -> 128 -> 32 -> 10
        '''
        super(DNN,self).__init__()
        self.net = nn.Sequential(       # 按顺序搭建各层
            nn.Linear(784, 256), nn.ReLU(), # 第 1 层：全连接层
            nn.Linear(256, 128), nn.ReLU(), # 第 2 层：全连接层
            nn.Linear(128, 32), nn.ReLU(), # 第 3 层：全连接层
            nn.Linear(32, 10)               # 第 4 层：全连接层
        )
    
    def forward(self, x):
        ''' 前向传播 '''
        y = self.net(x) # x 即输入数据
        return y        # y 即输出数据


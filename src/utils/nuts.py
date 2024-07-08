import torch
import torch.nn as nn
import torch.optim as optim


# 显示模型结构以及参数信息
def show_model(model):
    ''' 
        显示模型结构以及参数信息
    Args:
        model: nn.Module, 待显示的模型
    Returns:
        None
    '''
    print(model)
    print('Model parameters:')
    for name, param in model.named_parameters():
        print(f'Layer: {name} | Size: {param.size()} | Values : {param[:2]}')
    print('\n')

    return None

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

# 训练模型
def train(model, X, Y, epochs=1000, optimizer=None, 
          loss_fn=nn.MSELoss(), losses=None, lr=0.01,
          show_loss_every_n_epochs=0):
    '''
        训练模型
    Args:
        model: nn.Module, 待训练的模型
        X: torch.Tensor, 输入数据
        Y: torch.Tensor, 输出数据
        epochs: int, 训练轮数
        optimizer: torch.optim.Optimizer, 优化器
        loss_fn: nn.Module, 损失函数
        losses: list, 记录损失函数的变化
        lr: float, 学习率
        show_loss_every_n_epochs: int, 每隔 n 轮显示一次损失(为 0 则不显示)
    Returns:
        losses: list, 记录损失函数的变化
    '''

    if losses is None:
        losses = []
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr)
    if show_loss_every_n_epochs < 0:
        print("Warning: show_loss_every_n_epochs cannot be negative. Setting it to 0.")
        show_loss_every_n_epochs = 0
    for epoch in range(epochs):
        Pred = model(X)  # 一次前向传播（批量）
        loss = loss_fn(Pred, Y)  # 计算损失函数
        losses.append(loss.item())  # 记录损失函数的变化
        optimizer.zero_grad()  # 清理上一轮滞留的梯度
        loss.backward()  # 一次反向传播
        optimizer.step()  # 优化内部参数

        # 每隔 n 轮显示一次损失
        if show_loss_every_n_epochs != 0 and epoch % show_loss_every_n_epochs == 0: 
            print(f'Epoch: {epoch} | Loss: {loss.item()[epoch]}')

    return losses

# 测试模型
def test(model, X, Y):
    '''
        测试模型
    Args:
        model: nn.Module, 待测试的模型
        X: torch.Tensor, 输入数据
        Y: torch.Tensor, 输出数据
    Returns:
        None
    '''
    with torch.no_grad(): # 该局部关闭梯度计算功能
        Pred = model(X) # 一次前向传播（批量）
        Pred[:,torch.argmax(Pred, axis=1)] = 1
        Pred[Pred!=1] = 0
        correct = torch.sum( (Pred == Y).all(1) ) # 预测正确的样本
        total = Y.size(0) # 全部的样本数量
        print(f'测试集精准度: {100*correct/total} %')

    return None


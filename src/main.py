from torch.utils.data import DataLoader
from torch.utils.data import random_split

from modules import nets, DataClass
from utils import nuts


# 外部参数
data_path = r'database\MNIST\mnist_test.csv'
train_ratio = 0.8
epochs = 50
batch_size = 128

# 构建数据集，划分训练集和测试集
DATA = DataClass.MNIST_DATA(data_path)
train_size = int(len(DATA) * train_ratio)
test_size = len(DATA) - train_size
train_data, test_data = random_split(DATA, [train_size, test_size])
# 批次加载器
train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(dataset=test_data, shuffle=False, batch_size=test_size)

# 定义模型
model = nets.MNIST_DNN()

# 训练 及 测试
losses = model.train(train_loader, epochs=epochs)
_ = model.test(test_loader)
print(_)
nuts.plot_loss(losses)

# 保存模型
model.save(r'models\MNIST_DNN.pth')
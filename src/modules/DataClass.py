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
    
# MNIST_zh 数据集
class MNIST_zh_DATA(Dataset):

    def __init__(self, img_folder_path) -> None:
        super().__init__()
        self.path = img_folder_path
        self.X = []
        self.Y = []
        self.len:int = 0
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    def load_data(self):
        import pathlib as pl
        import torchvision.transforms as transforms
        import PIL.Image as Image
        transform = transforms.Compose([
            transforms.Resize((64, 64)),  # 所有图像调整为28x28大小
            transforms.ToTensor(),  # 将图像转换为Tensor
        ])
        for img_path in pl.Path(self.path).iterdir():
            try:
                image = Image.open(img_path)
                image = transform(image)  # 应用转换
                label = int(img_path.stem.split('_')[-1])
                self.X.append(image)
                self.Y.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        self.X = torch.stack(self.X).to(self.device) 
        self.Y = torch.tensor(self.Y, dtype=torch.long).to(self.device)
        self.len = len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len
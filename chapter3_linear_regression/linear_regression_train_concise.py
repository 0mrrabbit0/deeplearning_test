import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn
import pandas as pd

# 从CSV文件读取训练数据
data_df = pd.read_csv('linear_regression_training_data.csv')
features = torch.tensor(data_df[['feature_1', 'feature_2']].values, dtype=torch.float32)
labels = torch.tensor(data_df['label'].values, dtype=torch.float32).reshape(-1, 1)
print(f'已从CSV加载 {len(features)} 条训练样本')
print(f'特征维度: {features.shape}, 标签维度: {labels.shape}\n')

# 真实参数（用于评估）
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2

# 构造数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
w = net[0].weight.data
b = net[0].bias.data
print(f'学习到的权重: w={w.reshape(-1).detach().numpy()}, b={b.item():.4f}')
print('w的估计误差：', true_w - w.reshape(true_w.shape))
print('b的估计误差：', true_b - b)
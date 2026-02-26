import random
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用非交互式后端
from d2l import torch as d2l
import pandas as pd

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 定义模型
def linreg(X, w, b): #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y): #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size): #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
batch_size = 10

# 从CSV文件读取训练数据
data_df = pd.read_csv('linear_regression_training_data.csv')
features = torch.tensor(data_df[['feature_1', 'feature_2']].values, dtype=torch.float32)
labels = torch.tensor(data_df['label'].values, dtype=torch.float32).reshape(-1, 1)
print(f'已从CSV加载 {len(features)} 条训练样本')
print(f'特征维度: {features.shape}, 标签维度: {labels.shape}\n')

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size) # 使用参数的梯度更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print('\n训练完成！')
print(f'真实权重: w=[2, -3.4], b=4.2')
print(f'学习到的权重: w={w.reshape(-1).detach().numpy()}, b={b.item():.4f}')
print(f'误差: w_error={torch.abs(w.reshape(-1) - torch.tensor([2., -3.4])).detach().numpy()}, b_error={abs(b.item() - 4.2):.4f}')
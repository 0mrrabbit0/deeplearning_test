import random
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 使用非交互式后端
from d2l import torch as d2l
import pandas as pd

def synthetic_data(w, b, num_examples): #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0],'\nlabel:', labels[0])

# 保存数据为CSV文件
data_df = pd.DataFrame({
    'feature_1': features[:, 0].detach().numpy(),
    'feature_2': features[:, 1].detach().numpy(),
    'label': labels[:, 0].detach().numpy()
})
data_df.to_csv('linear_regression_training_data.csv', index=False)
print(f'\n数据已保存到 linear_regression_training_data.csv，共 {len(data_df)} 条样本')

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), s=1, alpha=0.6)
plt.xlabel('Feature 2')
plt.ylabel('Label')
plt.title('Linear Regression Training Data Visualization')
plt.grid(True, alpha=0.3)
plt.savefig('linear_regression_visualization.png', dpi=300, bbox_inches='tight')
print('可视化图表已保存到 linear_regression_visualization.png')


# Chapter 3: 线性回归

本章实现线性回归模型，包含从零实现和PyTorch简洁实现两个版本。

## 文件说明

- **linear_regression_data_generation.py** - 数据生成脚本
  - 生成符合线性关系的训练数据：y = 2×x₁ - 3.4×x₂ + 4.2 + 噪声
  - 保存为CSV文件供训练使用
  - 生成数据可视化图表

- **linear_regression_train.py** - 从零实现版本
  - 手动实现数据迭代器（带随机打乱）
  - 手动实现线性回归模型
  - 手动实现均方损失函数
  - 手动实现SGD优化算法
  - 完整展示梯度下降过程

- **linear_regression_train_concise.py** - PyTorch简洁实现
  - 使用 `nn.Linear` 定义模型
  - 使用 `nn.MSELoss` 损失函数
  - 使用 `torch.optim.SGD` 优化器
  - 代码更简洁高效

## 运行方法

### 1. 生成训练数据（⚠️ 必须先执行）

```bash
python linear_regression_data_generation.py
```

这将生成：
- `linear_regression_training_data.csv` - 1000条训练样本（3列：feature_1, feature_2, label）
- `linear_regression_visualization.png` - Feature 2 与 Label 的散点图

### 2. 运行从零实现版本

```bash
python linear_regression_train.py
```

**输出示例：**
```
已从CSV加载 1000 条训练样本
特征维度: torch.Size([1000, 2]), 标签维度: torch.Size([1000, 1])

epoch 1, loss 0.000053
epoch 2, loss 0.000053
epoch 3, loss 0.000053

训练完成！
真实权重: w=[2, -3.4], b=4.2
学习到的权重: w=[ 1.9994899 -3.3996058], b=4.2000
误差: w_error=[0.0005101  0.00039434], b_error=0.0000
```

### 3. 运行PyTorch简洁版本

```bash
python linear_regression_train_concise.py
```

**输出示例：**
```
已从CSV加载 1000 条训练样本
特征维度: torch.Size([1000, 2]), 标签维度: torch.Size([1000, 1])

epoch 1, loss 0.000310
epoch 2, loss 0.000105
epoch 3, loss 0.000105
学习到的权重: w=[ 1.9994 -3.3994], b=4.2008
w的估计误差： tensor([-0.0006,  0.0006])
b的估计误差： tensor([-0.0008])
```

## 核心概念

### 1. 线性回归模型

```
y = w₁×x₁ + w₂×x₂ + b
```

- **w**: 权重参数（本例中真实值为 [2, -3.4]）
- **b**: 偏置参数（本例中真实值为 4.2）
- **目标**: 通过训练数据学习出接近真实值的参数

### 2. 损失函数（均方误差）

```
L = (ŷ - y)² / 2
```

衡量模型预测值与真实值的差距。

### 3. 小批量随机梯度下降（Mini-batch SGD）

- **batch_size**: 10（每次用10个样本计算梯度）
- **学习率 (lr)**: 0.03
- **epochs**: 3（遍历整个数据集3次）

**更新公式：**
```
w = w - lr × ∂L/∂w
b = b - lr × ∂L/∂b
```

### 4. 反向传播

PyTorch通过 `loss.backward()` 自动计算梯度：
- 使用链式法则计算 ∂L/∂w 和 ∂L/∂b
- 梯度存储在 `w.grad` 和 `b.grad` 中
- 优化器使用梯度更新参数

## 实现对比

| 特性 | 从零实现 | PyTorch简洁实现 |
|------|---------|----------------|
| 模型定义 | 手动实现 `linreg(X, w, b)` | `nn.Linear(2, 1)` |
| 损失函数 | 手动实现 `squared_loss()` | `nn.MSELoss()` |
| 优化器 | 手动实现 `sgd()` | `torch.optim.SGD()` |
| 数据加载 | 手动实现 `data_iter()` | `DataLoader` |
| 代码量 | 较多，教学性强 | 简洁，工程实用 |

## 学习要点

1. **梯度下降的完整流程**：前向传播 → 计算损失 → 反向传播 → 参数更新
2. **随机性的重要性**：为什么需要 `random.shuffle()`？
3. **批量大小的影响**：batch_size 对训练的影响
4. **学习率的选择**：lr 太大会震荡，太小收敛慢
5. **epoch 的含义**：一个epoch = 完整遍历数据集一次

## 常见问题

**Q: 为什么要除以 batch_size？**
A: 因为损失是对batch求和的，除以batch_size相当于求平均梯度，使不同batch_size效果一致。

**Q: 为什么需要 `param.grad.zero_()`？**
A: PyTorch的 `.backward()` 会累加梯度，不清零会导致梯度越积越大。

**Q: 什么时候用 `with torch.no_grad()`？**
A: 评估模型或更新参数时使用，节省内存，避免构建计算图。

## 相关链接

- [返回项目主页](../)
- [《动手学深度学习》- 线性回归章节](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html)

# 深度学习实践项目

这是一个深度学习学习项目，包含线性回归的实现和训练。

## 项目结构

```
deeplearning_test/
├── chapter3_linear_regression/
│   ├── linear_regression_data_generation.py  # 数据生成脚本
│   ├── linear_regression_train.py            # 手动实现的线性回归训练
│   ├── linear_regression_train_concise.py    # 使用PyTorch简洁实现
│   ├── linear_regression_training_data.csv   # 训练数据集
│   └── linear_regression_visualization.png   # 数据可视化
└── README.md
```

## 功能说明

### 1. 数据生成 (linear_regression_data_generation.py)
- 生成符合线性关系的训练数据：y = 2×x₁ - 3.4×x₂ + 4.2 + 噪声
- 保存为CSV文件供训练使用
- 生成数据可视化图表

### 2. 手动实现训练 (linear_regression_train.py)
- 手动实现数据迭代器（带随机打乱）
- 手动实现线性回归模型
- 手动实现均方损失函数
- 手动实现SGD优化算法
- 完整展示梯度下降过程

### 3. PyTorch简洁实现 (linear_regression_train_concise.py)
- 使用 `nn.Linear` 定义模型
- 使用 `nn.MSELoss` 损失函数
- 使用 `torch.optim.SGD` 优化器
- 代码更简洁高效

## 运行方法

1. 生成训练数据：
```bash
python chapter3_linear_regression/linear_regression_data_generation.py
```

2. 运行手动实现版本：
```bash
python chapter3_linear_regression/linear_regression_train.py
```

3. 运行PyTorch简洁版本：
```bash
python chapter3_linear_regression/linear_regression_train_concise.py
```

## 依赖环境

- Python 3.x
- PyTorch
- pandas
- matplotlib
- d2l (Dive into Deep Learning 库)

## 学习笔记

这个项目帮助理解：
- 线性回归的基本原理
- 梯度下降优化算法
- 小批量随机梯度下降 (Mini-batch SGD)
- PyTorch 的基础使用
- 反向传播机制

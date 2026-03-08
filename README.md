# 深度学习学习项目

本项目是深度学习的学习实践代码库，基于《动手学深度学习》(Dive into Deep Learning) 教材。

## 项目结构

```
deeplearning_test/
├── chapter3_linear_regression/      # 第3章：线性回归
│   ├── README.md                    # 本章详细说明
│   ├── linear_regression_data_generation.py
│   ├── linear_regression_train.py
│   └── linear_regression_train_concise.py
├── chapter3_softmax_regression/     # 第3章：Softmax回归
│   ├── README.md                    # 本章详细说明
│   ├── softmax_regression.py        # Softmax回归完整实现
│   └── data/                        # Fashion-MNIST数据集（.gitignore）
├── .gitignore
└── README.md
```

## 章节列表

- **[Chapter 3: 线性回归](chapter3_linear_regression/)** - 线性回归模型、梯度下降、自动微分
- **[Chapter 3: Softmax回归](chapter3_softmax_regression/)** - 多分类、Softmax函数、交叉熵损失、Fashion-MNIST
- Chapter 4: 多层感知机 _(待添加)_
- 更多章节持续更新...

## 环境依赖

- Python 3.x
- PyTorch
- pandas
- matplotlib
- d2l (Dive into Deep Learning 库)

### 安装依赖

```bash
pip install torch pandas matplotlib d2l
```

## 使用说明

每个章节目录下都有详细的 README.md，请进入对应章节查看具体使用方法。

**⚠️ 注意：** 生成的数据文件（`.csv`）和可视化图片（`.png`）不包含在版本控制中，请先运行数据生成脚本。

## 学习资源

- [《动手学深度学习》官方网站](https://d2l.ai/)
- [《动手学深度学习》中文版](https://zh.d2l.ai/)

## License

本项目仅用于个人学习目的。

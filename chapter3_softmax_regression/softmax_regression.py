"""
Softmax回归从零实现
基于《动手学深度学习》第3章
包含完整的数据加载功能
"""

# ============================================================================
# 1. 导入库
# ============================================================================
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from d2l import torch as d2l


# ============================================================================
# 2. 超参数配置
# ============================================================================
batch_size = 256
num_inputs = 784
num_outputs = 10
lr = 0.1
num_epochs = 10


# ============================================================================
# 3. 数据加载函数
# ============================================================================
def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4


def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5, filename='predictions.png'):
    """绘制图像列表并保存"""
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy(), cmap='gray')
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'预测结果已保存到: {filename}')
    return axes


# 加载数据集
train_iter, test_iter = load_data_fashion_mnist(batch_size)


# ============================================================================
# 4. 辅助工具类
# ============================================================================
class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    """在动画中绘制数据（命令行版本，保存到文件）"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self._config_axes()
        # 保存图片而不是交互式显示
        self.fig.savefig('training_progress.png', dpi=100, bbox_inches='tight')

    def _config_axes(self):
        """配置坐标轴"""
        if self.xlabel:
            self.axes[0].set_xlabel(self.xlabel)
        if self.ylabel:
            self.axes[0].set_ylabel(self.ylabel)
        if self.xlim:
            self.axes[0].set_xlim(self.xlim)
        if self.ylim:
            self.axes[0].set_ylim(self.ylim)
        if self.xscale:
            self.axes[0].set_xscale(self.xscale)
        if self.yscale:
            self.axes[0].set_yscale(self.yscale)
        if self.legend:
            self.axes[0].legend(self.legend)
        self.axes[0].grid()

    def save(self, filename='training_progress.png'):
        """保存最终图表"""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(self.fig)
        print(f'训练曲线已保存到: {filename}')


# ============================================================================
# 5. 模型定义
# ============================================================================
def softmax(X):
    """Softmax函数"""
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


def net(X):
    """Softmax回归模型"""
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    """交叉熵损失函数"""
    return - torch.log(y_hat[range(len(y_hat)), y])


# ============================================================================
# 6. 评估函数
# ============================================================================
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# ============================================================================
# 7. 训练函数
# ============================================================================
def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        print(f'epoch {epoch + 1}, loss {train_metrics[0]:.3f}, train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}')
    train_loss, train_acc = train_metrics

    # 保存训练曲线
    animator.save('training_progress.png')

    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


# ============================================================================
# 8. 预测函数
# ============================================================================
def predict_ch3(net, test_iter, n=6):
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


# ============================================================================
# 9. 主程序
# ============================================================================
if __name__ == '__main__':
    # 初始化模型参数
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    # 定义优化器
    def updater(batch_size):
        return d2l.sgd([W, b], lr, batch_size)

    # 打印数据集信息
    print("="*60)
    print("Fashion-MNIST Softmax回归训练")
    print("="*60)
    print(f"训练集样本数: {len(train_iter.dataset)}")
    print(f"测试集样本数: {len(test_iter.dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"训练批次数: {len(train_iter)}")
    print(f"测试批次数: {len(test_iter)}")
    print(f"学习率: {lr}")
    print(f"训练轮数: {num_epochs}")
    print("="*60)
    print()

    # 训练模型
    print("开始训练...")
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

    print()
    print("="*60)
    print("训练完成！")
    print("="*60)

    # 预测展示
    print("\n生成预测结果...")
    predict_ch3(net, test_iter)

    print("\n所有输出文件：")
    print("  - training_progress.png (训练曲线)")
    print("  - predictions.png (预测结果)")


# ============================================================================
# 10. 测试/演示代码（来自教材的示例）
# ============================================================================
# 以下代码用于理解各个函数的功能，可以单独运行测试

# 测试数据加载
# X, y = next(iter(train_iter))
# print(f"X shape: {X.shape}, dtype: {X.dtype}")
# print(f"y shape: {y.shape}, dtype: {y.dtype}")

# 查看样本
# X, y = next(iter(data.DataLoader(train_iter.dataset, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y), filename='samples.png')

# 测试softmax函数
# X = torch.normal(0, 1, (2, 5))
# X_prob = softmax(X)
# print(X_prob, X_prob.sum(1))

# 测试索引操作
# y = torch.tensor([0, 2])
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# print(y_hat[[0, 1], y])

# 测试交叉熵损失
# print(cross_entropy(y_hat, y))

# 测试准确率
# print(accuracy(y_hat, y) / len(y))

# 测试模型初始精度
# print(evaluate_accuracy(net, test_iter))

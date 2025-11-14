#!/usr/bin/env python3
import torch.nn as nn  # 导入 PyTorch 神经网络模块

# 定义一个继承自 nn.Module 的神经网络类
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类 nn.Module 的构造函数

        # 第一层卷积层
        # 输入通道: 1（灰度图）
        # 输出通道: 32（特征图数量）
        # 卷积核大小: 3x3
        # 步长: 1
        self.conv1 = nn.Conv2d(1, 32, 3, 1)

        # 第二层卷积层
        # 输入通道: 32
        # 输出通道: 64
        # 卷积核大小: 3x3
        # 步长: 1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # 第一个全连接层
        # 输入特征数: 64 * 5 * 5 = 1600（这是经过两次卷积和两次池化后，特征图展平后的尺寸）
        # 输出特征数: 128（隐藏层神经元数量）
        self.fc1 = nn.Linear(in_features=64 * 5 * 5, out_features=128)

        # 第二个全连接层（输出层）
        # 输入特征数: 128
        # 输出特征数: 10（对应 0-9 共 10 个类别）
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # 第一次卷积和激活
        # 形状变化示例: (N, 1, 28, 28) -> Conv1 -> (N, 32, 26, 26) -> ReLU
        x = nn.functional.relu(self.conv1(x))

        # 第一次最大池化
        # 形状变化示例: (N, 32, 26, 26) -> MaxPool -> (N, 32, 13, 13)
        x = nn.functional.max_pool2d(x, kernel_size=2)

        # 第二次卷积和激活
        # 形状变化示例: (N, 32, 13, 13) -> Conv2 -> (N, 64, 11, 11) -> ReLU
        x = nn.functional.relu(self.conv2(x))

        # 第二次最大池化
        # 形状变化示例: (N, 64, 11, 11) -> MaxPool -> (N, 64, 5, 5)
        x = nn.functional.max_pool2d(x, kernel_size=2)

        # 展平（Flatten）操作，将多维特征图转换为一维向量
        # -1 表示保持 batch size 不变
        # 形状变化示例: (N, 64, 5, 5) -> view -> (N, 1600)
        x = x.view(-1, 64 * 5 * 5)

        # 第一个全连接层和激活
        # 形状变化示例: (N, 1600) -> FC1 -> (N, 128) -> ReLU
        x = nn.functional.relu(self.fc1(x))

        # 第二个全连接层（输出层）
        # 形状变化示例: (N, 128) -> FC2 -> (N, 10)
        # 注意:这里没有应用 Softmax，因为 Softmax 通常集成在 PyTorch 的交叉熵损失函数（nn.CrossEntropyLoss）里。
        x = self.fc2(x)
        return x

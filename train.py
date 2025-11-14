#!/usr/bin/env python3
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from model import NeuralNetwork

# 模型训练函数
def train(train_dataloader, device, model, loss_fn, optimizer):
    # 将模型设置为训练模式
    model.train()
    # 用于累积当前训练轮次的损失值
    running_loss = 0.0
    # 迭代
    for batch, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)  # 输入数据
        labels = labels.to(device)  # 输入标签
        optimizer.zero_grad()  # 清零梯度
        outputs = model(inputs)  # 前向传播
        loss = loss_fn(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        running_loss += loss.item()  # 损失值累加到 running_loss
    print(f'loss: {running_loss/len(train_dataloader):>0.3f}')

# 模型测试函数
def test(dataloader, device, model):
    # 将模型设置为评估模式
    model.eval()
    # 用于累积正确预测的样本数
    correct = 0
    # 用于累积测试的总样本数
    total = 0
    with torch.no_grad():  # 评估模型性能时禁用梯度计算
        for inputs, labels in dataloader:
            inputs = inputs.to(device)  # 输入数据
            labels = labels.to(device)  # 输入标签
            outputs = model(inputs)  # 前向传播
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
            total += labels.size(0)  # 样本数累加到 total
            correct += (predicted == labels).sum().item()  # 正确预测样本数累加到 correct
    print(f'accuracy: {100.0*correct/total:>0.2f} %')

# 主函数
def main():
    print('loading training data...')
    # 使用torchvision.datasets.MNIST自动下载并加载MNIST训练集
    # 训练数据
    train_data = datasets.MNIST(
        root='./data', train=True, download=True, transform=ToTensor()
    )
    print('loading test data...')
    # 测试数据
    test_data = datasets.MNIST(
        root='./data', train=False, download=True, transform=ToTensor()
    )

    # 创建数据加载器（DataLoader），用于将数据集按批（batch_size=64）加载到内存中进行训练
    train_dataloader = DataLoader(train_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    # 检查系统是否支持CUDA GPU，优先使用GPU，否则使用CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device')
    model = NeuralNetwork().to(device)
    print(model)

    # 定义损失函数，使用交叉熵损失，适用于多分类问题
    loss_fn = nn.CrossEntropyLoss()
    # 定义优化器，使用 Adam 优化算法，学习率设置为0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 循环5个轮次（epochs）
    epochs = 5
    for t in range(epochs):
        start_time = time()
        print(f'epoch {t+1} / {epochs}\n----------')
        train(train_dataloader, device, model, loss_fn, optimizer)
        test(test_dataloader, device, model)
        end_time = time()
        print(f'time: {end_time-start_time:>0.2f} seconds')
    print('done!')
    # 训练出来的模型文件路径
    path = 'mnist.pth'
    # 保存模型
    torch.save(model.state_dict(), path)
    print(f'model saved: {path}')

if __name__ == '__main__':
    main()

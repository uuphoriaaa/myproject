#!/usr/bin/env python3
import torch
from torchvision import transforms
from PIL import Image, ImageOps
from model import NeuralNetwork

# 基本配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device}')
model = NeuralNetwork().to(device)
path = './mnist.pth'
model.load_state_dict(torch.load(path))
print(f'loaded model from {path}')
print(model)

# 核心测试函数
def test(path):
    print(f'test {path}...')
    image = Image.open(path).convert('RGB').resize((28, 28))  # 读取图片转换为RGB格式并缩放为28*28像素尺寸
    image = ImageOps.invert(image)  # 反转颜色，反转用户手写图片的白底黑字为MNIST训练数据的黑底白字
    trans = transforms.Compose([
        transforms.Grayscale(1),  # 将RGB图像转换为单通道灰度图
        transforms.ToTensor()  # 将PIL图像转换为PyTorch张量
    ])
    image_tensor = trans(image).unsqueeze(0).to(device)  # 添加一个Batch维度
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 评估模型性能时禁用梯度计算
        output = model(image_tensor)  # 前向传播
        probs = torch.nn.functional.softmax(output[0], 0)  # 对模型的原始输出应用Softmax函数转换为10个类别的概率分布
        predict = torch.argmax(probs).item()  # 获取预测结果
        return predict, probs[predict], probs  # 返回预测的数字、该数字对应的概率值，完整的10个概率值张量

def main():
    for i in range(10):
        predict, prob, probs = test(f'./input/test-{i}.png')  # 循环10次，测试10张图片
        print(f'expected {i}, actual {predict}, {prob}, {probs}')

if __name__ == '__main__':
    main()

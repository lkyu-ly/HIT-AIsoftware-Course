import collections
import itertools
import random
import time
import timeit
from functools import cache

import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.nn import Conv2d, Linear, Module, ReLU, Sequential


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, datacsv, transform=None):
        super(MyDataset, self).__init__()
        with open(f"{root}/{datacsv}", "r") as f:
            imgs = []
            # 读取csv信息到imgs列表
            for path, label in map(lambda line: line.rstrip().split(","), f):
                imgs.append((path, int(label)))
            self.imgs = imgs
        self.transform = transform if transform is not None else lambda x: x

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = self.transform(Image.open(path).convert("1"))
        return img, label

    def __len__(self):
        return len(self.imgs)


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnnLayers = Sequential(Conv2d(1, 1, kernel_size=1, stride=1, bias=True))
        self.linearLayers = Sequential(ReLU(), Linear(16, 2))

    def forward(self, x):
        x = self.cnnLayers(x)
        x = x.view(x.shape[0], -1)
        x = self.linearLayers(x)
        return x


def chooseData(dataset, scale):
    # 将类别为1的排序到前面
    dataset.imgs.sort(key=lambda x: x[1], reverse=True)
    # 获取类别1的数量，取scale倍的数组，得数据不那么偏斜
    trueNum = collections.Counter(itertools.chain.from_iterable(dataset.imgs))[1]
    end = min(trueNum * scale, len(dataset))
    dataset.imgs = dataset.imgs[:end]
    random.shuffle(dataset.imgs)


def evaluateAccuracy(dataIter, net):
    accSum, n = 0.0, 0
    with torch.no_grad():
        for X, y in dataIter:
            accSum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return accSum / n


import matplotlib.pyplot as plt


def trainAndTestWithPlot(
    net, trainIter, testIter, loss, numEpochs, batchSize, optimizer
):
    epochs = []
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    @cache
    def trainAndTest(net, trainIter, testIter, loss, numEpochs, batchSize, optimizer):
        for epoch in range(numEpochs):
            trainLossSum, trainAccSum, n = 0.0, 0.0, 0
            for X, y in trainIter:
                yHat = net(X)
                l = loss(yHat, y).sum()
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                # 计算训练准确度和loss
                trainLossSum += l.item()
                trainAccSum += (yHat.argmax(dim=1) == y).sum().item()
                n += y.shape[0]

            # 评估测试准确度
            testAcc = evaluateAccuracy(testIter, net)

            # 记录数据
            epochs.append(epoch + 1)
            train_losses.append(trainLossSum / n)
            train_accuracies.append(trainAccSum / n)
            test_accuracies.append(testAcc)

            # 打印结果
            print(
                "epoch {:d}, loss {:.4f}, train acc {:.3f}, test acc {:.3f}".format(
                    epoch + 1, trainLossSum / n, trainAccSum / n, testAcc
                )
            )

    trainAndTest(net, trainIter, testIter, loss, numEpochs, batchSize, optimizer)
    print(f"{trainAndTest.cache_info()}")

    # 绘图
    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(epochs, train_losses, color=color, label="Loss", linestyle="--")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Accuracy", color=color)
    ax2.plot(epochs, train_accuracies, color="tab:blue", label="Train Accuracy")
    ax2.plot(epochs, test_accuracies, color="tab:green", label="Test Accuracy")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.title("Training Loss and Accuracy")
    ax1.legend()
    ax2.legend()
    plt.show()


root = "data"
trainData = MyDataset(
    root=root, datacsv="trainDataInfo.csv", transform=transforms.ToTensor()
)

testData = MyDataset(
    root=root, datacsv="testDataInfo.csv", transform=transforms.ToTensor()
)
scale = 10
chooseData(trainData, scale)
print(len(trainData))
print(len(testData))

# 超参数
batchSize = 64
lr = 0.1
numEpochs = 10

trainIter = Data.DataLoader(dataset=trainData, batch_size=batchSize, shuffle=True)
testIter = Data.DataLoader(dataset=testData, batch_size=batchSize)

# 交叉熵损失函数
loss = nn.CrossEntropyLoss()
net = Net()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

trainAndTestWithPlot(net, trainIter, testIter, loss, numEpochs, batchSize, optimizer)

import copy
import time
import torchvision.transforms as transforms
import torch
from torchvision.datasets import FashionMNIST
import numpy as np
from model import ResNet18
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd


def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=227),transforms.ToTensor()]),
                              download=True)

    train_data, val_data = Data.random_split(train_data, [int(len(train_data)*0.8),int(len(train_data)*0.2)])
    train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=4,pin_memory=True)
    val_loader = Data.DataLoader(dataset=val_data, batch_size=64, shuffle=True, num_workers=4,pin_memory=True)

    return train_loader, val_loader

def train_model(model, train_loader, val_loader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #定义设备
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 将模型放入设备中
    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    since = time.time()
    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch+1, num_epochs)) # python计数从0开始
        print('-' * 10)

        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        val_num = 0
        train_num = 0

        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.train()  # 使得模型进入训练模式
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)

            loss = criterion(output, b_y) # 这个loss是这个批次的平均loss值
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_x.size(0) # 总误差需要用平均loss值来计算
            train_acc += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval() # 模型验证模式
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
            val_loss += loss.item() * b_x.size(0)
            val_acc += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)

        train_losses.append(train_loss / train_num)
        val_losses.append(val_loss / val_num)

        train_accuracies.append((train_acc.double() / train_num).item()) # 要item()转tensor to 标量
        val_accuracies.append((val_acc.double() / val_num).item())

        # 只取该轮次最后的损失
        print("{} Train Loss: {:.4f} Train Acc: {:.4f}"
              .format(epoch, train_losses[-1], train_accuracies[-1])) # :表示要填值
        print("{} Val Loss: {:.4f} Val Acc: {:.4f}"
              .format(epoch, val_losses[-1], val_accuracies[-1]))

        if val_accuracies[-1] > best_acc:
            best_acc = val_accuracies[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        # 训练耗时
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

    torch.save(best_model_wts, './best_model.pth')

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss": train_losses, "train_acc": train_accuracies,
                                       "val_loss": val_losses, "val_acc": val_accuracies})
    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(16, 10))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process["train_acc"], 'ro-', label='Train Acc')
    plt.plot(train_process["epoch"], train_process["val_acc"], 'bs-', label='Val Acc')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_loss"], 'ro-', label='Train Loss')
    plt.plot(train_process["epoch"], train_process["val_loss"], 'bs-', label='Val Loss')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.show()

if __name__ == "__main__":
    # 模型实例化
    model = ResNet18(1)
    # 加载数据集
    train_loader, val_loader = train_val_data_process()
    # 训练
    train_process = train_model(model, train_loader, val_loader, num_epochs=20)
    # 画图
    matplot_acc_loss(train_process)



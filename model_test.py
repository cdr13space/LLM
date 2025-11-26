import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.utils.data as Data
from torchvision.datasets import FashionMNIST
from model import ResNet18
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def test_data_process():
    test_data = FashionMNIST(root='./data',train=False,
                             transform=transforms.Compose([transforms.Resize(size=227),transforms.ToTensor()]),
                             download=True)

    test_loader = Data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    return test_loader

def test_model(model, test_loader):
    val_acc = 0.0
    val_num = 0
    real_record = []
    pre_record = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad(): # 训练完了无需进行反向传播，自动微分，参数更新
        for  step, (b_x,b_y) in enumerate(test_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()
            output = model(b_x)

            test_label = torch.argmax(output, dim=1)

            val_acc += torch.sum(test_label == b_y.data).item()
            val_num += 1
            result =  test_label.item()

            print("预测值：{} ------ 真实值： {}".format(result, b_y.item()))
            pre_record.append(result)
            real_record.append(b_y.item())

        result_record = pd.DataFrame(data={"predict":pre_record,"real":real_record})
        val_acc = val_acc / val_num

    print("Test Acc: {:.4f}".format(val_acc))
    return result_record


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18(1).to(device)
    model.load_state_dict(torch.load('./best_model.pth'))
    test_loader = test_data_process()
    result_record = test_model(model, test_loader)
    result_record.to_csv('./result.csv', index=False)






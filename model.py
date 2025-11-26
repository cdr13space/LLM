

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class Residual_Block(nn.Module):
    def __init__(self, in_channels, num_channels, use_conv11=False, stride=1):
        super(Residual_Block, self).__init__()
        self.Relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=num_channels)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=num_channels)
        if use_conv11:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        out = self.Relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv3 is not None:
            out = out + self.conv3(x)
        else:
            out = out + x
        out = self.Relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, in_channels):
        super(ResNet18, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
        )
        self.b2 = nn.Sequential(
            Residual_Block(in_channels=64, num_channels=64, use_conv11=False, stride=1),
            Residual_Block(in_channels=64, num_channels=64, use_conv11=False, stride=1)
        )
        self.b3 = nn.Sequential(
            Residual_Block(in_channels=64, num_channels=128, use_conv11=True, stride=2),
            Residual_Block(in_channels=128, num_channels=128, use_conv11=False, stride=1)
        )
        self.b4 = nn.Sequential(
            Residual_Block(in_channels=128, num_channels=256, use_conv11=True, stride=2),
            Residual_Block(in_channels=256, num_channels=256, use_conv11=False, stride=1)
        )
        self.b5 = nn.Sequential(
            Residual_Block(in_channels=256, num_channels=512, use_conv11=True, stride=2),
            Residual_Block(in_channels=512, num_channels=512, use_conv11=False, stride=1)
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=10),
        )
    def forward(self, x):
        out = self.b1(x)
        out = self.b2(out)
        out = self.b3(out)
        out = self.b4(out)
        out = self.b5(out)
        out = self.b6(out)

        return out



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18(in_channels=3)
    model = model.to(device)
    torchsummary.summary(model, input_size=(3, 224, 224))

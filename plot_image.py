from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt
train_data = FashionMNIST(root='./data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize((224, 224)),
                                                        transforms.ToTensor()]),
                          download=True)

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
batch_x = b_x.squeeze().numpy()
batch_y = b_y.numpy()
classes_labels = train_data.classes

plt.figure(figsize=(16, 10))
for i in range(len(b_y)):
    plt.subplot(4, int(len(b_y)/4), i + 1)
    plt.imshow(batch_x[i], cmap='gray')
    plt.title(classes_labels[int(b_y[i])])

plt.show()
print(b_y.data)





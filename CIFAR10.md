PyTorch之CIFAR10

@[TOC](PyTorch之CIFAR10)

# 前言
其实一直想学深度学习，都2021年了，还不学点深度学习恐将被社会淘汰，难得有这么好的一段时间，那就开始吧。本期内容以PyTorch官网60分钟入门教程里面的CIFAR10项目为蓝本，动手实验了一番加了一些自己的理解，很多都是依葫芦画瓢，从从本文你将要学到

- 如何利用torchvision读取datasets数据集并正规化处理
- 如何定义一个简单的卷积神经网络
- 如何定义损失函数
- 如何用定义好的网络来训练数据
- 如何在测试集上测试
- 如何把训练好的模型保存到本地
- 如何重新加载本地模型对新数据进行预测

# 背景
CIFAR10是[kaggle计算机视觉竞赛](https://www.kaggle.com/c/cifar-10/overview)的一个图像分类项目。该数据集共有60000张32*32彩色图像，一共分为"plane", "car", "bird","cat", "deer", "dog", "frog","horse","ship", "truck" 10类，每类6000张图。有50000张用于训练，构成了5个训练批，每一批10000张图；10000张用于测试，单独构成一批。

![CIFAR10](https://img-blog.csdnimg.cn/20201110161419575.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3plbmdib3dlbmdvb2Q=,size_16,color_FFFFFF,t_70#pic_center)

# 读取数据，正规化处理

一些经典的数据集，如Imagenet, CIFAR10, MNIST都可以通过torchvision来获取，并且torchvision还提供了transforms类可以用来正规化处理数据。

```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train =True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=0)
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
```
# 数据可视化

主要是看一看CIFAR10里面到底是一些什么图片，利用matplotlib模块进行可视化,此时需要将原来正规化的数据再还原回去，原正规数据主要是用来建模用。

```python
import matplotlib.pyplot as plt
import numpy as np

def imshow(img): #绘图，看一下CIFAR10里面是什么东西
    img = img/2+0.5
    npimg =img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

预览结果


```python
Files already downloaded and verified
Files already downloaded and verified
  car plane  deer  frog
```

# 定义卷积神经网络

神经网络是深度学习最核心的东西，直接关系到最后结果的好坏，本次依葫芦画瓢利用nn类搭建一个2层的卷积网络。

```python

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module): #继承的nn.Module类
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net() #网络实例化 
```
# 设定损失函数和收敛准则

需要告诉网络什么时候训练结束，以什么标准结束，这就是损失函数和收敛准则要做的事情了。

```python
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
criterion = nn.CrossEntropyLoss()
```

# 训练数据

当一切准备工作就绪，就该利用设计好的网络来对数据进行训练了

```python
for epoch in range(2):

    running_loss =0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if i% 2000 ==1999:
            print("[%d, %5d] loss: %.3f" %(epoch +1, i+1, running_loss/2000))
            running_loss = 0.0
print("Finished Training")
```
# 测试数据

训练的模型好不好，只有经过测试集测试才知道，需要看模型在测试集上面的表现,可以看总体的准确率如何，还可以进一步看每一类的准确率如何。

```python
correct = 0
total =0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total +=labels.size(0)
        correct += (predicted==labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' %(100 *correct/ total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
```

输出预览

```python
[1,  2000] loss: 2.230
[1,  4000] loss: 1.938
[1,  6000] loss: 1.721
[1,  8000] loss: 1.589
[1, 10000] loss: 1.516
[1, 12000] loss: 1.462
[2,  2000] loss: 1.396
[2,  4000] loss: 1.336
[2,  6000] loss: 1.313
[2,  8000] loss: 1.308
[2, 10000] loss: 1.291
[2, 12000] loss: 1.284
Finished Training
Accuracy of the network on the 10000 test images: 53 %
Accuracy of plane : 54 %
Accuracy of   car : 71 %
Accuracy of  bird : 60 %
Accuracy of   cat :  7 %
Accuracy of  deer : 34 %
Accuracy of   dog : 62 %
Accuracy of  frog : 77 %
Accuracy of horse : 41 %
Accuracy of  ship : 68 %
Accuracy of truck : 53 %
```
# 保存模型

当新建的模型通过测试集的考验后效果还不如就可以保存到本地，方便以后调用进行新数据的预测

```python
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
```
# 调用本地模型预测

模型最终是要拿来用的，所以保存在本地的数据要给其用武之地，于是要加载模型，对新数据进行预测

```python

imshow(torchvision.utils.make_grid(images))
print("GroundTruth: ", ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = Net()
net.load_state_dict(torch.load(PATH))
dataiter = iter(testloader)
images, labels = dataiter.next()
outputs = net(images)
```

输出预览

```python
GroundTruth:    car horse plane plane
预测结果为： tensor([[ 3.4536,  7.1814, -0.7881, -2.3248, -0.7801, -4.2190, -2.5536, -2.3879,
          1.4071,  2.3139],
        [ 0.0749, -1.8976,  1.2119, -0.7189,  3.3229,  0.6713, -2.0983,  8.1980,
         -5.5707, -2.7974],
        [ 6.1693,  1.6508,  5.0074, -0.9648, -2.4487, -2.3632, -5.0884, -4.9947,
          6.4254, -3.8381],
        [ 1.5996, -1.0670,  0.1218, -0.0796,  1.4387,  0.0360, -0.6056, -0.0158,
         -0.9976, -1.4885]], grad_fn=<AddmmBackward>)
```
PyTorch之TensorBoard

@[TOC](PyTorch之tensorboard)

TensorBoard是专门为TensorFlow打造的可视化工具包，PyTorch从1.2.0版本开始，正式集成了TensorBoard，可以像在TensorFlow里面一样调用TensorBoard进行机器学习可视化工作, 不再需要借助于其他包如Visdom, TensorBoardX等。今天我们就来学习一下TensorBoard工具包， 从这篇文章，你将要学到TensorBoard的背景，TensorBoard的应用，TensorBoard的常见错误排查。

# TensorBoard的背景与介绍

TensorBoard是谷歌为TensorFlow配套打造的可视化工具，能够帮助开发人员跟踪，调试和评估模型，据闻Facebook在为PyTorch打造TensorBoard的时候与谷歌团队有着非常精紧密的合作。

![tensorboard](D:/项目/torch_test/tensorboard.gif)

这篇文章主要介绍PyTorch配套的TensorBoard，TensorBoard 主要目的是为机器学习实验提供所需的可视化功能和工具，具体包括

- 跟踪和可视化损失及准确率等指标；
- 可视化模型图（操作和层）；
- 查看权重、偏差或其他张量随时间变化的直方图；
- 将嵌入投射到较低的维度空间；
- 显示图片、文字和音频数据；
- 剖析PyTorch程序；

# TensorBoard的应用

TensorBoard省去了以往在机器学习过程中保存数据再自己手动绘图的麻烦，而把这些工作像积木一样可以add到TensorBoard数据看板上面。下面以Fashion-MNIST数据集为例，展示一下 Tensorboard 的基本使用方法。

## 导库导数据及数据可视化

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))]) #定义数据标准化变换

trainset = torchvision.datasets.FashionMNIST('./data',
download = True, train =True, transform = transform) #加载训练数据集
testset = torchvision.datasets.FashionMNIST('./data',
download= True, train = False, transform=transform) #加载测试数据集

trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 0) #训练集形成一批一批的可迭代的数据对象

testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle = True, num_workers =0) #测试集形成一批一批的可迭代的数据对象

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot') #10个标签类

def matplotlib_imshow(img, one_channel = False): #样本数据可视化
    if one_channel:
        img = img.mean(dim =0)
    img = img /2 +0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap = 'Greys')
    else:
        plt.imshow(np.transpose(npimg, (1,2,0)))
```

## 定义神经网络，损失函数和优化函数

要进行神经网络训练，首先要定义出网络结构和网络中参数的优化方式以及评判优化好坏的损失函数。

```python
class Net(nn.Module): #定义一个神经网络类
    def __init__(self): #网络变量
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x): #网络结构
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net() #实例化
criterion = nn.CrossEntropyLoss() #交叉信息熵，损失函数
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum= 0.9) #优化方式为随机梯度下降
```
## TensorBoard初始化

把TensorBoard当成一个数据看板进行初始化，后面的绘图之类的工作都可以add到这个看板上面去，初始化后会在代码的同级文件夹创建一个runs/fashion_mnist_experiment_1文件夹用来存放可视化对象。


```python
from torch.utils.tensorboard import SummaryWriter #tensorboard初始化
writer = SummaryWriter('runs/fashion_mnist_experiment_1') #创建一个runs/fashion_mnist_experiment_1文件夹
```
![image](D:/项目/torch_test/tensorboard_first_view.png)
## add image

有了TensorBoard数据看板，现在就可以往里面add数据图像图表，如同win10桌面应用磁贴一样，比如随机抽取4个训练样本组合网格拼图写入TensorBoard里面。

```python
dataiter = iter(trainloader) #创建迭代器
images, labels = dataiter.next() #获取迭代器中的下一个值
img_grid = torchvision.utils.make_grid(images) #将若干幅图像拼成一幅图像
matplotlib_imshow(img_grid, one_channel= True) #调用matplotlib_imshow显示图片
writer.add_image('four_fashion_mnist_images', img_grid) #images写入到TensorBoard
```
## add  graph

TensorBoard不单单可以将样本数据可视化，更重要的还可以监察模型，展示模型结构，可以点击节点的“+”号展开查看更详细的参数和结构。

```python
writer.add_graph(net, images) #神经网络graph写入tensorboard
writer.close()
```
![graph](D:/项目/torch_test/graph.png)

## add Projector

一些高维数据还可以通过UMAP, PCA，T-SNE等方法降维到2维或者3维等低维空间进行可视化,主要通过add_embedding函数实现投影。

```python
def select_n_random(data, labels, n = 100): #随机抽样100个
    assert len(data)==len(labels) #断言
    perm = torch.randperm(len(data)) #先置换
    return data[perm][:n], labels[perm][:n] #再取前100个

images, labels = select_n_random(trainset.data, trainset.targets) #从训练集抽样100个

class_labels = [classes[lab] for lab  in labels] #标签列表
features = images.view(-1, 28 * 28) #数据重塑
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1)) #add_embedding手段将高维数据低维表示
writer.close()
```
![projector](D:/项目/torch_test/projector.png)

## tracking model & add scalar

TensorBoard还能对模型训练过程和评估过程进行跟踪，同时，常用的如 loss、accuracy 等都是数值标量，也可以通过TensorBoard进行展示。

```python
def images_to_probs(net, images): #概率预测
    output =net(images) 
    _, preds_tensor = torch.max(output, 1) 
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim = 0)[i].item() for i, el in zip(preds, output)] #softmax多个输出值转换成多个概率值

def plot_classes_preds(net, images, labels): #概率类别结果可视化
    preds, probs = images_to_probs(net, images) 
    fig = plt.figure(figsize = (12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks =[], yticks = [])
        matplotlib_imshow(images[idx], one_channel= True)
        ax.set_title("{0}, {1:.1}%\n(label: {2}".format(
            classes[preds[idx]],
            probs[idx]*100.0,
            classes[labels[idx]]),
            color = ("green" if preds[idx]==labels[idx].item() else "red")) #颜色标志
    return fig

running_loss = 0.0
for epoch in range(1): 
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:   
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(trainloader) + i)
            writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(net, inputs, labels),
                            global_step=epoch * len(trainloader) + i)
            running_loss = 0.0
print('Finished Training')
```
![scalar](D:/项目/torch_test/scalar.png)

## assessing trained model

在经典的分类模型中，常常通过precision-recall曲线（PR CURVES）来帮助评估模型好坏，在TensorBoard里面同样可以更简便的做到。

```python
class_probs = []
class_preds = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = net(images)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]
        _, class_preds_batch = torch.max(output, 1)

        class_probs.append(class_probs_batch)
        class_preds.append(class_preds_batch)

test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_preds = torch.cat(class_preds)

def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0): #绘制precision-recall曲线
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]
    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

for i in range(len(classes)): #为每一类绘制precision-recall曲线
    add_pr_curve_tensorboard(i, test_probs, test_preds)
```

![](D:/项目/torch_test/prcurve.png)

# TensorBoard demo

命令行路径切换到runs/fashion_mnist_experiment_1文件夹所在路径，运行下面列命令启动 Tensorboard，再浏览器输入localhost://6006即可遨游TensorBoard数据面板了。

```python
 tensorboard --logdir=runs
```
# 常见错误

跑代码的时候遇到如下warning

```python
warning: Embedding dir exists, did you set global_step for add_embedding()?
```
字面意思是embedding路径已经存在，需要为add_embedding()函数进行全局化设置，可能原因是因为之前

```python
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
```
已经创建一个runs/fashion_mnist_experiment_1文件夹里面包含add_embedding文件导致冲突，不清楚什么具体原因，如果有知情朋友还请留言告知，谢谢。

add_embedding源码
```python
def add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None):
        """Add embedding projector data to summary.
        Args:
            mat (torch.Tensor or numpy.array): A matrix which each row is the feature vector of the data point
            metadata (list): A list of labels, each element will be convert to string
            label_img (torch.Tensor or numpy.array): Images correspond to each data point. Each image should be square.
            global_step (int): Global step value to record
            tag (string): Name for the embedding
        Shape:
            mat: :math:`(N, D)`, where N is number of data and D is feature dimension
            label_img: :math:`(N, C, H, W)`, where `Height` should be equal to `Width`.
        Examples::
            import keyword
            import torch
            meta = []
            while len(meta)<100:
                meta = meta+keyword.kwlist # get some strings
            meta = meta[:100]
            for i, v in enumerate(meta):
                meta[i] = v+str(i)
            label_img = torch.rand(100, 3, 32, 32)
            for i in range(100):
                label_img[i]*=i/100.0
            writer.add_embedding(torch.randn(100, 5), metadata=meta, label_img=label_img)
            writer.add_embedding(torch.randn(100, 5), label_img=label_img)
            writer.add_embedding(torch.randn(100, 5), metadata=meta)
        """
        from .x2num import make_np
        mat = make_np(mat)
        if global_step is None:
            global_step = 0
            # clear pbtxt?
        # Maybe we should encode the tag so slashes don't trip us up?
        # I don't think this will mess us up, but better safe than sorry.
        subdir = "%s/%s" % (str(global_step).zfill(5), self._encode(tag))
        save_path = os.path.join(self._get_file_writer().get_logdir(), subdir)
        try:
            os.makedirs(save_path)
        except OSError:
            print(
                'warning: Embedding dir exists, did you set global_step for add_embedding()?')
        if metadata is not None:
            assert mat.shape[0] == len(
                metadata), '#labels should equal with #data points'
            make_tsv(metadata, save_path, metadata_header=metadata_header)
        if label_img is not None:
            assert mat.shape[0] == label_img.shape[0], '#images should equal with #data points'
            assert label_img.shape[2] == label_img.shape[3], 'Image should be square, see tensorflow/tensorboard#670'
            make_sprite(label_img, save_path)
        assert mat.ndim == 2, 'mat should be 2D, where mat.size(0) is the number of data points'
        make_mat(mat, save_path)
        # new funcion to append to the config file a new embedding
        append_pbtxt(metadata, label_img,
                     self._get_file_writer().get_logdir(), subdir, global_step, tag)
```

# 参考文献
1， TensorBoard官方教程
https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
2， TensorFlow的官方教程
https://tensorflow.google.cn/tensorboard




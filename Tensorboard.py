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

from torch.utils.tensorboard import SummaryWriter #tensorboard初始化
writer = SummaryWriter('runs/fashion_mnist_experiment_1') #创建一个runs/fashion_mnist_experiment_1文件夹

dataiter = iter(trainloader) #创建迭代器
images, labels = dataiter.next() #获取迭代器中的下一个值
img_grid = torchvision.utils.make_grid(images) #将若干幅图像拼成一幅图像
matplotlib_imshow(img_grid, one_channel= True) #调用matplotlib_imshow显示图片
writer.add_image('four_fashion_mnist_images', img_grid) #images写入到TensorBoard

writer.add_graph(net, images) #神经网络graph写入tensorboard
writer.close()

def select_n_random(data, labels, n = 100): #随机抽样100个
    assert len(data)==len(labels) #断言
    perm = torch.randperm(len(data)) #先置换
    return data[perm][:n], labels[perm][:n] #再取前100个

images, labels = select_n_random(trainset.data, trainset.targets) #从训练集抽样100个

class_labels = [classes[lab] for lab  in labels] #标签列表
features = images.view(-1, 28 * 28) #数据重塑
writer.add_embedding(features,
                    metadata=class_labels,
                    global_step=0,
                    label_img=images.unsqueeze(1)) #add_embedding手段将高维数据低维表示
writer.close()

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

# 利用tensorboard评估训练模型
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

for i in range(len(classes)): #为每一类标签绘制precision-recall曲线
    add_pr_curve_tensorboard(i, test_probs, test_preds)


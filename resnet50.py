import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as Data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES']='0'
## 注意：
## 因为一开始是通过对图片进行resize改变wh，因为是灰度图导致图像resize畸形，也不能用transfroms.ToPILImage，也是因为单通道的缘故吧
## 因此最后手动将tensor ->numpy -> Image，通过Image.resize将28x28 ->32, 32. 
## 也可以通过将resnet第一个卷积kernel_size从7降到5，这样使输入tensor从32x32变为28x28，避免对图片resize
## 不使用resize第一个epoch准确率达到了96.5%。而简单的两层隐藏层的神经网络第一个epoch准确率为90.85%

## conv_bloack是在shotcut上多了次卷积和BN操作，来改变通道数, 并且通过stride减少wh。
## 而identity block输入和输出的维度和大小不变

## parameters
LR = 0.0001
Batch_size = 32
EPOCH = 1

## model
class identity_block(nn.Module):
    def __init__(self, in_channels, filters):
        super(identity_block, self).__init__()
        filter1, filter2, filter3 = filters
        self.conv1 = nn.Conv2d(in_channels, filter1, kernel_size=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(filter1)
        self.conv2 = nn.Conv2d(filter1, filter2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filter2)
        self.conv3 = nn.Conv2d(filter2, filter3, kernel_size=1, bias=False) ##因为卷积后有BN，所以卷积过程偏执没用，因为bias=False
        self.bn3 = nn.BatchNorm2d(filter3)
        self.relu = nn.ReLU(inplace=True) ##直接修改，不拷贝

    def forward(self, x):
        short_cut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x) 

        x = x + short_cut
        x = self.relu(x)  ## 在两者add后再ReLU
        return x

class conv_block(nn.Module): ## 负责改变通道数和压缩wh
    def __init__(self, in_channels, filters, stride=2):
        super(conv_block, self).__init__()
        filter1, filter2, filter3 = filters
        self.conv1 = nn.Conv2d(in_channels, filter1, kernel_size=1, stride=stride, bias=False) 
        self.bn1 = nn.BatchNorm2d(filter1)
        self.conv2 = nn.Conv2d(filter1, filter2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filter2)
        self.conv3 = nn.Conv2d(filter2, filter3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filter3)
        self.conv_short = nn.Conv2d(in_channels, filter3, kernel_size=1, stride=stride, bias=False)
        self.bn_short = nn.BatchNorm2d(filter3)
        self.relu = nn.ReLU(inplace=True)

    
    def forward(self, x):
        short_cut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        short_cut = self.conv_short(short_cut)
        short_cut = self.bn_short(short_cut)

        x = x + short_cut
        x = self.relu(x)
        return x

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer2_1 = conv_block(64, [64, 64, 256], stride=1)
        self.layer2_2 = identity_block(256, [64, 64, 256])
        self.layer2_3 = identity_block(256, [64, 64, 256])

        self.layer3_1 = conv_block(256, [128, 128, 512], stride=2)
        self.layer3_2 = identity_block(512, [128, 128, 512])
        self.layer3_3 = identity_block(512, [128, 128, 512])
        self.layer3_4 = identity_block(512, [128, 128, 512])

        self.layer4_1 = conv_block(512, [256, 256, 1024], stride=2)
        self.layer4_2 = identity_block(1024, [256, 256, 1024])
        self.layer4_3 = identity_block(1024, [256, 256, 1024])
        self.layer4_4 = identity_block(1024, [256, 256, 1024])

        self.layer5_1 = conv_block(1024, [512, 512, 2048], stride=2)
        self.layer5_2 = identity_block(2048, [512, 512, 2048])
        self.layer5_3 = identity_block(2048, [512, 512, 2048])

        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) ##表示输出的结果是1x1大小
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)

        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)
        x = self.layer3_4(x)

        x = self.layer4_1(x)
        x = self.layer4_2(x)
        x = self.layer4_3(x)
        x = self.layer4_4(x)

        x = self.layer5_1(x)
        x = self.layer5_2(x)
        x = self.layer5_3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

## 数据处理
transform = transforms.Compose([
    transforms.Resize((32,32)), 
    transforms.ToTensor()
])

## data_loader
train_data = torchvision.datasets.MNIST(
    root='../mnist',  ##表示MNIST数据集下载/已存在的位置，../表示是相对于当前py文件上一级目录的mnist文件夹
    train=True,
    transform=transform,
    download=False  ## 如果没有下载就改为True自动下载
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=Batch_size, shuffle=True)
test_data = torchvision.datasets.MNIST(root='../mnist', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:1000]
test_x_1 = torch.empty(test_x.size(0), 1, 32, 32)
for i,v in enumerate(test_x):   
    temp = v[0].numpy()
    temp = Image.fromarray(temp)
    #temp = transforms.ToPILImage()(v[0])  ## 自己动手将tensor转换为Image，不要用这个函数，血的教训
    temp = transforms.Resize((32,32))(temp)
    temp = np.array(temp)
    temp = torch.Tensor(temp)/255.
    test_x_1[i][0] = temp
test_x = test_x_1.cuda()
test_x_1 = 0
test_y = test_data.test_labels[:1000] ## volatile=True表示依赖这个节点的所有节点都不会进行反向求导，用于测试集

'''  展示图片，测试用的，看看resize是否正常
test_img = test_x[1]
test_img = test_img.squeeze(0).cpu().data.numpy()
plt.imshow(test_img)
plt.show()
'''
## loss, optimizer
resnet50 = ResNet50()
resnet50.cuda()

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet50.parameters(), lr=LR)

## train
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        x = Variable(x).cuda()
        y = Variable(y).cuda()
        out_put = resnet50(x)
        optimizer.zero_grad()
        loss = loss_func(out_put, y)
        loss.backward()
        optimizer.step()
        
        if step%100 == 0:
            print(step)
 
#resnet50.eval()  ##单个图片predict要加eval()，不过我直接用2000个数据集测试准确率，其实可以注释掉
test_output = resnet50(test_x)
pred_y = torch.max(test_output, 1)[1].cpu().data.squeeze()
test_y = test_y.data.numpy()
pred_y = pred_y.data.numpy()
accuracy = sum(pred_y == test_y)/ len(test_y)
print('Epoch:', epoch, '|train loss:%.4f' % loss.item(), '|test accuracy:',accuracy)

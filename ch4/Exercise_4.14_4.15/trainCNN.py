import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init

#用于图像分类的CNN网络
class CNN_NET(torch.nn.Module):
    def __init__(self):
        super(CNN_NET,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 3,
                                     out_channels = 64,
                                     kernel_size = 5,
                                     stride = 1,
                                     padding = 0)
        self.pool = torch.nn.MaxPool2d(kernel_size = 3,
                                       stride = 2)
        self.conv2 = torch.nn.Conv2d(64,64,5)
        self.fc1 = torch.nn.Linear(64*4*4,384)
        self.fc2 = torch.nn.Linear(384,192)
        self.fc3 = torch.nn.Linear(192,10)

    def close(self):
        self.conv1.requires_grad = False
        self.pool.requires_grad = False
        self.conv2.requires_grad = False
        self.fc1.requires_grad = False
        self.fc2.requires_grad = False
        self.fc3.requires_grad = False

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,64*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    #hyper parameter
    BATCH_SIZE = 128
    EPOCH = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root='../data2',train = True,
                                        download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size = BATCH_SIZE,
                                          shuffle = True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='../data2',train = False,
                                       download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset,batch_size = BATCH_SIZE,
                                         shuffle = False, num_workers=1)

    #预训练好语用任务的分类网络
    net = CNN_NET()
    optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
    loss_func =torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        print(epoch)
        running_loss = 0.0
        for step, data in enumerate(trainloader):
            b_x,b_y=data
            outputs = net.forward(b_x)
            loss = loss_func(outputs, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 打印状态信息
            running_loss += loss.item()
            if step % 100 == 99:    # 每100批次打印一次
                print('[%d, %5d] loss: %.3f' % (epoch + 1, step + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training!')

    correct = 0
    total = 0
    with torch.no_grad():
        #不计算梯度，节省时间
        for (images,labels) in testloader:
            outputs = net(images)
            numbers,predicted = torch.max(outputs.data,1)
            total +=labels.size(0)
            correct+=(predicted==labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    #保存模型
    torch.save(net,r'D:\paper coding\ADJSCC_Task\model\img_class.pth')
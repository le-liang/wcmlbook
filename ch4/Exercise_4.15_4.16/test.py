import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
from torch.utils import data
from encoder import Encoder
from deepjscc import DeepJSCC
from torchvision.utils import save_image
import numpy as np
from torchvision import transforms
import math

#计算PSNR指标
def PSNR(loss):
    return 10 * math.log10(1/loss)

# 数据集的预处理
transform = transforms.Compose([transforms.ToTensor()])

data_path = r'D:\paper coding\ADJSCC(image reconstruction)\ADJSCC_Task\data2'
# 获取数据集
train_data = CIFAR10(data_path,train=True,transform=transform,download=False)
test_data = CIFAR10(data_path,train=False,transform=transform,download=False)
#train_data = train_data[0] #只取图片，不需要label
#test_data = test_data[0]

#迭代器生成
train_loader = data.DataLoader(train_data,batch_size=128,shuffle=True)

#导入训练好的模型参数
model = torch.load("./model/BDJSCC_19_7.pth")
path = './model/pytorch-cifar-models-master/'
model2 = torch.hub.load(path, "cifar10_resnet44", source='local', pretrained=True)
model.SNR = 20
model.ratio = math.sqrt(1 / (math.pow(10,model.SNR/10)))

#样本可视化
from torchvision import transforms

#定义损失
loss_func = nn.MSELoss()
max_val = 0

for epoch in range(1):
    for i,(x,y) in enumerate(train_loader):
        batch_x = Variable(x) # torch.Size([128, 1, 28, 28])
        #batch_y = Variable(y) # torch.Size([128])
        # 获取最后输出
        out = model.forward1(batch_x) # torch.Size([128,10])
        loss = loss_func(out,batch_x)
        max_val = max_val * i / (i+1) + PSNR(loss.item()) / (i+1)
        if i % 50 == 0:
            print('{}:\t'.format(i), PSNR(loss.item()))
        #print(out.shape)
        # if i % 10 == 0:
        #     #save_image(batch_x,'./picture/'+str(i)+'a.jpg')
        #     #save_image(out,'./picture/'+str(i)+'b.jpg')

        #     print('{}:\t'.format(i), loss.item())
        #     save_image(batch_x, './picture/' + str(i) + 'a.jpg')
        #     save_image(out, './picture/' + str(i) + 'b.jpg')

print(max_val)
import torch
import torchvision
import torch.nn as nn
import math
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
from torch.utils import data
from deepjscc import DeepJSCC
import numpy as np
from torchvision import transforms
from trainCNN import CNN_NET

#计算PSNR指标
def PSNR(loss):
    return 10 * math.log10(1/loss)

def EachImg(img):
    img=img/2+0.5   #将图像数据转换为0.0->1.0之间，才能正常对比度显示（以前-1.0->1.0色调对比度过大）
    plt.imshow(np.transpose(img,(1,2,0)))
    plt.show()

#加速计算
torch.manual_seed(3407)
#CPU\GPU转换
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集的预处理
transform = transforms.Compose([transforms.ToTensor()])

data_path = r'D:\paper coding\BDJSCC(Baseline)\ADJSCC_Task\data2'
# 获取数据集
train_data = CIFAR10(data_path,train=True,transform=transform,download=False)
test_data = CIFAR10(data_path,train=False,transform=transform,download=False)

#样本可视化
image = train_data[0][0]
#print(image)
#EachImg(image)

#迭代器生成
train_loader = data.DataLoader(train_data,batch_size=64,shuffle=True)
# train_loader = data.DataLoader(train_data,batch_size=128,shuffle=True,num_workers=8)
test_loader = data.DataLoader(test_data,batch_size=100,shuffle=True)
# test_loader = data.DataLoader(test_data,batch_size=100,shuffle=True,num_workers=8)

#定义损失和优化器
model = DeepJSCC(19) #设置信道SNR为1dB
# model = torch.load('./model/BDJSCC_19_6.pth')
loss_func = nn.MSELoss()
loss_func2 = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(),lr=0.0001)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)

if __name__ == '__main__':
    #训练网络
    loss_count = []
    for epoch in range(800):
        if epoch % 10 == 0:
            print(epoch)
            torch.save(model,r'D:\paper coding\BDJSCC(Baseline)\ADJSCC_Task\model\BDJSCC_19_7.pth')

        for i,(x,y) in enumerate(train_loader):
            batch_x = Variable(x) # torch.Size([128, 1, 28, 28])
            #batch_y = Variable(y) # torch.Size([128])
            # 获取最后输出
            #print(batch_x[:,0,:,:].unsqueeze(1).shape)
            #model.SNR = torch.randint(0,21,(1,)).item()
            #model.ratio = math.sqrt(1 / (math.pow(10,model.SNR/10)))

            out1 = model.forward1(batch_x) # torch.Size([128,10])
            loss1 = loss_func(out1,batch_x)
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss1.backward() # 误差反向传播，计算参数更新值
            opt.step() # 将参数更新值施加到net的parmeters上
            # 使用优化器优化损失

            if i % 120 == 0:
                loss_count.append(loss1.detach().numpy())
                print('{}:\t'.format(i), loss1.item())

        #scheduler.step()

    print('PSNR:',PSNR(loss1),'dB')
    torch.save(model,r'D:\paper coding\BDJSCC(Baseline)\ADJSCC_Task\model\BDJSCC_19_7.pth')
    plt.figure('PyTorch_CNN_Loss')
    plt.plot(loss_count,label='Loss')
    plt.legend()
    plt.show()
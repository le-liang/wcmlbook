import torch
import math
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from channel import channel_forward,channel_forward2
from trainCNN import CNN_NET

class DeepJSCC(nn.Module):
    def __init__(self,SNR):
        super(DeepJSCC,self).__init__()
        self.encoder1 = Encoder()
        self.decoder1 = Decoder()
        #self.img_class = torch.load("./model/img_class.pth") #导入预先训练好的CNN网络
        #self.img_class.close() #关闭不必要的语用任务模块梯度计算
        self.SNR = SNR
        self.ratio = math.sqrt(1 / (math.pow(10,self.SNR/10)))

    def forward1(self,x):
        x = self.encoder1.forward(x,self.SNR)
        x = channel_forward(x,self.ratio)
        x = self.decoder1.forward(x,self.SNR)
        return x

    def forward2(self,x):
        x = self.encoder1.forward(x,self.SNR)
        x = channel_forward2(x,self.ratio)
        x = self.decoder1.forward(x,self.SNR)
        return x
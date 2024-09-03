import torch
import math
import torch.nn as nn
from fl_module import FL1
from af_module import AF

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.fl1 = FL1(5,3,3,1,0)
        self.af1 = AF(3)
        self.fl2 = FL1(3,3,16,2,1)
        self.af2 = AF(16)
        self.fl3 = FL1(3,16,32,2,1)
        self.af3 = AF(32)
        self.fl4 = FL1(3,32,64,2,1)


    def forward(self, x,SNR):
        x = self.fl1.forward(x)
        #x = self.af1.forward(x,SNR)
        x = self.fl2.forward(x)
        #x = self.af2.forward(x,SNR)
        x = self.fl3.forward(x)
        #x = self.af3.forward(x,SNR)
        x = self.fl4.forward(x)
        #加入能量Pnorm模块
        temp = torch.sum(torch.sum(torch.sum(x*x,dim=1),dim=1),dim=1)
        #print(temp.shape)
        self.norm = torch.sqrt(temp.reshape(x.shape[0],1,1,1))
        x = x * (1 / self.norm) * math.sqrt(64*4*4/2)
        #print(x.shape)
        return x

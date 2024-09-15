import torch
import torch.nn as nn
from gdn import GDN

class FL1(nn.Module):
    def __init__(self,F,D_in,D_out,S,P):
        super(FL1,self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=D_in,
                      out_channels=D_out,
                      kernel_size=F,
                      stride=S,
                      padding=P)
        )
        self.gdn = GDN(D_out,'cpu')
        self.last = nn.PReLU()

    def forward(self,x):
        x = self.module(x)
        x = self.gdn.forward(x)
        x = self.last(x)
        return x

class FL2(nn.Module):
    def __init__(self,F,D_in,D_out,S,P):
        super(FL2,self).__init__()
        self.module = nn.Sequential(
            nn.ConvTranspose2d(in_channels=D_in,
                      out_channels=D_out,
                      kernel_size=F,
                      stride=S,
                      padding=P)
        )
        self.gdn = GDN(D_out,'cpu',inverse=True)
        self.last = nn.PReLU()

    def forward(self,x):
        x = self.module(x)
        x = self.gdn.forward(x)
        x = self.last(x)
        return x
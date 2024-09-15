import torch
import numpy as np

#暂时只考虑AWGN信道
def channel_forward(x,ratio):
    # return (x + 0.1 * torch.randn_like(x)) #信噪比为20dB
    return (x + ratio * torch.randn_like(x)) #信噪比为SNR，可指定


#考虑慢瑞利衰落和AWGN信道
def channel_forward2(x,ratio):
    temp = x.reshape(x.shape[0],1,2,(int)(x.shape[1]*x.shape[2]*x.shape[3]/2))
    h = torch.randn(x.shape[0],1,2,2)
    h[:,:,1,0] = -h[:,:,0,1]
    h[:,:,1,1] = h[:,:,0,0]
    h = h * np.sqrt(1/2)
    #print(h.shape)
    temp = torch.matmul(h,temp) #记得乘上归一化因子
    #print(temp.shape)
    x = temp.reshape(x.shape)
    #print(x.shape)
    return (x + ratio * torch.randn_like(x)) #信噪比为SNR，可指定
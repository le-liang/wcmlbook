import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time
import scipy.io as sio
import os
import math
import function_wmmse_powercontrol as wf
class DNN(nn.Module):
    def __init__(self, n_input, h_hidden1, h_hidden2, h_hidden3, n_output):
        super(DNN, self).__init__()
        self.fc_1 = nn.Linear(n_input, h_hidden1)
        self.fc_1.weight.data.normal_(0, 0.1)
        self.fc_2 = nn.Linear(h_hidden1, h_hidden2)
        self.fc_2.weight.data.normal_(0, 0.1)
        self.fc_3 = nn.Linear(h_hidden2, h_hidden3)
        self.fc_3.weight.data.normal_(0, 0.1)
        self.fc_4 = nn.Linear(h_hidden3, n_output)
        self.fc_4.weight.data.normal_(0, 0.1)

    def forward(self, x, input_keep_prob=1, hidden_keep_prob=1):
        # m = nn.Dropout(1-input_keep_prob)
        # n = nn.Dropout(1-hidden_keep_prob)
        x = F.relu(self.fc_1((x)))
        x = F.relu(self.fc_2((x)))
        x = F.relu(self.fc_3((x)))
        output = F.sigmoid(self.fc_4((x)))
        return output

class PowerControl:
    def __init__(self, X, Y, traintestsplit=0.01, n_hidden_1=200, n_hidden_2=80, n_hidden_3=80, LR=0.0001):
        self.num_total = X.shape[1]  # number of total samples
        self.num_val = int(self.num_total * traintestsplit)  # number of validation samples
        self.num_train = self.num_total - self.num_val  # number of training samples
        self.X_train = torch.tensor(np.transpose(X[:, 0:self.num_train]), dtype=torch.float32)  # training data
        self.Y_train = torch.tensor(np.transpose(Y[:, 0:self.num_train]), dtype=torch.float32)  # training label
        self.X_val = torch.tensor(np.transpose(X[:, self.num_train:self.num_total]), dtype=torch.float32)  # validation data
        self.Y_val = torch.tensor(np.transpose(Y[:, self.num_train:self.num_total]), dtype=torch.float32)  # validation label
        self.n_input = X.shape[0]  # input size
        self.n_output = Y.shape[0]  # output size
        self.lr = LR
        self.DNNs = []
        self.DNNpara = list()
        for i in range(5):
            self.DNN = DNN(self.n_input, n_hidden_1, n_hidden_2, n_hidden_3, self.n_output)
            self.DNNs.append(self.DNN)
            self.DNNpara += list(self.DNN.parameters())
        self.optimizer = torch.optim.RMSprop(self.DNNpara, lr=self.lr)
        self.lamda = 1.0
    def train(self, location, training_epochs=300, batch_size= 100,  LRdecay=0):
        input_keep_prob = 1
        hidden_keep_prob = 1
        total_batch = int(self.num_total/ batch_size)
        start_time = time.time()
        MSETime = np.zeros((training_epochs, 3))
        for i in range(len(self.DNNs)):
            self.DNNs[i].train()
        for epoch in range(1200):
            for i in range(total_batch):
                idx = np.random.randint(self.num_train, size=batch_size)
                for i in range(len(self.DNNs)):
                    pred = self.DNNs[i](self.X_train[idx, :], input_keep_prob, hidden_keep_prob)
                    sum_rate = self.sum_rate(self.X_train[idx, :].detach(), pred, batch_size)
                    self.loss = -sum_rate
                    self.optimizer.zero_grad()
                    self.loss.backward()
                    self.optimizer.step()
                    if epoch % (10) == 0:
                        print('epoch:%d, ' % epoch, 'train:%0.2f%%, ' % (self.loss * 100))

        for epoch in range(training_epochs):
            for i in range(total_batch):
                sum_rate = []
                idx = np.random.randint(self.num_train, size=batch_size)
                for i in range(len(self.DNNs)):
                    pred = self.DNNs[i](self.X_train[idx, :], input_keep_prob, hidden_keep_prob)
                    sum_rate.append(self.sum_rate(self.X_train[idx, :].detach(), pred, batch_size))
                self.loss = -max(sum_rate)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                # if LRdecay:
                #     self.lr = self.lr/(epoch+1)

            MSETime[epoch, 0] = np.asarray(self.loss.item())
            # MSETime[epoch, 1] = np.asarray(-(self.sum_rate(self.X_val.detach(), self.DNN(self.X_val).detach(), self.num_val)).item())
            MSETime[epoch, 2] = np.asarray(time.time() - start_time)
            if epoch%(10)==0:
                print('epoch:%d, '%epoch, 'train:%0.2f%%, '%(self.loss*100))
        print("training time: %0.2f s" % (time.time() - start_time))

        sio.savemat('MSETime_qos%d_%d_%d' % (self.n_output, batch_size, self.lr * 10000),
                    {'train': MSETime[:, 0], 'validation': MSETime[:, 1], 'time': MSETime[:, 2]})
        # torch.save(self.DNN.state_dict(), location)
        return 0

    def test(self, H, X, save_name, model_path, binary=0):
        # self.DNN.load_state_dict(torch.load(model_path))
        for i in range(len(self.DNNs)):
            self.DNNs[i].eval()
        X = torch.tensor(np.transpose(X), dtype=torch.float32)
        start_time = time.time()
        num_sample = H.shape[2]
        nnrate = np.zeros(num_sample)
        for j in range(num_sample):
            rate = []
            for i in range(len(self.DNNs)):
                pred = self.DNNs[i](X).detach()[j, :]
                rate.append(wf.obj_IA_sum_rate(H[:, :, j], pred, 1, H.shape[0]))
            nnrate[j] = max(rate)
        # testtime = time.time() - start_time
        # sio.savemat(save_name, {'pred': y_pred})
        return nnrate

    def sum_rate(self, H, P, num_sample, K=10, var_noise=1):
        H = torch.transpose(H, 0, 1)
        H = torch.tensor(np.reshape(H, (K, K, H.shape[1]), order="F"), dtype=torch.float32).clone().detach()
        nnrate = torch.zeros(num_sample)
        nnrate = self.obj_IA_sum_rate(H, P, var_noise, K, num_sample)
        return torch.mean(nnrate)
    def obj_IA_sum_rate(self, H, p, var_noise, K, num_sample):
        y = torch.zeros(num_sample)
        r = torch.zeros((num_sample, K))
        r_min = torch.full((num_sample, K), 1.0)
        for i in range(K):
            s = var_noise
            for j in range(K):
                if j != i:
                    s = s + H[i, j, :] ** 2 * p[:, j]
            y = y + torch.log2(1 + H[i, i, :] ** 2 * p[:, i] / s)
            r[:, i] = F.relu(r_min[:, i] - torch.log2(1 + H[i, i, :] ** 2 * p[:, i] / s))
        r = self.lamda * torch.sum(r, dim=1)
        return y-r

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time
import scipy.io as sio
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
        m = nn.Dropout(1-input_keep_prob)
        n = nn.Dropout(1-hidden_keep_prob)
        x = F.relu(self.fc_1(n(x)))
        x = F.relu(self.fc_2(n(x)))
        x = F.relu(self.fc_3(n(x)))
        output = F.relu6(self.fc_4(m(x)))/6
        return output

class PowerControl:
    def __init__(self, X, Y, traintestsplit=0.01, n_hidden_1=200, n_hidden_2=200, n_hidden_3=200, LR=0.001):
        self.num_total = X.shape[1]  # number of total samples
        num_val = int(self.num_total * traintestsplit)  # number of validation samples
        self.num_train = self.num_total - num_val  # number of training samples
        self.X_train = torch.tensor(np.transpose(X[:, 0:self.num_train]), dtype=torch.float32)  # training data
        self.Y_train = torch.tensor(np.transpose(Y[:, 0:self.num_train]), dtype=torch.float32)  # training label
        self.X_val = torch.tensor(np.transpose(X[:, self.num_train:self.num_total]), dtype=torch.float32)  # validation data
        self.Y_val = torch.tensor(np.transpose(Y[:, self.num_train:self.num_total]), dtype=torch.float32)  # validation label
        self.n_input = X.shape[0]  # input size
        self.n_output = Y.shape[0]  # output size
        self.lr = LR
        self.DNN = DNN(self.n_input, n_hidden_1, n_hidden_2, n_hidden_3, self.n_output)
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.DNN.parameters(), lr=self.lr)
    def train(self, location,training_epochs=300, batch_size=1000,  LRdecay=0):
        input_keep_prob = 1
        hidden_keep_prob = 1
        total_batch = int(self.num_total/ batch_size)
        start_time = time.time()
        MSETime = np.zeros((training_epochs, 3))
        self.DNN.train()
        for epoch in range(training_epochs):
            for i in range(total_batch):
                idx = np.random.randint(self.num_train, size=batch_size)
                pred = self.DNN(self.X_train[idx, :], input_keep_prob, hidden_keep_prob)
                loss = self.loss_func(pred, self.Y_train[idx, :].detach())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if LRdecay:
                    self.lr = self.lr/(epoch+1)

            MSETime[epoch, 0] = np.asarray(loss.item())
            MSETime[epoch, 1] = np.asarray(self.loss_func(self.DNN(self.X_val), self.Y_val).item())
            MSETime[epoch, 2] = np.asarray(time.time() - start_time)
            if epoch%(int(training_epochs/10))==0:
                print('epoch:%d, '%epoch, 'train:%0.2f%%, '%(loss.item()*100), 'validation:%0.2f%%.'%(MSETime[epoch, 1]*100))
        print("training time: %0.2f s" % (time.time() - start_time))

        sio.savemat('MSETime_%d_%d_%d' % (self.n_output, batch_size, self.lr * 10000),
                    {'train': MSETime[:, 0], 'validation': MSETime[:, 1], 'time': MSETime[:, 2]})
        torch.save(self.DNN.state_dict(), location)
        return 0

    def test(self, X, save_name, model_path, binary=0):
        self.DNN.load_state_dict(torch.load(model_path))
        self.DNN.eval()
        X = torch.tensor(np.transpose(X), dtype=torch.float32)
        start_time = time.time()
        y_pred = self.DNN(X)
        testtime = time.time() - start_time
        if binary == 1:
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0
        y_pred = np.asarray(y_pred.detach())
        sio.savemat(save_name, {'pred': y_pred})
        return testtime

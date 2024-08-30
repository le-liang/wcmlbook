import torch
import numpy as np
from scipy.optimize import minimize, differential_evolution, LinearConstraint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def log_string(logger, str):
    logger.info(str)
    print(str)

def cal_W_opt_de(P):
    n = len(P)
    index = np.zeros((n, n), dtype=np.int)
    idx_list = np.arange(int(n * (n - 1) / 2))
    idx = 0

    def w_list_to_matrix(w_l):
        idx = 0
        W = np.zeros((n, n))
        for i in range(n):
            W[i, i + 1:] = w_l[idx:idx + (n - i - 1)]
            idx += n - i - 1
        W += W.T
        for i in range(n):
            W[i, i] = 1 - np.sum(W[:, i])
        return W

    def obj(w_l):
        # change w_l to W
        W = w_list_to_matrix(w_l)
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    # print('terms', i, np.outer(P[i, :]*W[i, :], P[i,:]*W[i, :]))
                    C[i, j] = 1 - 2 * np.sum(P[i, :] * W[i, :]) + 2 * np.sum(P[i, :] * W[i, :] * W[i, :]) + np.sum(
                        np.outer(P[i, :] * W[i, :], P[i, :] * W[i, :])) - np.sum(
                        (P[i, :] * W[i, :] * P[i, :] * W[i, :]))
                else:
                    C[i, j] = np.sum(P[i, :] * W[i, :] * P[j, :] * W[j, :]) + P[i, j] * W[i, j] * (
                            2 - np.sum(P[i, :] * W[i, :]) - np.sum(P[j, :] * W[j, :]))
                    C[j, i] = C[i, j]
        w, v = np.linalg.eigh(C - np.ones_like(P) / n)
        # print('w:', W, max(w))
        return max(w)
    for i in range(n):
        index[i, i + 1:] = idx_list[idx:idx + (n - i - 1)]
        idx += n - i - 1
    index += index.T
    contraint_list = []
    def create_cons(idx):
        return lambda w: 1 - sum(w[idx])
    current_index = []
    for i in range(n):
        current_index.append([t for t in index[i] if t != -1])
        contraint_list.append({'type': 'ineq', 'fun': create_cons(current_index[i])})


    sol = differential_evolution(obj,  bounds=(((0, 1),) * len(idx_list)), disp=True)
    print(sol.x, sol.fun)
    return w_list_to_matrix(sol.x)


def cal_W_opt(P):
    n = len(P)
    index = np.zeros((n, n), dtype=np.int)
    idx_list = np.arange(int(n * (n - 1) / 2))
    idx = 0
    def w_list_to_matrix(w_l):
        idx = 0
        W = np.zeros((n, n))
        for i in range(n):
            W[i, i + 1:] = w_l[idx:idx + (n - i - 1)]
            idx += n - i - 1
        W += W.T
        for i in range(n):
            W[i, i] = 1 - np.sum(W[:, i])
        return W

    def obj(w_l):
        # change w_l to W
        W = w_list_to_matrix(w_l)
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    # print('terms', i, np.outer(P[i, :]*W[i, :], P[i,:]*W[i, :]))
                    C[i, j] = 1 - 2 * np.sum(P[i, :] * W[i, :]) + 2 * np.sum(P[i, :] * W[i, :] * W[i, :]) + np.sum(
                        np.outer(P[i, :] * W[i, :], P[i, :] * W[i, :])) - np.sum(
                        (P[i, :] * W[i, :] * P[i, :] * W[i, :]))
                else:
                    C[i, j] = np.sum(P[i, :] * W[i, :] * P[j, :] * W[j, :]) + P[i, j] * W[i, j] * (
                                2 - np.sum(P[i, :] * W[i, :]) - np.sum(P[j, :] * W[j, :]))
                    C[j, i] = C[i, j]
        w, v = np.linalg.eigh(C - np.ones_like(P) / n)
        # print('w:', W, max(w))
        return max(w)
    for i in range(n):
        index[i, i + 1:] = idx_list[idx:idx + (n - i - 1)]
        idx += n - i - 1
    index += index.T
    contraint_list = []
    def create_cons(idx):
        return lambda w: 1 - sum(w[idx])
    current_index = []
    for i in range(n):
        current_index.append([t for t in index[i] if t != -1])
        contraint_list.append({'type': 'ineq', 'fun': create_cons(current_index[i])})

    sol = minimize(obj, np.ones(len(idx_list)) * (1 / (n-1)), method='SLSQP', constraints=contraint_list,
                   bounds=(((0, 1),) * len(idx_list)), options={'maxiter': 2000, 'disp': True})

    #
    #init_vector =np.random.uniform(size=len(idx_list))
    #sol = minimize(obj, init_vector, method='SLSQP', constraints=contraint_list,
    #               bounds=(((0, 1),) * len(idx_list)), options={'maxiter': 2000, 'disp': True}, )
    W_opt = w_list_to_matrix(sol.x)
    return W_opt


def cal_w_tcp_2(P, thresh):
    adj_matrix = P > thresh
    tcp_degree = np.sum(adj_matrix, axis=0)
    W_tcp = np.zeros_like(P)
    for i in range(len(W_tcp)):
        for j in range(i + 1, len(W_tcp)):
            if adj_matrix[i, j]:
                W_tcp[i, j] = W_tcp[j, i] = 1 / (max(tcp_degree[i], tcp_degree[j]) + 1)
    for i in range(len(W_tcp)):
        W_tcp[i, i] = 1 - np.sum(W_tcp[:, i])
    return W_tcp, adj_matrix

def UDP_comm(model_set, W, P):
    para_old = []
    for model in model_set:
        para = {}
        for n, p in model.named_parameters():
            para[n] = p.clone()
        para_old.append(para)
    for rx in range(len(model_set)):
        for n, p in model_set[rx].named_parameters():
            for tx in range(len(model_set)):
                if tx == rx:
                    continue
                mask = torch.rand(p.size()) < P[tx, rx]
                p.data[mask] += (para_old[tx][n][mask] - para_old[rx][n][mask]) * W[tx, rx]

def TCP_comm(model_set, W, A):
    para_old = []
    for model in model_set:
        para = {}
        for n, p in model.named_parameters():
            para[n] = p.clone()
        para_old.append(para)
    for rx in range(len(model_set)):
        for n, p in model_set[rx].named_parameters():
            p.data = W[rx, rx] * p.data
            for tx in range(len(model_set)):
                if tx == rx or A[tx, rx] == False:
                    continue
                p.data += para_old[tx][n] * W[tx, rx]

def local_update(datas, model_set, optimizer_set, criterion):
    average_loss = 0
    average_acc = 0
    n_workers = len(model_set)
    for i in range(n_workers):
        data, target = datas[i]
        data, target = data.to(device), target.to(device)
        optimizer_set[i].zero_grad()
        output = model_set[i](data)
        pred = output.argmax(dim=1, keepdim=True)
        average_acc += pred.eq(target.view_as(pred)).sum().item() / len(target)
        loss = criterion(output, target)
        loss.backward()
        optimizer_set[i].step()
        average_loss += loss.item()
    return average_loss / n_workers, average_acc / n_workers


def train(train_loader, model_set, optimizer_set, epoch, communication_mode, criterion, P, W, A,logger):
    for model in model_set:
        model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    for batch_idx, datas in enumerate(train_loader):
        loss, acc = local_update(datas, model_set, optimizer_set, criterion)
        if communication_mode == 'UDP':
            UDP_comm(model_set, W, P)
        else:
            TCP_comm(model_set, W, A)
        losses.update(loss, len(datas[0]))
        top1.update(acc, 1)
    log_string(logger, 'Train Epoch: {} Loss: {:.6f} \t Accuracy: {:.6f}'.format(epoch, losses.avg, top1.avg))

def validate(val_loader, model_set, mean_model, criterion, logger):
    losses = AverageMeter()
    top1 = AverageMeter()
    for k in range(len(model_set)):
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model_set[k](data)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
            losses.update(loss.item(), data.size(0))
            top1.update(acc, data.size(0))
    log_string(logger, 'Evaluation Mean Performance: * Prec@1 {top1.avg:.3f} Loss {losses.avg:.3f}'.format(top1=top1, losses=losses))

def lr_update(scheduler_set):
    for scheduler in scheduler_set:
        scheduler.step()

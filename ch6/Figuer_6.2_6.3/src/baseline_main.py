#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils import get_dataset
from options import args_parser
from update import DatasetSplit,test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar


if __name__ == '__main__':
    args = args_parser()
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    idxs = np.arange(len(train_dataset))
    idxs_train = idxs[:int(0.9 * len(idxs))]
    idxs_test = idxs[int(0.9 * len(idxs)):]
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    rainloader = DataLoader(DatasetSplit(train_dataset, idxs_train),
                            batch_size=64, shuffle=True)
    testloader = DataLoader(DatasetSplit(train_dataset, idxs_test),
                            batch_size=int(len(idxs_test) / 10), shuffle=False)
    criterion = torch.nn.NLLLoss().to(device)


    epoch_loss = []
    epoch_accuracy = []
    total, correct = 0.0, 0.0

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            global_model.train()
            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs_t = global_model(images)
            _, pred_labels = torch.max(outputs_t, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        accuracy = correct / total
        print('\nTrain accuracy:', accuracy)
        epoch_accuracy.append(accuracy)

    # Plot loss
    plt.figure()
    plt.title('Training Loss vs Epochs')
    plt.plot(range(len(epoch_loss)), epoch_loss,color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Train loss')
    plt.savefig('./save/nn_{}_{}_{}_loss.png'.format(args.dataset, args.model,
                                                 args.epochs))
    epoch_loss_record = np.array(epoch_loss).reshape(1,args.epochs)
    df1 = pd.DataFrame(data=epoch_loss_record, columns=range(len(epoch_loss)),
                      index=['Loss']).round(2)
    df1.to_csv('./save/nn_{}_{}_{}_loss.csv'.format(args.dataset, args.model,
                                                 args.epochs))
    # Plot accuracy
    plt.figure()
    plt.title('Training Accuracy vs Epochs')
    plt.plot(range(len(epoch_accuracy)), epoch_accuracy,color='k')
    plt.xlabel('Epochs')
    plt.ylabel('Train accuracy')
    plt.savefig('./save/nn_{}_{}_{}_acc.png'.format(args.dataset, args.model,
                                                args.epochs))
    epoch_accuracy_record = np.array(epoch_accuracy).reshape(1, args.epochs)
    df2 = pd.DataFrame(data=epoch_accuracy_record, columns=range(len(epoch_accuracy)),
                      index=['Accuracy']).round(2)
    df2.to_csv('./save/nn_{}_{}_{}_acc.csv'.format(args.dataset, args.model,
                                              args.epochs))

    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))




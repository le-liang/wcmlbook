import copy
import torchvision
from torch.utils.data import DataLoader
from utils_datasets import *
from utils_models import *
from utils_training import *
from scipy.spatial.distance import cdist
import argparse
import logging


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='resnet20', help='model name [default: resnet20]')
    parser.add_argument('--epochs', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate in training')
    parser.add_argument('--log_dir', type=str, default='./logs', help='log directory ')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--communication_mode', type=str, default='TCP', help='optimizer for training')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--k', type=float, default=0.7, help='expontional decay factor')
    parser.add_argument('--r', type=float, default=0.4, help='distance normalizer')
    parser.add_argument('--threshold', type=float, default=0.7, help='threshold of communi graph')
    parser.add_argument('--weights_type', type=str, default='OPT', help='select weight types, OPT or UNI')
    parser.add_argument('--np_random_seed', type=int, default=100, help='randomness of communication graph')
    parser.add_argument('--torch_random_seed', type=int, default=200, help='randomness of model training')
    return parser.parse_args()



def main(args):
    ''' --------- CREATE DIR ---------- '''
    log_dir = args.log_dir
    data_dir = args.data_dir
    n_workers = args.num_workers
    np.random.seed(args.np_random_seed)
    torch.manual_seed(args.torch_random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''--------- LOG -----------'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s_%s_%s_k_%s_r_%s_thresh_%s_seed_%s_%s.txt' % (log_dir, args.model, args.communication_mode,args.weights_type, str(args.k), str(args.r), str(args.threshold), str(args.np_random_seed), str(args.torch_random_seed)))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string(logger, 'PARAMETER ...')
    log_string(logger, args)

    ''' --------- DATA LOADING ---------'''
    log_string(logger, 'Load dataset ...')
    dataset = torchvision.datasets.CIFAR10
    data_mean = (0.4914, 0.4822, 0.4465)
    data_stddev = (0.2023, 0.1994, 0.2010)
    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop((32, 32), 4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )
    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )
    training_set = dataset(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = dataset(root=data_dir, train=False, download=True, transform=transform_test)
    dataset_split_train = torch.utils.data.random_split(training_set, uniformly_split_dataset(training_set, n_workers))
    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(ConcatDataset(dataset_split_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128)
    print('size of training data loader:', len(ConcatDataset(dataset_split_train)))

    '''--------- MODEL LOADING ---------'''
    learning_rate = args.learning_rate
    criterion = torch.nn.CrossEntropyLoss()
    mean_model = resnet20()
    mean_model.train()
    mean_model.to(device)
    model_set = [copy.deepcopy(mean_model).to(device) for i in range(n_workers)]
    optimizer_set = [torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) for model in model_set]
    scheduler_set = [torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 100]) for optimizer in optimizer_set]

    ''' --------- COMMUNICATION NETWORK GENERATION ---------'''
    positions = np.random.uniform(size=(n_workers, 2))
    distances = cdist(positions, positions, 'euclidean')
    k = args.k
    r = args.r
    P = k ** ((distances / r) ** 2)
    np.fill_diagonal(P, 0)
    communication_mode = args.communication_mode
    A = []
    if communication_mode == 'TCP':
        W, A = cal_w_tcp_2(P, args.threshold)
    elif communication_mode == 'UDP':
        if args.weights_type == 'OPT':
            W = cal_W_opt(P)
        elif args.weights_type == 'UNI':
            W = np.ones_like(P)/len(P)
        else:
            raise Exception('Please define the weight type (Optimal or Uniform)!')
    else:
        raise Exception('Please define the communication mode (TCP or UDP)!')

    '''--------- TRAINING ---------'''
    epoch_num = args.epochs
    for epoch in range(epoch_num):
        train(train_loader, model_set, optimizer_set, epoch, communication_mode, criterion, W=W, P=P, A=A, logger=logger)
        validate(test_loader, model_set, mean_model, criterion, logger)
        lr_update(scheduler_set)

if __name__ == '__main__':
    args = parse_args()
    main(args)






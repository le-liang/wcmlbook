import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

def uniformly_split_dataset(data, n):
    res = [round(len(data) / n) for x in range(n)]
    res[-1] = len(data) - sum(res) + res[-1]
    return res





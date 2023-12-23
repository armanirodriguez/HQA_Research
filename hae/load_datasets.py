import numpy as np
from tqdm import tqdm
#import pytorch_lightning as pl
from matplotlib import pyplot as plt
from collections import OrderedDict

from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader, Subset
from utils import *
from sklearn.model_selection import train_test_split
import os

# File paths for downloading mnist related datasets
MNIST_TRAIN_PATH = '/tmp/mnist'
MNIST_TEST_PATH = '/tmp/mnist_test_'


def _make_train_valid_split(dataset, len_ds_test):
    train_idxs, valid_idxs, _, _ = train_test_split(
            range(len(dataset)),
            dataset.targets,
            stratify = dataset.targets,
            test_size = len_ds_test / len(dataset), 
            random_state = RANDOM_SEED
        )
    ds_train = Subset(dataset, train_idxs)
    ds_valid = Subset(dataset, valid_idxs)
    
    return ds_train, ds_valid

def _make_data_loaders(ds_train, ds_test, validate, return_tiled, num_tiles, tile_split):
    """
    Arguments: 
    - ds_train : Training Dataset instance. 
    - ds_test : Testing Dataset instance."""

    ds_valid = None
    if validate:
        ds_train, ds_valid = _make_train_valid_split(ds_train, len(ds_test))


    dl_train = DataLoader(ds_train, batch_size=MNIST_BATCH_SIZE, shuffle=True, num_workers=NUM_DATA_LOADER_WORKERS)
    dl_test = DataLoader(ds_test, batch_size=MNIST_BATCH_SIZE, shuffle=False, num_workers=NUM_DATA_LOADER_WORKERS)
    dl_valid = None
    if ds_valid:
        dl_valid = DataLoader(ds_valid, batch_size=MNIST_BATCH_SIZE, shuffle=True, num_workers=NUM_DATA_LOADER_WORKERS)
    
    return dl_train, dl_valid, dl_test


def load_mnist(validate = False, return_tiled = False, num_tiles = 2, tile_split = "v"):
    ds_train = MNIST(MNIST_TRAIN_PATH, download=True, train=True, transform=MNIST_TRANSFORM)
    ds_test = MNIST(MNIST_TEST_PATH, download=True, train=False, transform=MNIST_TRANSFORM)
    return _make_data_loaders(ds_train, ds_test, validate, return_tiled, num_tiles, tile_split)


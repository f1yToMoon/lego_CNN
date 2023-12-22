import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pickle

train_file = open('lego-train.pickle', 'rb')
test_file = open('lego-test.pickle', 'rb')
train_data = pickle.load(train_file)
test_data = pickle.load(test_file)

train_data_X = np.array([img for (img, _) in train_data]).astype(np.float32)
train_data_y = np.array([lbl for (_, lbl) in train_data])

test_data_X = np.array([img for (img, _) in test_data]).astype(np.float32)
test_data_y = np.array([lbl for (_, lbl) in test_data])

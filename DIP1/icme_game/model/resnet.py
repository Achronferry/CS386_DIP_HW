import sys 
sys.path.append("..") 
# print(sys.path)
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import os
from read_data import load_batch

#load resnet
def network():
    resnet101 = models.resnet101(pretrained = True)
    for param in resnet101.parameters():
        param.requires_grad = True

    fc_inputs = 2048
    resnet101.fc = nn.Sequential(
        nn.Linear(fc_inputs,10),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(10,1),
        nn.Sigmoid()
    )
    return resnet101
def evaluate():
    




model1 = network()
model2 = network()
model3 = network()
model4 = network()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data,label = load_batch.get_batch()

data = torch.from_numpy(data.transpose(0,3,1,2).to(device)
print(model1(data))
print(label.shape)

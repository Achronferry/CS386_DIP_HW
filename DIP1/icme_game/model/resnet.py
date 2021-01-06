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
import warnings
warnings.simplefilter("ignore")

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
        nn.Linear(10,2),
        nn.Softmax()
    )
    return resnet101

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = network().to(device)  
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1,last_epoch=-1)

def train(model,loss_func,optimizer,choose,batch_size = 32, epochs = 50):
    for epoch in range(epochs):
        scheduler.step()
        train_loss = 0
        size = 0
        for length in range(5000):
            train_data,labels = load_batch.get_batch(choose=choose, batch_size=batch_size)
            # print("length",length)
            for i in range(len(train_data)):
                batch_data = train_data[i]
                batch_label = labels[i]
                batch_data = torch.from_numpy(batch_data.transpose(0,3,1,2)).float().to(device)
                batch_label = torch.from_numpy(batch_label).long().to(device)
                optimizer.zero_grad()
                outputs = model(batch_data)
 
                loss = loss_func(outputs, batch_label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*batch_data.size(0)
                size += batch_data.size(0)
        print("train ---- epoch == > {},loss==>{}".format(epoch,train_loss/size))
        torch.save(model,str(choose)+"/resnet"+str(epoch)+'.pt')

if __name__ =="__main__":
    train(model,loss_func,optimizer,0)
    # model = torch.load("resnet0.pt")
    # print(model)
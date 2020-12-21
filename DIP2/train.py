import torch
from torch import nn

from model import SegNet
from utils import load_dataset


device = torch.device('cuda')


def loss_f(out, gold):
    """
    out: [b, 300, 400]
    gold: [b, 300, 400]
    """
    return nn.MSELoss()(out, gold)


def evaluate(x, y):
    return


if __name__ == "__main__":

    model = SegNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))

    train_dataset, test_dataset = load_dataset('dataset/TrainingData')

    for epoch in range(10):
        epoch_loss = 0
        for i, (x, y) in enumerate(train_dataset.batch(16)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_f(out, y)
            epoch_loss += loss.cpu().item()
            loss.backward()
            optimizer.step()
        print(epoch, epoch_loss/len(train_dataset))



import torch
from torch import nn
import numpy as np

from model import SegNet
from utils import load_dataset
from metrics import eval_all


device = torch.device('cuda')


def loss_f(out, gold):
    """
    out: [b, 300, 400]
    gold: [b, 300, 400]
    """
    return nn.MSELoss()(out, gold)


def evaluate(test_dataset):
    scores = []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_dataset.batch(16)):
            x, y = x.to(device), y.to(device)
            out = model(x)
            for (out_, y_) in zip(out, y):
                out_, y_ = out_.cpu().detach().numpy(), y_.cpu().detach().numpy()
                # print(out_.shape, y_.shape)
                # print(eval_all(out_, y_))
                scores.append(eval_all(out_, y_))
    scores = np.array(scores)
    scores = np.mean(scores, axis=0)
    return scores


if __name__ == "__main__":

    model = SegNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))

    train_dataset, test_dataset = load_dataset('dataset/TrainingData')

    scores_avg = evaluate(test_dataset)
    print(scores_avg)

    for epoch in range(30):
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
        scores_avg = evaluate(test_dataset)
        print(epoch, scores_avg)
        print()
        model_path = f'models/epoch-{epoch}.pth'
        torch.save(model.state_dict(), model_path)




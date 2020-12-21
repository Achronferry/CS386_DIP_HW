from read_data.data import DatasetFromFolder, TestFromFolder
# coding=utf-8
import time
import argparse
import os
import numpy as np
import operator
from functools import reduce
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from model.network import ResNet50
import torchvision.models as models
import pandas as pd
from scipy import stats
import csv
from utils_set.utils import save_checkpoint_val, adjust_learning_rate, print_log, save_checkpoint_val_best,print_log_test,print_log_scroo,create_csv
# from emd import EMDLoss

# Training settings
parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
parser.add_argument("--tag", type=str, help="tag for this training", default='color')
parser.add_argument("--train", default="/home/ivipc/ICME game", type=str,
                    help="path to load train datasets(default: none)")
parser.add_argument("--test", default="/home/ivipc/ICME game", type=str,
                    help="path to load test datasets(default: none)")
parser.add_argument("--batchSize", type=int, default=1, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0005, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=500, help="step to test the model performance. Default=2000")
parser.add_argument("--cuda", action="store_true", help="Use cuda?", default='--cuda')
parser.add_argument("--gpus", type=int, default=1, help="nums ofz gpu to use")
parser.add_argument("--resume", default=" ", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--num_mcs", type=str, help="num of mcs", default='color')

def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = torch.cumsum(y_true, axis=-1)
    cdf_ypred = torch.cumsum(y_pred, axis=-1)
    samplewise_emd = torch.sqrt(torch.mean(torch.square(torch.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return torch.mean(samplewise_emd)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    global opt, name, logger, model, criterion, SSIM_loss, start_time, mcs_num
    opt = parser.parse_args()
    print(opt)

    # Tag_BatchSize
    name = "%s_%d" % (opt.tag, opt.batchSize)

    mcs_num = "%s" % (opt.num_mcs)

    logger = SummaryWriter("runs/" + name)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    seed = 1334
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    cudnn.benchmark = True

    print("==========> Loading datasets")

    train_dataset = DatasetFromFolder(opt.train, transform=Compose([
        ToTensor()
    ]))
    indoor_test_dataset = TestFromFolder(opt.test, transform=Compose([
        ToTensor()
    ]))



    training_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.threads, batch_size=opt.batchSize,
                                      pin_memory=True, shuffle=True)
    testing_data_loader = DataLoader(dataset=indoor_test_dataset, num_workers=opt.threads, batch_size=1, pin_memory=True,
                                    shuffle=True)


    print("==========> Building model")

    backbone = models.resnet50(pretrained=True)
    model = ResNet50(backbone, num_classes=15)

    # criterion = EMDLoss()
    criterion =nn.L1Loss()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["state_dict"])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['state_dict'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("==========> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    else:
        model = model.cpu()
        criterion = criterion.cpu()

    print("==========> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)

    old = -1


    print("==========> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):

        start_time = time.time()
        val = test(testing_data_loader, epoch)
        train(training_data_loader, optimizer, epoch)

        if val >= old:
            save_checkpoint_val(model, epoch, name, mcs_num)
            old = val

def train(training_data_loader,optimizer, epoch):

    adjust_learning_rate(optimizer, epoch)
    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
    for iteration, batch in enumerate(training_data_loader, 1):

        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        steps = len(training_data_loader) * (epoch-1) + iteration

        data, label = \
            Variable(batch[0]), \
            Variable(batch[1])

        if opt.cuda:
            data = data.cuda().float()
            label = label.cuda().float()

        else:
            data = data.cpu()
            label = label.cpu()

        output = model(data)

        loss = criterion(output, label)

        loss.backward()

        optimizer.step()


        if iteration % 10 == 0:

            one_time = time.time() - start_time
            print_log(one_time, epoch, iteration, len(training_data_loader),
                      loss.item(), mcs_num)

            logger.add_scalar('loss', loss.item(), steps)

        torch.cuda.empty_cache()

def test(test_data_loader, epoch):
    avg = 0
    for iteration, batch in enumerate(test_data_loader, 1):
        model.eval()
        data, label, name = \
            Variable(batch[0], volatile=True), \
            Variable(batch[1]), Variable(batch[2])

        if opt.cuda:
            data = data.cuda()
            label = label.cuda()
            name = name.cpu()
            label = label.cpu()
        else:
            data = data.cpu()
            name = name.cpu

        with torch.no_grad():
            output = model(data)

        output = (1-output)*15

        output = output.cpu()

        output = output.numpy()
        label = label.numpy()
        name = name.numpy()

        # np.set_printoptions(precision=1)

        label = reduce(operator.add, label)
        output = reduce(operator.add, output)

        output1 = pd.Series(output)

        output2 = output1.rank()

        # out = pd.DataFrame(output2)

        #create_csv('/home/ivipc/icme_game/train/training_log/test.csv', output2)

        output3 = output2.values

        print('out{}：{}:'.format(epoch, output3))
        print('lab{}：{}:'.format(epoch, label))

        value = stats.spearmanr(label, output3)

        cor = value.correlation

        avg = cor + avg

        print_log_test(epoch, name, output, label, mcs_num)

    avg = avg/20
    print_log_scroo(epoch, avg, mcs_num)
    return avg

if __name__ == "__main__":
    os.system('clear')
    main()

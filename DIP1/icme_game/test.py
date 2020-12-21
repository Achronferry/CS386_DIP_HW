from read_data.data import DatasetFromFolder, TestFolder
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
parser.add_argument("--resume", default="/home/ivipc/whr/icme_game/checkpoints/Texture_1/Texture_Texture.pth", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
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

    
    indoor_test_dataset = TestFolder(opt.test, transform=Compose([
        ToTensor()
    ]))

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
    



    print("==========> Testing")
    

    start_time = time.time()
    test(testing_data_loader)
        

def test(test_data_loader):
    avg = 0
    for iteration, batch in enumerate(test_data_loader, 1):
        model.eval()
        data, name = \
            Variable(batch[0], volatile=True), \
            Variable(batch[1])

        if opt.cuda:
            data = data.cuda()           
            name = name.cpu()
           
        else:
            data = data.cpu()
            name = name.cpu

        with torch.no_grad():
            output = model(data)

        output = (1-output)*15

        output = output.cpu()

        output = output.numpy()
        name = name.numpy()
        name = str(name[0])

        output = reduce(operator.add, output)

        output1 = pd.Series(output)

        output2 = output1.rank()

        exposure = ['Texture']

        print(output2)
        out = output2.values



        # out = pd.DataFrame(output2.values, columns=exposure, index=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O'])
        name = '0{}'.format(name)
        create_csv('/home/ivipc/whr/icme_game/result/',out,name,exposure)



if __name__ == "__main__":
    os.system('clear')
    main()

# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GCN(nn.Module):
    def __init__(self, dim_1_channels, dim_2_channels):
        super().__init__()

        self.conv1d_1 = nn.Conv1d(dim_1_channels, dim_1_channels, 1)
        self.conv1d_2 = nn.Conv1d(dim_2_channels, dim_2_channels, 1)

    def forward(self, x):
        h = self.conv1d_1(x).permute(0, 2, 1)
        return self.conv1d_2(h).permute(0, 2, 1)

class SPD_GCN(nn.Module):
    def __init__(self, in_channels, mid_channels=4, N=16):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.N = N

        self.phi = nn.Conv2d(in_channels, mid_channels, 1)
        self.theta = nn.Conv2d(in_channels, N, 1)
        self.theta1 = nn.Conv2d(in_channels, N, 1)
        self.gcn = GCN(N, mid_channels)
        self.phi_inv = nn.Conv2d(mid_channels, in_channels, 1)
        self.sigmoid = nn.Tanh()

    def forward(self, x):
        batch_size, in_channels, h, w = x.shape
        mid_channels = self.mid_channels
        N = self.N
        B = self.theta(x).view(batch_size, N, -1)
        B1 = self.theta1(x).view(batch_size, N, -1)
        x_reduced = self.phi(x).view(batch_size, mid_channels, h * w)
        x_reduced = x_reduced.permute(0, 2, 1)
        v = B.bmm(x_reduced)

        z = self.gcn(v)
        y = B1.permute(0, 2, 1).bmm(z).permute(0, 2, 1)
        y = y.view(batch_size, mid_channels, h, w)
        x_res = self.phi_inv(y)

        return self.sigmoid(x_res + x)

class CAD_GCN(nn.Module):
    def __init__(self, in_channels, mid_channels=1, N=15):
        super().__init__()
        self.in_channels = in_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.theta = nn.Conv2d(in_channels, N, 1)
        self.gcn = GCN(N, mid_channels)
        self.mid_channels = mid_channels
        self.N = N
        self.phi_inv = nn.Conv2d(mid_channels, in_channels, 1)
        self.sigmoid = nn.Tanh()

    def forward(self, x):
        batch_size, in_channels, h, w = x.shape
        N = self.N
        mid_channels = self.mid_channels

        x_node = self.pool(x).view(batch_size, in_channels, -1)
        B = self.theta(x).view(batch_size, N, -1)
        z1 = self.gcn(x_node)
        y = B.permute(0, 2, 1).bmm(z1).permute(0, 2, 1)
        y = y.view(batch_size, mid_channels, h, w)
        x_res = self.phi_inv(y)

        return self.sigmoid(x_res + x)

def whitening(im):
    batch_size, channel, h, w = im.shape
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    im = torch.cat([(im[:,[0]]-mean[0])/std[0],
                    (im[:,[1]]-mean[1])/std[1],
                    (im[:,[2]]-mean[2])/std[2]], 1)
    return im

def l2_norm(x):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x = torch.div(x, norm)
    return x

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, dropRate=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.conv2(x1)
        out = x + x2
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out


class ResNet18(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet18, self).__init__()
        self.backbone = model

        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # x = whitening(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class ResNet34(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet34, self).__init__()
        self.backbone = model

        self.fc1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # x = whitening(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class ResNet50(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet50, self).__init__()
        self.backbone = model

        self.fc1 = nn.Linear(2048, 2048)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, num_classes)

        self.refine = nn.Conv2d(15, 3, kernel_size=3, stride=1, padding=1)

        self.conv_1 = nn.Sequential(nn.Conv2d(45, 3, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(3),
                                    nn.ReLU())

        self.conv_2 = nn.Sequential(nn.Conv2d(30, 64, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 15, 3, padding=1, bias=False)
                                    )

        # self.res = ResidualBlock(45)
        # self.res1 = ResidualBlock(45)
        # self.ca = CAD_GCN(15, 1, 15)
        # self.sp = SPD_GCN(15, 15, 16)
        self.ad = nn.AdaptiveAvgPool2d(1)
        self.sg = nn.Sigmoid()


    def forward(self, x):
        #x = whitening(x)

        # H_feature = self.res(x)

        H_F = self.conv_1(x)

        # H_i = self.sp(H_image)
        #
        # H_j = self.ca(H_image)
        #
        # H_m = torch.cat((H_i, H_j), 1)
        #
        # H_G = self.conv_2(H_m)
        #
        # H_F = self.refine(H_G)

        x = self.backbone.conv1(H_F)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)

        x = self.ad(x)
        
        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.sg(self.fc2(x))

        return x


class ResNet101(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet101, self).__init__()
        self.backbone = model

        self.fc1 = nn.Linear(8192, 2048)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        #x = whitening(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class ResNet152(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet152, self).__init__()
        self.backbone = model

        self.fc1 = nn.Linear(8192, 2048)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, num_classes)


    def forward(self, x):
        #x = whitening(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = l2_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

if __name__ == '__main__':
    backbone = models.resnet101(pretrained=True)
    models = ResNet101(backbone, 21)
    data = torch.randn(1, 3, 256, 256)
    x = models(data)
    #print(x)
    print(x.size())


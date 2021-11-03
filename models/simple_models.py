import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class GaussianDropout(nn.Module):
    def __init__(self, dropout):
        super(GaussianDropout, self).__init__()

        self.dropout = dropout

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        # N(1, alpha)
        if self.training:
            dropout = F.sigmoid(self.dropout)
            if x.is_cuda:
                epsilon = torch.randn(x.size()).cuda() * (dropout / (1 - dropout)) + 1
            else:
                epsilon = torch.randn(x.size()) * (dropout / (1 - dropout)) + 1
            return x * epsilon
        else:
            '''
            epsilon = torch.randn(x.size()).double() * (model.dropout / (1 - model.dropout)) + 1
            if x.is_cuda:
                epsilon = epsilon.cuda()
            return x * epsilon
            '''
            return x


class BernoulliDropout(nn.Module):
    def __init__(self, dropout):
        super(BernoulliDropout, self).__init__()

        self.dropout = dropout

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        temperature = 0.5
        # N(1, alpha)
        if self.training:
            u = Variable(torch.rand(x.size()))
            if x.is_cuda:
                u = u.cuda()
            z = F.sigmoid(self.dropout) + torch.log(u / (1 - u))
            a = F.sigmoid(z / temperature)
            return x * a
        else:
            return x


class reshape(nn.Module):
    def __init__(self, size):
        super(reshape, self).__init__()
        self.size = size

    def forward(self, x):
        return x.view(-1, self.size)


class SimpleConvNet(nn.Module):
    def __init__(self, batch_norm=True, dropType='bernoulli', conv_drop1=0.0, conv_drop2=0.0, fc_drop=0.0):
        super(SimpleConvNet, self).__init__()

        self.batch_norm = batch_norm

        self.dropType = dropType
        if dropType == 'bernoulli':
            self.conv1_dropout = nn.Dropout(conv_drop1)
            self.conv2_dropout = nn.Dropout(conv_drop2)
            self.fc_dropout = nn.Dropout(fc_drop)
        elif dropType == 'gaussian':
            self.conv1_dropout = GaussianDropout(conv_drop1)
            self.conv2_dropout = GaussianDropout(conv_drop2)
            self.fc_dropout = GaussianDropout(fc_drop)

        if batch_norm:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                self.conv1_dropout,
                nn.MaxPool2d(2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                self.conv2_dropout,
                nn.MaxPool2d(2))
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, padding=2),
                nn.ReLU(),
                self.conv1_dropout,
                nn.MaxPool2d(2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                self.conv2_dropout,
                nn.MaxPool2d(2))

        self.fc = nn.Linear(7*7*32, 10)



    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc_dropout(self.fc(out))
        return out


class CNN(nn.Module):
    def __init__(self, num_layers, dropout, size, weight_decay, in_channel, imsize, do_alexnet=False, num_classes=10,
                 aug_model=None):
        super(CNN, self).__init__()
        self.dropout = Variable(torch.FloatTensor([dropout]), requires_grad=True)
        self.weight_decay = Variable(torch.FloatTensor([weight_decay]), requires_grad=True)
        self.do_alexnet = do_alexnet
        self.num_classes = num_classes
        self.in_channel = in_channel
        self.imsize = imsize
        if self.do_alexnet:
            self.features = nn.Sequential(
                nn.Conv2d(self.in_channel, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )
            if imsize == 32:
                self.view_size = 256 * 2 * 2
            elif imsize == 28:
                self.view_size = 256
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.view_size, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, self.num_classes),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(self.in_channel, 20, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2)
            )
            self.view_size = imsize * imsize * in_channel # More simple definition of view_size
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.view_size, 250),
                nn.ReLU(inplace=True),
                #nn.Dropout(),
                #nn.Linear(250, 250),
                #nn.ReLU(inplace=True),
                nn.Linear(250, self.num_classes),
            )

        self.model_params_keys = dict(self.named_parameters()).keys()
        self.aug_model = aug_model

    def do_train(self):
        self.features.train()
        self.classifier.train()

    def do_eval(self):
        self.features.train()
        self.classifier.train()

    def set_aug_model(self, aug_model):
        self.aug_model = aug_model

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def L2_loss(self):
        loss = 0
        for p in self.parameters():
            loss += torch.sum(torch.mul(p, p))
        return loss * (10 ** self.weight_decay)

    def all_L2_loss(self):
        loss = 0
        count = 0
        for p in self.parameters():
            #val = torch.flatten(p) - self.weight_decay[count: count + p.numel()]
            loss += torch.sum(
                torch.mul(torch.exp(self.weight_decay[count: count + p.numel()]), torch.flatten(torch.mul(p, p))))
            #loss += 1e-3 * torch.sum(torch.mul(val, val))
            count += p.numel()
        return loss

    def model_parameters(self): # To prevent aug_model params are included in model params
        if self.aug_model is None:
            for param in self.parameters():
                yield param
        else:
            for name, param in self.named_parameters():
                if name in self.model_params_keys:
                    yield param

class Net(nn.Module):
    def __init__(self, num_layers, dropout, size, channel, weight_decay, num_classes=10, do_res=False,
                 do_classification=True, aug_model=None):
        super(Net, self).__init__()
        self.dropout = Variable(torch.FloatTensor([dropout]), requires_grad=True)
        self.weight_decay = Variable(torch.FloatTensor([weight_decay]), requires_grad=True)
        self.imsize = size * size * channel
        if not do_classification: self.imsize = size * channel
        self.do_res = do_res
        l_sizes = [self.imsize, self.imsize] + [50] * 20
        network = []
        # self.Gaussian = BernoulliDropout(self.dropout)
        # network.append(nn.Dropout())
        for i in range(num_layers):
            network.append(nn.Linear(l_sizes[i], l_sizes[i + 1]))
            # network.append(self.Gaussian)
            network.append(nn.ReLU())
            #network.append(nn.Dropout())
        network.append(nn.Linear(l_sizes[num_layers], num_classes))
        self.net = nn.Sequential(*network)
        self.model_params_keys = dict(self.named_parameters()).keys()
        self.aug_model = aug_model

    def forward(self, x):
        cur_shape = x.shape
        if not self.do_res:
            return self.net(x.view(-1, self.imsize))# .reshape(cur_shape)
        else:
            res = self.net(x.view(-1, self.imsize)).reshape(cur_shape)
            return x + res

    def do_train(self):
        self.net.train()

    def do_eval(self):
        self.net.eval()

    def set_aug_model(self, aug_model):
        self.aug_model = aug_model

    def L2_loss(self):
        loss = .0
        for p in self.parameters():
            loss = loss + torch.sum(torch.mul(p, p)) * torch.exp(self.weight_decay)
        return loss

    def all_L2_loss(self):
        loss = .0
        count = 0
        for p in self.parameters():
            loss = loss + torch.sum(
                torch.mul(torch.exp(self.weight_decay[count: count + p.numel()]), torch.flatten(torch.mul(p, p))))
            count += p.numel()
        return loss

    def model_parameters(self): # To prevent aug_model params are included in model params
        if self.aug_model is None:
            for param in self.parameters():
                yield param
        else:
            for name, param in self.named_parameters():
                if name in self.model_params_keys:
                    yield param

    def model_named_parameters(self):
        if self.aug_model is None:
            for name, param in self.named_parameters():
                yield name, param
        else:
            for name, param in self.named_parameters():
                if name in self.model_params_keys:
                    yield name, param



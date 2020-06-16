import os
import sys
import pickle
import argparse
from logger import Logger
from itertools import cycle
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd import Variable

from torchvision import datasets, transforms

# Local imports
import data_loaders
from models import models
from kfac import KFACOptimizer
from util import eval_hessian, eval_jacobain, gather_flat_grad, conjugate_gradiant, eval_jacobian_matrix
sys.path.insert(0,'/scratch/gobi1/datasets')


###############################################################################
# Hyperparameters
###############################################################################
parser = argparse.ArgumentParser(description='Training Script')
# Optimization hyperparameters
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--hepochs', type=int, default=15, metavar='HN',
                    help='number of hyperparameter epochs to train (default: 10)')
parser.add_argument('--datasize', type=int, default=10000, metavar='DS',
                    help='train datasize')
parser.add_argument('--valsize', type=int, default=10000, metavar='DS',
                    help='valid datasize')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lrh', type=float, default=0.01, metavar='LRH',
                    help='hyperparameter learning rate (default: 0.01)')
parser.add_argument('--imsize', type=float, default=28, metavar='IMZ',
                    help='image size')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--restart', default=False, type=lambda x: (str(x).lower() == 'true'), help='whether to reset parameter')
parser.add_argument('--jacobian',type=str, default="direct",
                    help='which method to compute jacobian')
parser.add_argument('--hessian',type=str, default="KFAC",
                    help='which method to compute hessian')
parser.add_argument('--model',type=str, default="mlp",
                    help='which model to train')
# Regularization hyperparameters
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout rate')
parser.add_argument('--l2', type=float, default=0,
                    help='l2')
parser.add_argument('--num_layers', type=int, default=0,
                    help='number of layers in network')
parser.add_argument('--input_dropout', type=float, default=0.,
                    help='dropout rate on input')
# Miscellaneous hyperparameters
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=100, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', action='store_true', default=False,
                    help='whether to save current run')
parser.add_argument('--dataset',type=str, default="MNIST",
                    help='which dataset to train' )
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
use_device = 'cuda:0' if args.cuda else 'cpu'

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def half_image_noise(image):
    image[0, 14:] = torch.randn(14, 28)
    return image


if args.dataset == 'MNIST':
    train_loader, val_loader, test_loader = data_loaders.load_mnist(args.batch_size, val_split=True)
    in_channel = 1
    fc_shape = 800
elif args.dataset == 'CIFAR10':
    train_loader, val_loader, test_loader = data_loaders.load_cifar10(args.batch_size, val_split=True)
    in_channel = 3
    fc_shape = 1250


###############################################################################
# Saving
###############################################################################
short_args = {'num_layers': 'l', 'dropout': 'drop', 'input_dropout': 'indrop'}
flags = {}
subdir = "normal_mnist"
files_used = ['mnist/train']

train_labels = ("global_step", "epoch", "batch", "loss")
valid_labels = ("global_step", "loss", "acc")
stats = {"train": train_labels, "valid": valid_labels}
logger = Logger(sys.argv, args, stats)


###############################################################################
# Training
###############################################################################
if args.model == 'mlp':
    model = models.Net(args.num_layers, args.dropout, args.imsize, args.l2)
elif args.model == 'cnn':
    model = models.CNN(args.num_layers, args.dropout, fc_shape, args.l2, in_channel)

model = model.to(use_device)

model.weight_decay = model.weight_decay.to(use_device)
hyper_train = 'weight'

# TODO(PV): Why is the default momentum 0.5? This is non-standard, usually it should be 0.9 with Nesterov, with init lr=0.1
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch, global_step):
    model.net.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(use_device), target.to(use_device)
        data = F.dropout(data, args.input_dropout)

        # Process data and take a step.
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target) + model.L2_loss()
        loss.backward()
        optimizer.step()

        # Occasionally record stats.
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item()))
            step_stats = (global_step, epoch, batch_idx, loss.item())
            logger.write("train", step_stats)

        global_step += 1

    return global_step, loss.data


def test(global_step, test_loader):
    model.net.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(use_device), target.to(use_device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
           test_loss, correct, len(test_loader.dataset),
           100. * correct / len(test_loader.dataset)))
    logger.write("valid", (global_step, test_loss, 100. * correct / (len(test_loader.dataset) * 10)))
    return correct, test_loss


def KFAC_optimize():
    kfac_opt = KFACOptimizer(model, TInv=1)
    model.net.eval()
    model.zero_grad()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        kfac_opt.zero_grad()
        data, target = data.to(use_device), target.to(use_device)
        data = F.dropout(data, args.input_dropout)

        output = model(data)
        loss = F.cross_entropy(output, target, size_average=False) + model.L2_loss()
        
        if batch_idx % 2 == 0:
            loss.backward(retain_graph=True)
            total_loss += loss
        else:
            loss.backward()
        
        kfac_opt.step()

    total_loss /= len(train_loader.dataset) / 2
    if hyper_train in ['weight', 'all_weight']:
        d_loss_d_l = grad(total_loss, model.weight_decay, create_graph=True)
    else:
        d_loss_d_l = grad(total_loss, model.dropout, create_graph=True)

    if args.jacobian == "direct":
        jacobian = eval_jacobain(gather_flat_grad(d_loss_d_l), model, args.cuda).permute(1, 0)
    elif args.jacobian == "product":
        d_loss_d_w = grad(total_loss, model.parameters(), create_graph=True)
        jacobian = torch.ger(gather_flat_grad(d_loss_d_l), gather_flat_grad(d_loss_d_w))

    if args.hessian == "KFAC":
        with torch.no_grad():
            current = 0
            cnt = 0
            for m in model.modules():
                if m.__class__.__name__ in ['Linear', 'Conv2d']:
                    if m.__class__.__name__ == 'Conv2d':
                        size0 = m.weight.size(0)
                        size1 = m.weight.view(m.weight.size(0), -1).size(1)
                    else:
                        size0 = m.weight.size(0)
                        size1 = m.weight.size(1)
                    size = size0 * (size1 + 1 if m.bias is not None else size1)
                    shape = (-1, size0, (size1 + 1 if m.bias is not None else size1))
                    next = current + size
                    jacobians = jacobian[:, current:next].view(shape)
                    d_t_d_l_m = kfac_opt._get_natural_grad(m, jacobians, 0.01)
                    d_theta_d_lambda = d_t_d_l_m.view(d_t_d_l_m.size(0), -1) if cnt == 0 else torch.cat(
                        [d_theta_d_lambda, d_t_d_l_m.view(d_t_d_l_m.size(0), -1)], 1)
                    current = next
                    cnt = 1

    elif args.hessian == "direct":
        d_loss_d_w = grad(total_loss, model.parameters(), create_graph=True)
        hessian = eval_hessian(gather_flat_grad(d_loss_d_w),model, args.cuda)
        inv_hessian = torch.inverse(hessian)
        d_theta_d_lambda = inv_hessian @ jacobian
    
    del kfac_opt,  total_loss, d_loss_d_l
    
    model.zero_grad()
    test_loss = 0
    for i in range(1):
        for data, target in test_loader:
            data, target = data.to(use_device), target.to(use_device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False)  # sum up batch loss
    test_loss /= len(test_loader.dataset)
    test_loss_grad = grad(test_loss, model.parameters())
    grad_vec = gather_flat_grad(test_loss_grad)
    d_loss_d_lambda = d_theta_d_lambda @ grad_vec
    update = args.lrh * d_loss_d_lambda
    update = update.to(use_device)

    if hyper_train in ['weight', 'all_weight']:
        print("weight={}, update={}".format(model.weight_decay.norm(), update.norm()))
        hyper = model.weight_decay - update
    else:
        hyper = model.dropout - update

    model.zero_grad()
    return hyper, i, loss.item(), test_loss.item()


def CG_optimize():
    global model
    # TODO(PV): Why is the conversion to double() and back necessary?
    model = model.double()
    model.dropout = Variable(model.dropout.double(), requires_grad=True)
    if args.cuda:
        model.weight_decay = Variable(model.weight_decay.double(), requires_grad=True).cuda()
    else:
        model.weight_decay = Variable(model.weight_decay.double(), requires_grad=True)

    print(model.parameters)
    model.net.eval()
    model.zero_grad()
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Load data.
        data = data.double()
        data, target = data.to(use_device), target.to(use_device)
        data = F.dropout(data, args.input_dropout)

        output = model(data)
        train_loss += F.cross_entropy(output, target, size_average=False)
    train_loss /= len(train_loader.dataset)
    train_loss = model.all_L2_loss() + train_loss
    train_loss_grad = grad(train_loss, model.parameters(), create_graph=True)
    grad_vec = gather_flat_grad(train_loss_grad).double()
    if hyper_train in ['weight', 'all_weight']:
        d_loss_d_l = grad(train_loss, model.weight_decay, create_graph=True)
    else:
        d_loss_d_l = grad(train_loss, model.dropout, create_graph=True)
    jacobian = eval_jacobain(gather_flat_grad(d_loss_d_l).double(), model, args.cuda).double()
    print(jacobian.size())
    if 0:
        hessian = eval_hessian(grad_vec, model, args.cuda)
    else:
        hessian = None
    d_theta_d_lambda = torch.DoubleTensor(np.zeros((jacobian.size(1), jacobian.size(0))))
    d_theta_d_lambda = d_theta_d_lambda.to(use_device)
    for i in range(jacobian.size(1)):
        con_grad, k = conjugate_gradiant(grad_vec, jacobian[:, i].unsqueeze(0).permute(1,0), model, args.cuda, hessian)
        d_theta_d_lambda[i] = con_grad.view(-1)

    optimizer.zero_grad()
    test_loss = 0
    for i in range(1):
        for data, target in test_loader:
            data = data.double()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False)  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    test_loss_grad = grad(test_loss, model.parameters(), retain_graph=True)
    grad_vec = gather_flat_grad(test_loss_grad)

    d_loss_d_lambda = d_theta_d_lambda @ grad_vec
    update = args.lrh * d_loss_d_lambda
    update = update.to(use_device)

    if hyper_train in ['weight', 'all_weight']:
        print("weight={}, update={}".format(model.weight_decay.norm(), update.norm()))
        hyper = model.weight_decay - update
    else:
        hyper = model.dropout - update

    model = model.float()
    model.dropout = Variable(model.dropout.float(), requires_grad=True)

    if args.cuda:
        model.weight_decay = Variable(model.weight_decay.float(), requires_grad=True).cuda()
    else:
        model.weight_decay = Variable(model.weight_decay.float(), requires_grad=True)

    return hyper, i, train_loss.item(), test_loss.item()


global_step = 0
i = 0
test_loss = 0
train_loss = 0

try:
    for p in range(1):
        if hyper_train == 'weight':
            hp = args.l2 + p * 3.5
        else:
            hp = args.dropout + p * 0.2

        if hp > 10:
            break

        if hyper_train == 'weight':
            model.weight_decay = Variable(torch.FloatTensor([hp]), requires_grad=True)
            model.weight_decay = model.weight_decay.to(use_device)
        elif hyper_train == 'all_weight':
            num_p = sum(p.numel() for p in model.parameters())
            weights = np.ones(num_p) * hp
            model.weight_decay = Variable(torch.FloatTensor(weights), requires_grad=True)
            model.weight_decay = model.weight_decay.to(use_device)
            print(model.weight_decay.size())
        else:
            model.dropout = Variable(torch.FloatTensor([hp]), requires_grad=True)

        directory = "model={}_dataset=_{}_jacob={}_hessian={}_size={}_valsize={}_lrh={}_{}={}_layers={}_restart={}".format(args.model, args.dataset, args.jacobian, args.hessian, args.datasize, args.valsize, args.lrh, hyper_train, hp, args.num_layers,args.restart)

        if not os.path.exists(directory):
            os.mkdir(directory, 0o0755)

        for epoch_h in range(1, args.hepochs + 1):
            if hyper_train == 'weight':
                value = model.weight_decay.item()
            else:
                value = model.dropout.item()
            if (not args.restart) and epoch_h == 2:
                args.epochs = 10

            min_loss = 10000000
            test_loss = 1000
            same_min = 0
            epoch = 0

            while (test_loss - min_loss) / min_loss < 0.1 or epoch < args.epochs:
                epoch += 1
                global_step, train_loss = train(epoch, global_step)
                test_corr, test_loss = test(global_step, test_loader)
                same_min += 1
                if test_loss < min_loss:
                    if np.abs((test_loss - min_loss) / min_loss) > 10e-3:
                        same_min = 0
                        min_loss = test_loss
                if same_min == args.epochs:
                    break

            if np.isnan(test_loss):
                break

            train_correct, train_loss = test(global_step, train_loader)
            torch.save(model, '{}/model_{}_{}_{}_{}_{}_{}_{}_.pkl'.format(directory, epoch_h, test_loss, train_loss, test_corr,
                train_correct, value, i))

            if value == 0:
                break

            hp_k, i, train_loss, test_loss = KFAC_optimize()
            print("hyper parameter={}".format(hp_k))

            if args.restart:
                for layer in model.net.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()

            if hyper_train == 'weight':
                model.weight_decay = Variable(torch.FloatTensor([hp_k.item()]), requires_grad=True)
                model.weight_decay = model.weight_decay.to(use_device)
            elif hyper_train == 'all_weight':
                model.weight_decay = Variable(hp_k.data.float(), requires_grad=True)
            else:
                model.dropout = Variable(torch.FloatTensor([hp_k.item()]), requires_grad=True)
except KeyboardInterrupt:
    print('=' * 80)
    print('Exiting training early...')
    print('=' * 80)


# TODO(PV): Final evaluation on the test set!


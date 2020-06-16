from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pdb
import os
import sys
sys.path.insert(0, os.path.abspath(".."))
from logger import Logger


###############################################################################
# Hyperparameters
###############################################################################
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
# Optimization hyperparameters
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
# Regularization hyperparameters
parser.add_argument('--dropout', type=float, default=0.,
                    help='dropout rate')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers in network')
parser.add_argument('--input_dropout', type=float, default=0.,
                    help='dropout rate on input')
# Miscellaneous hyperparameters
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', action='store_true', default=False,
                    help='whether to save current run')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('~/data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('~/data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

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


###############################################################################
# Model definitions 
###############################################################################
class Net(nn.Module):
    def __init__(self, num_layers, dropout):
        super(Net, self).__init__()
        self.dropout = dropout
        l_sizes = [784, 250] + [100]*20
        self.layers = [nn.Linear(l_sizes[i], l_sizes[i+1]) for i in range(num_layers)]
        self.layers = nn.ModuleList(self.layers)
        self.fc = nn.Linear(l_sizes[num_layers], 10)

    def forward(self, x):
        x = x.view(-1, 784)
        for layer in self.layers:
            x = F.relu(F.dropout(layer(x), self.dropout))
        x = self.fc(x)
        return x

###############################################################################
# Training
###############################################################################
model = Net(args.num_layers, args.dropout)
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch, global_step):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Load data.
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        data = F.dropout(data, args.input_dropout)

        # Process data and take a step.
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # Occasionally record stats.
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
            step_stats = (global_step, epoch, batch_idx, loss.data[0])
        global_step += 1

    return global_step


def test(global_step):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

global_step = 0 
for epoch in range(1, args.epochs + 1):
    global_step = train(epoch, global_step)
    test(global_step)

# python train_checkpoint.py --dataset cifar10 --model resnet18 --data_augmentation
# python train_checkpoint.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16
import os
import ipdb
import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision import datasets, transforms

# Local imports
import data_loaders
from cutout import Cutout
from models.resnet import ResNet18
from models.wide_resnet import WideResNet
from utils.csv_logger import CSVLogger

model_options = ['resnet18', 'wideresnet']
dataset_options = ['cifar10', 'cifar100']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--optimizer', type=str, default='sgdm',
                    help='Choose an optimizer')
parser.add_argument('--wdecay', type=float, default=5e-4,
                    help='Amount of weight decay')
parser.add_argument('--data_augmentation', action='store_true', default=True,
                    help='augment data by flipping and cropping')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')
parser.add_argument('--save_dir', type=str, default='baseline_checkpoints',
                    help='Base save directory')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

test_id = '{}_{}_{}_lr{}_wd{}_aug{}'.format(args.dataset, args.model, args.optimizer, args.lr, args.wdecay,
                                            int(args.data_augmentation))

print(args)

# Image Preprocessing
normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])
if args.data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
if args.cutout:
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

test_transform = transforms.Compose([transforms.ToTensor(), normalize])

if args.dataset == 'cifar10':
    num_classes = 10
    train_loader, val_loader, test_loader = data_loaders.load_cifar10(args.batch_size, val_split=True,
                                                                      augmentation=args.data_augmentation)
elif args.dataset == 'cifar100':
    num_classes = 100
    train_loader, val_loader, test_loader = data_loaders.load_cifar100(args.batch_size, val_split=True,
                                                                       augmentation=args.data_augmentation)

if args.model == 'resnet18':
    cnn = ResNet18(num_classes=num_classes)
elif args.model == 'wideresnet':
    cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)

cnn = cnn.cuda()
criterion = nn.CrossEntropyLoss().cuda()

if args.optimizer == 'sgdm':
    cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wdecay)
elif args.optimizer == 'sgd':
    cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adam':
    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr,
                                     weight_decay=args.wdecay)  # TODO(PV): The PyTorch implementation of wdecay is not correct for Adam/RMSprop
elif args.optimizer == 'rmsprop':
    cnn_optimizer = torch.optim.RMSprop(cnn.parameters(), lr=args.lr,
                                        weight_decay=args.wdecay)  # TODO(PV): The PyTorch implementation of wdecay is not correct for Adam/RMSprop

scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

filename = os.path.join(args.save_dir, test_id + '.csv')
csv_logger = CSVLogger(fieldnames=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc'],
                       filename=filename)


def test(loader):
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    losses = []
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)

        xentropy_loss = criterion(pred, labels)
        losses.append(xentropy_loss.item())

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    avg_loss = float(np.mean(losses))
    acc = correct / total
    cnn.train()
    return avg_loss, acc


def saver(epoch, model, optimizer, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


try:
    for epoch in range(args.epochs):

        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.

        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            images = images.cuda()
            labels = labels.cuda()

            cnn.zero_grad()
            pred = cnn(images)

            xentropy_loss = criterion(pred, labels)
            xentropy_loss.backward()
            cnn_optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total

            progress_bar.set_postfix(
                xentropy='%.5f' % (xentropy_loss_avg / (i + 1)),
                acc='%.3f' % accuracy)

        val_loss, val_acc = test(val_loader)
        test_loss, test_acc = test(test_loader)
        tqdm.write('val loss: {:6.4f} | val acc: {:6.4f} | test loss: {:6.4f} | test_acc: {:6.4f}'.format(
            val_loss, val_acc, test_loss, test_acc))

        scheduler.step(epoch)

        row = {'epoch': str(epoch),
               'train_loss': str(xentropy_loss_avg / (i + 1)), 'train_acc': str(accuracy),
               'val_loss': str(val_loss), 'val_acc': str(val_acc),
               'test_loss': str(test_loss), 'test_acc': str(test_acc)}
        csv_logger.writerow(row)
        if epoch % 10 == 0:
            saver(epoch, cnn, cnn_optimizer, os.path.join(args.save_dir, test_id + '.pt'))
except KeyboardInterrupt:
    progress_bar.close()
    print('=' * 80)
    print('Exiting training early...')
    print('=' * 80)

saver(epoch, cnn, cnn_optimizer, os.path.join(args.save_dir, test_id + '.pt'))
csv_logger.close()

import os
import ipdb
import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

# Local imports
import data_loaders
from csv_logger import CSVLogger


parser = argparse.ArgumentParser(description='CNN Hyperparameter Fine-tuning')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'],
                    help='Choose a dataset')
parser.add_argument('--model', default='resnet18', choices=['resnet18', 'wideresnet'],
                    help='Choose a model')
parser.add_argument('--num_finetune_epochs', type=int, default=10,
                    help='Number of fine-tuning epochs')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Learning rate')
parser.add_argument('--optimizer', type=str, default='sgdm',
                    help='Choose an optimizer')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Mini-batch size')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='Whether to use data augmentation')
parser.add_argument('--wdecay', type=float, default=5e-4,
                    help='Amount of weight decay')
parser.add_argument('--load_checkpoint', type=str,
                    help='Path to pre-trained checkpoint to load and finetune')
parser.add_argument('--save_dir', type=str, default='finetuned_checkpoints',
                    help='Save directory for the fine-tuned checkpoint')
args = parser.parse_args()


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

test_id = '{}_{}_{}_lr{}_wd{}_aug{}'.format(args.dataset, args.model, args.optimizer, args.lr, args.wdecay, int(args.data_augmentation))
filename = os.path.join(args.save_dir, test_id + '.csv')
csv_logger = CSVLogger(fieldnames=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc'],
                       filename=filename)

model = torch.load(args.load_checkpoint)


# TODO(PV): Load saved optimizer from the training run
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wdecay)

# Set random regularization hyperparameters
# data_augmentation_hparams = {}  # Random values for hue, saturation, brightness, contrast, rotation, etc.
if args.dataset == 'cifar10':
    num_classes = 10
    train_loader, val_loader, test_loader = data_loaders.load_cifar10(args.batch_size, val_split=True, augmentation=args.data_augmentation)
elif args.dataset == 'cifar100':
    num_classes = 100
    train_loader, val_loader, test_loader = data_loaders.load_cifar100(args.batch_size, val_split=True, augmentation=args.data_augmentation)


def test(loader):
    model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    losses = []
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = model(images)

        xentropy_loss = F.cross_entropy(pred, labels)
        losses.append(xentropy_loss.item())

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    avg_loss = float(np.mean(losses))
    acc = correct / total
    model.train()
    return avg_loss, acc

for epoch in range(args.num_finetune_epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Finetune Epoch ' + str(epoch))

        images, labels = images.cuda(), labels.cuda()

        pred = model(images)
        xentropy_loss = F.cross_entropy(pred, labels)
        optimizer.zero_grad()
        xentropy_loss.backward()
        optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    val_loss, val_acc = test(val_loader)
    test_loss, test_acc = test(test_loader)
    tqdm.write('val loss: {:6.4f} | val acc: {:6.4f} | test loss: {:6.4f} | test_acc: {:6.4f}'.format(
                val_loss, val_acc, test_loss, test_acc))

    # scheduler.step(epoch)

    row = {'epoch': str(epoch),
           'train_loss': str(xentropy_loss_avg / (i+1)), 'train_acc': str(accuracy),
           'val_loss': str(val_loss), 'val_acc': str(val_acc),
           'test_loss': str(test_loss), 'test_acc': str(test_acc) }
    csv_logger.writerow(row)

import os
import sys
import csv
import ipdb
import time
import random
import datetime
import argparse
import numpy as np
from tqdm import tqdm

# YAML setup
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

# Local imports
import data_loaders
from csv_logger import CSVLogger
from models import wide_resnet, resnet_cifar, models


def cnn_val_loss(config={}, reporter=None, callback=None, return_all=False):
    print("Starting cnn_val_loss...")

    ###############################################################################
    # Arguments
    ###############################################################################
    dataset_options = ['cifar10', 'cifar100', 'fashion']

    ## Tuning parameters: all of the dropouts
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--dataset', default='cifar10', choices=dataset_options,
                        help='Choose a dataset (cifar10, cifar100)')
    parser.add_argument('--model', default='resnet32', choices=['resnet32', 'wideresnet', 'simpleconvnet'],
                        help='Choose a model (resnet32, wideresnet, simpleconvnet)')

    #### Optimization hyperparameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=int(config['epochs']),
                        help='Number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=float(config['lr']),
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=float(config['momentum']),
                        help='Nesterov momentum')
    parser.add_argument('--lr_decay', type=float, default=float(config['lr_decay']),
                        help='Factor by which to multiply the learning rate.')

    # parser.add_argument('--weight_decay', type=float, default=float(config['weight_decay']),
    #                     help='Amount of weight decay to use.')
    # parser.add_argument('--dropout', type=float, default=config['dropout'] if 'dropout' in config else 0.0,
    #                     help='Amount of dropout for wideresnet')
    # parser.add_argument('--dropout1', type=float, default=config['dropout1'] if 'dropout1' in config else -1,
    #                     help='Amount of dropout for wideresnet')
    # parser.add_argument('--dropout2', type=float, default=config['dropout2'] if 'dropout2' in config else -1,
    #                     help='Amount of dropout for wideresnet')
    # parser.add_argument('--dropout3', type=float, default=config['dropout3'] if 'dropout3' in config else -1,
    #                     help='Amount of dropout for wideresnet')
    parser.add_argument('--dropout_type', type=str, default=config['dropout_type'],
                        help='Type of dropout (bernoulli or gaussian)')

    # Data augmentation hyperparameters
    parser.add_argument('--inscale', type=float, default=0 if 'inscale' not in config else config['inscale'],
                        help='defines input scaling factor')
    parser.add_argument('--hue', type=float, default=0. if 'hue' not in config else config['hue'],
                        help='hue jitter rate')
    parser.add_argument('--brightness', type=float, default=0. if 'brightness' not in config else config['brightness'],
                        help='brightness jitter rate')
    parser.add_argument('--saturation', type=float, default=0. if 'saturation' not in config else config['saturation'],
                        help='saturation jitter rate')
    parser.add_argument('--contrast', type=float, default=0. if 'contrast' not in config else config['contrast'],
                        help='contrast jitter rate')

    # Weight decay and dropout hyperparameters for each layer
    parser.add_argument('--weight_decays', type=str, default='0.0',
                        help='Amount of weight decay to use for each layer, represented as a comma-separated string of floats.')
    parser.add_argument('--dropouts', type=str, default='0.0',
                        help='Dropout rates for each layer, represented as a comma-separated string of floats')

    parser.add_argument('--nonmono', '-nonm', type=int, default=60,
                        help='how many previous epochs to consider for nonmonotonic criterion')
    parser.add_argument('--patience', type=int, default=75,
                        help='How long to wait for the val loss to improve before early stopping.')

    parser.add_argument('--data_augmentation', action='store_true', default=config['data_augmentation'],
                        help='Augment data by cropping and horizontal flipping')

    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many steps before logging stats from training set')
    parser.add_argument('--valid_log_interval', type=int, default=50,
                        help='how many steps before logging stats from validations set')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--save', action='store_true', default=False,
                        help='whether to save current run')
    parser.add_argument('--seed', type=int, default=11,
                        help='random seed (default: 11)')
    parser.add_argument('--save_dir', default=config['save_dir'],
                        help='subdirectory of logdir/savedir to save in (default changes to date/time)')

    args, unknown = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    cudnn.benchmark = True  # Should make training should go faster for large models

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print(args)
    sys.stdout.flush()

    # args.dropout1 = args.dropout1 if args.dropout1 != -1 else args.dropout
    # args.dropout2 = args.dropout2 if args.dropout2 != -1 else args.dropout
    # args.dropout3 = args.dropout3 if args.dropout3 != -1 else args.dropout

    ###############################################################################
    # Saving
    ###############################################################################
    timestamp = '{:%Y-%m-%d}'.format(datetime.datetime.now())
    random_hash = random.getrandbits(16)
    exp_name = '{}-dset:{}-model:{}-seed:{}-hash:{}'.format(
        timestamp, args.dataset, args.model, args.seed if args.seed else 'None', random_hash)

    dropout_rates = [float(value) for value in args.dropouts.split(',')]
    weight_decays = [float(value) for value in args.weight_decays.split(',')]

    # Create log folder
    BASE_SAVE_DIR = 'experiments'
    save_dir = os.path.join(BASE_SAVE_DIR, args.save_dir, exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Check whether the result.csv file exists already
    if os.path.exists(os.path.join(save_dir, 'result.csv')):
        if not args.overwrite:
            print('The result file {} exists! Run with --overwrite to overwrite this experiment.'.format(
                os.path.join(save_dir, 'result.csv')))
            sys.exit(0)

    # Save command-line arguments
    with open(os.path.join(save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    epoch_csv_logger = CSVLogger(fieldnames=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'],
                                 filename=os.path.join(save_dir, 'epoch_log.csv'))

    ###############################################################################
    # Data Loading/Model/Optimizer
    ###############################################################################

    if args.dataset == 'cifar10':
        train_loader, valid_loader, test_loader = data_loaders.load_cifar10(args, args.batch_size, val_split=True,
                                                                            augmentation=args.data_augmentation)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_loader, valid_loader, test_loader = data_loaders.load_cifar100(args, args.batch_size, val_split=True,
                                                                             augmentation=args.data_augmentation)
        num_classes = 100
    elif args.dataset == 'fashion':
        train_loader, valid_loader, test_loader = data_loaders.load_fashion_mnist(args.batch_size, val_split=True)
        num_classes = 10

    if args.model == 'resnet32':
        cnn = resnet_cifar.resnet32(dropRates=dropout_rates)
    elif args.model == 'wideresnet':
        cnn = wide_resnet.WideResNet(depth=16,
                                     num_classes=num_classes,
                                     widen_factor=8,
                                     dropRates=dropout_rates,
                                     dropType=args.dropout_type)
        # cnn = wide_resnet.WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=args.dropout)
    elif args.model == 'simpleconvnet':
        cnn = models.SimpleConvNet(dropType=args.dropout_type,
                                   conv_drop1=args.dropout1,
                                   conv_drop2=args.dropout2,
                                   fc_drop=args.dropout3)

    def optim_parameters(model):
        module_list = [m for m in model.modules() if type(m) == nn.Linear or type(m) == nn.Conv2d]
        weight_decays = [1e-4] * len(module_list)
        return [{'params': layer.parameters(), 'weight_decay': wdecay} for (layer, wdecay) in
                zip(module_list, weight_decays)]

    cnn = cnn.to(device)
    criterion = nn.CrossEntropyLoss()
    # cnn_optimizer = torch.optim.SGD(cnn.parameters(),
    #                                 lr=args.lr,
    #                                 momentum=args.momentum,
    #                                 nesterov=True,
    #                                 weight_decay=args.weight_decay)
    cnn_optimizer = torch.optim.SGD(optim_parameters(cnn),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    nesterov=True)

    ###############################################################################
    # Training/Evaluation
    ###############################################################################
    def evaluate(loader):
        """Returns the loss and accuracy on the entire validation/test set."""
        cnn.eval()
        correct = total = loss = 0.
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                pred = cnn(images)
                loss += F.cross_entropy(pred, labels, reduction='sum').item()
                hard_pred = torch.max(pred, 1)[1]
                total += labels.size(0)
                correct += (hard_pred == labels).sum().item()

        accuracy = correct / total
        mean_loss = loss / total
        cnn.train()
        return mean_loss, accuracy

    epoch = 1
    global_step = 0
    patience_elapsed = 0
    stored_loss = 1e8
    best_val_loss = []
    start_time = time.time()

    # This is based on the schedule used for WideResNets. The gamma (decay factor) can also be 0.2 (= 5x decay)
    # Right now we're not using the scheduler because we use nonmonotonic lr decay (based on validation performance)
    # scheduler = MultiStepLR(cnn_optimizer, milestones=[60,120,160], gamma=args.lr_decay)

    while epoch < args.epochs + 1 and patience_elapsed < args.patience:

        running_xentropy = correct = total = 0.

        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))
            images, labels = images.to(device), labels.to(device)

            if args.inscale > 0:
                noise = torch.rand(images.size(0), device=device)
                scaled_noise = ((1 + args.inscale) - (1 / (1 + args.inscale))) * noise + (1 / (1 + args.inscale))
                images = images * scaled_noise[:, None, None, None]

            # images = F.dropout(images, p=args.indropout, training=True)  # TODO: Incorporate input dropout
            cnn.zero_grad()
            pred = cnn(images)

            xentropy_loss = criterion(pred, labels)
            xentropy_loss.backward()
            cnn_optimizer.step()

            running_xentropy += xentropy_loss.item()

            # Calculate running average of accuracy
            _, hard_pred = torch.max(pred, 1)
            total += labels.size(0)
            correct += (hard_pred == labels).sum().item()
            accuracy = correct / float(total)

            global_step += 1
            progress_bar.set_postfix(xentropy='%.3f' % (running_xentropy / (i + 1)),
                                     acc='%.3f' % accuracy,
                                     lr='%.3e' % cnn_optimizer.param_groups[0]['lr'])

        val_loss, val_acc = evaluate(valid_loader)
        print('Val loss: {:6.4f} | Val acc: {:6.4f}'.format(val_loss, val_acc))
        sys.stdout.flush()
        stats = {'global_step': global_step, 'time': time.time() - start_time, 'loss': val_loss, 'acc': val_acc}
        # logger.write('valid', stats)

        if (len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
            cnn_optimizer.param_groups[0]['lr'] *= args.lr_decay
            print('Decaying the learning rate to {}'.format(cnn_optimizer.param_groups[0]['lr']))
            sys.stdout.flush()

        if val_loss < stored_loss:
            with open(os.path.join(save_dir, 'best_checkpoint.pt'), 'wb') as f:
                torch.save(cnn.state_dict(), f)
            print('Saving model (new best validation)')
            sys.stdout.flush()
            stored_loss = val_loss
            patience_elapsed = 0
        else:
            patience_elapsed += 1

        best_val_loss.append(val_loss)

        # scheduler.step(epoch)

        avg_xentropy = running_xentropy / (i + 1)
        train_acc = correct / float(total)

        if callback is not None:
            callback(epoch, avg_xentropy, train_acc, val_loss, val_acc, config)

        if reporter is not None:
            reporter(timesteps_total=epoch, mean_loss=val_loss)

        if cnn_optimizer.param_groups[0]['lr'] < 1e-7:  # Another stopping criterion based on decaying the lr
            break

        epoch += 1

        epoch_row = {'epoch': str(epoch), 'train_loss': avg_xentropy, 'train_acc': str(train_acc),
                     'val_loss': str(val_loss), 'val_acc': str(val_acc)}
        epoch_csv_logger.writerow(epoch_row)

    # Load best model and run on test
    with open(os.path.join(save_dir, 'best_checkpoint.pt'), 'rb') as f:
        cnn.load_state_dict(torch.load(f))

    train_loss = avg_xentropy
    train_acc = correct / float(total)

    # Run on val and test data.
    val_loss, val_acc = evaluate(valid_loader)
    test_loss, test_acc = evaluate(test_loader)

    print('=' * 89)
    print(
        '| End of training | trn loss: {:8.5f} | trn acc {:8.5f} | val loss {:8.5f} | val acc {:8.5f} | test loss {:8.5f} | test acc {:8.5f}'.format(
            train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))
    print('=' * 89)
    sys.stdout.flush()

    # Save the final val and test performance to a results CSV file
    with open(os.path.join(save_dir, 'result_{}.csv'.format(time.time())), 'w') as result_file:
        result_writer = csv.DictWriter(result_file,
                                       fieldnames=['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss',
                                                   'test_acc'])
        result_writer.writeheader()
        result_writer.writerow({'train_loss': train_loss,
                                'train_acc': train_acc,
                                'val_loss': val_loss, 'val_acc': val_acc,
                                'test_loss': test_loss, 'test_acc': test_acc})
        result_file.flush()

    if return_all:
        print("RETURNING ", train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)
        sys.stdout.flush()
        return train_loss, train_acc, val_loss, val_acc, test_loss, test_acc
    else:
        print("RETURNING ", stored_loss)
        sys.stdout.flush()
        return stored_loss


if __name__ == '__main__':

    dataset_options = ['cifar10', 'cifar100']

    ## Tuning parameters: all of the dropouts
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--dataset', default='cifar10', choices=dataset_options,
                        help='Choose a dataset (cifar10, cifar100)')
    parser.add_argument('--model', default='resnet32', choices=['resnet32', 'wideresnet'],
                        help='Choose a model (resnet32, wideresnet)')

    #### Optimization hyperparameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Nesterov momentum')
    parser.add_argument('--lr_decay', type=float, default=0.2,
                        help='Factor by which to multiply the learning rate.')

    # Weight decay and dropout hyperparameters for each layer
    # parser.add_argument('--weight_decays', type=str, default='0.0',
    #                     help='Amount of weight decay to use for each layer, represented as a comma-separated string of floats.')

    for i in range(32):  # For now, hard-coded loop over 32 per-layer weight decay values
        parser.add_argument('--weight_decay_{}'.format(i), type=float, default=0.0,
                            help='Amount of weight decay for layer {}'.format(i))

    # Data augmentation hyperparameters
    parser.add_argument('--inscale', type=float, default=0.,
                        help='defines input scaling factor')
    parser.add_argument('--hue', type=float, default=0.,
                        help='hue jitter rate')
    parser.add_argument('--brightness', type=float, default=0.,
                        help='brightness jitter rate')
    parser.add_argument('--saturation', type=float, default=0.,
                        help='saturation jitter rate')
    parser.add_argument('--contrast', type=float, default=0.,
                        help='contrast jitter rate')

    # Dropout hyperparameters
    parser.add_argument('--dropouts', type=str, default='0.0',
                        help='Dropout rates for each layer, represented as a comma-separated string of floats')
    parser.add_argument('--dropout_type', type=str, default='bernoulli',
                        help='Type of dropout (bernoulli or gaussian)')

    parser.add_argument('--nonmono', type=int, default=60,
                        help='how many previous epochs to consider for the nonmonotonic decay criterion')
    parser.add_argument('--patience', type=int, default=75,
                        help='How long to wait for the val loss to improve before early stopping.')

    parser.add_argument('--data_augmentation', action='store_true', default=False,
                        help='Augment data by cropping and horizontal flipping')

    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many steps before logging stats from training set')
    parser.add_argument('--valid_log_interval', type=int, default=50,
                        help='how many steps before logging stats from validations set')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--save', action='store_true', default=False,
                        help='whether to save current run')
    parser.add_argument('--seed', type=int, default=11,
                        help='random seed (default: 11)')
    parser.add_argument('--save_dir', default='saves',
                        help='subdirectory of logdir/savedir to save in (default changes to date/time)')

    args, unknown = parser.parse_known_args()
    config = vars(args)
    cnn_val_loss(config=config, reporter=None, callback=None)

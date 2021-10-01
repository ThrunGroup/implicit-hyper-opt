import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd import Variable

# Local imports
from data_loaders import load_mnist
from models.simple_models import Net
from utils.util import gather_flat_grad
from utils.csv_logger import CSVLogger
from ruamel.yaml import YAML

sys.path.insert(0, '/scratch/gobi1/datasets')


def experiment(args):
    print(f"Running experiment with args: {args}")
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Do this since
    args.train_batch_num -= 1
    args.val_batch_num -= 1
    args.eval_batch_num -= 1

    # TODO (JON): What is yaml for right now?
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.boolean_representation = ['False', 'True']

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ###############################################################################
    # Setup dataset
    ###############################################################################
    num_train = args.datasize
    train_loader, val_loader, test_loader = load_mnist(args.batch_size,
                                                       subset=[args.datasize, args.valsize, args.testsize],
                                                       num_train=num_train)
    in_channel = 1
    imsize = 28
    num_classes = 10

    ###############################################################################
    # Setup model
    ###############################################################################
    model = Net(args.num_layers, args.dropout, imsize, in_channel, args.l2, num_classes=num_classes)

    def init_hyper_train():
        init_hyper = args.l2
        num_p = sum(p.numel() for p in model.parameters())
        weights = np.ones(num_p) * init_hyper
        model.weight_decay = Variable(torch.FloatTensor(weights), requires_grad=True)
        return init_hyper

    def get_hyper_train():
        return model.weight_decay

    hyper = init_hyper_train()  # We need this when doing all_weight

    ###############################################################################
    # Setup Optimizer
    ###############################################################################
    init_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # , momentum=0.9)

    hyper_optimizer = torch.optim.RMSprop([get_hyper_train()])  # , lr=args.lrh)  # try 0.1 as lr

    ###############################################################################
    # Setup Saving
    ###############################################################################
    directory_name_vals = {'model': args.model, 'lrh': 0.1, 'jacob': args.jacobian,
                           'hessian': args.hessian, 'size': args.datasize, 'valsize': args.valsize,
                           'dataset': args.dataset, 'hyper_train': args.hyper_train, 'layers': args.num_layers,
                           'restart': args.restart, 'hyper_value': hyper}
    directory = ""
    for key, val in directory_name_vals.items():
        directory += f"{key}={val}_"

    if not os.path.exists(directory):
        os.mkdir(directory, 0o0755)

    # Save command-line arguments
    with open(os.path.join(directory, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    epoch_h_csv_logger = CSVLogger(
        fieldnames=['epoch_h', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc', 'hyper_param',
                    'hp_update'],
        filename=os.path.join(directory, 'epoch_h_log.csv'))

    ###############################################################################
    # Setup Training
    ###############################################################################
    def train_loss_func(x, y, network, reduction='elementwise_mean'):
        predicted_y = network(x)
        reg_loss = network.all_L2_loss()
        return F.cross_entropy(predicted_y, y, reduction=reduction) + reg_loss, predicted_y

    def val_loss_func(x, y, network, reduction='elementwise_mean'):
        predicted_y = network(x)
        loss = F.cross_entropy(predicted_y, y, reduction=reduction)
        regularizer = 0
        return loss + regularizer, predicted_y

    def test_loss_func(x, y, network, reduction='elementwise_mean'):
        return val_loss_func(x, y, network, reduction=reduction)

    def prepare_data(x, y):
        return Variable(x), Variable(y)

    def batch_loss(x, y, network, loss_func, reduction='elementwise_mean'):
        loss, predicted_y = loss_func(x, y, network, reduction=reduction)
        return loss, predicted_y

    def train(elementary_epoch, step):
        model.train()
        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            # Take a gradient step for this mini-batch
            optimizer.zero_grad()
            x, y = prepare_data(x, y)
            loss, _ = batch_loss(x, y, model, train_loss_func)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1
            if batch_idx >= args.train_batch_num: break

        # Occasionally record stats.
        if epoch % args.elementary_log_interval == 0:
            print(f'Train Epoch: {elementary_epoch} \tLoss: {total_loss:.6f}')

        return step, total_loss / (batch_idx + 1)

    def evaluate(step, data_loader, name=None):
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            model.eval()

            for batch_idx, (x, y) in enumerate(data_loader):
                x, y = prepare_data(x, y)
                loss, predicted_y = batch_loss(x, y, model, test_loss_func)
                total_loss += loss.item()

                pred = predicted_y.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(y.data.view_as(pred)).cpu().sum()
                if batch_idx >= args.eval_batch_num:
                    break

            total_loss /= (batch_idx + 1)

        data_size = args.batch_size * (batch_idx + 1)
        acc = float(correct) / data_size
        print(f'Evaluate {name}, {step}: Average loss: {total_loss:.4f}, Accuracy: {correct}/{data_size} ({acc * 100}%)')
        return acc, total_loss

    ###############################################################################
    # Setup Inversion Algorithms
    ###############################################################################

    def KFAC_optimize():
        # set up placeholder for the partial derivative in each batch
        total_d_val_loss_d_lambda = torch.zeros(get_hyper_train().size(0))

        num_weights = sum(p.numel() for p in model.parameters())
        d_val_loss_d_theta = torch.zeros(num_weights)
        model.train()
        for batch_idx, (x, y) in enumerate(val_loader):
            model.zero_grad()
            x, y = prepare_data(x, y)
            val_loss, _ = batch_loss(x, y, model, val_loss_func)
            val_loss_grad = grad(val_loss, model.parameters())
            d_val_loss_d_theta += gather_flat_grad(val_loss_grad)
            if batch_idx >= args.val_batch_num:
                break

        d_val_loss_d_theta /= (batch_idx + 1)

        # get d theta / d lambda
        pre_conditioner = d_val_loss_d_theta
        flat_pre_conditioner = pre_conditioner  # 2*pre_conditioner - args.lr*hessian_term

        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = prepare_data(x, y)
            train_loss, _ = batch_loss(x, y, model, train_loss_func)

            model.zero_grad(), hyper_optimizer.zero_grad()
            d_train_loss_d_theta = grad(train_loss, model.parameters(), create_graph=True)
            flat_d_train_loss_d_theta = gather_flat_grad(d_train_loss_d_theta)

            model.zero_grad(), hyper_optimizer.zero_grad()
            flat_d_train_loss_d_theta.backward(flat_pre_conditioner)
            if get_hyper_train().grad is not None:
                total_d_val_loss_d_lambda -= get_hyper_train().grad

            if batch_idx >= args.train_batch_num:
                break

        total_d_val_loss_d_lambda /= (batch_idx + 1)

        direct_d_val_loss_d_lambda = torch.zeros(get_hyper_train().size(0))

        model.train()
        for batch_idx, (x_val, y_val) in enumerate(val_loader):
            model.zero_grad(), hyper_optimizer.zero_grad()
            x_val, y_val = prepare_data(x_val, y_val)
            val_loss, _ = batch_loss(x_val, y_val, model, val_loss_func)
            val_loss_grad = grad(val_loss, get_hyper_train(), allow_unused=True)

            if val_loss_grad is not None and val_loss_grad[0] is not None:
                direct_d_val_loss_d_lambda += gather_flat_grad(val_loss_grad)
            else:
                break

            if batch_idx >= args.val_batch_num:
                break

        direct_d_val_loss_d_lambda /= (batch_idx + 1)

        get_hyper_train().grad = direct_d_val_loss_d_lambda + total_d_val_loss_d_lambda
        print("weight={}, update={}".format(get_hyper_train().norm(), get_hyper_train().grad.norm()))

        hyper_optimizer.step()
        model.zero_grad(), hyper_optimizer.zero_grad()
        return get_hyper_train(), get_hyper_train().grad


    ###############################################################################
    # Perform the training
    ###############################################################################
    global_step = 0

    hp_k, update = 0, 0
    for epoch_h in range(0, args.hepochs + 1):
        print(f"Hyper epoch: {epoch_h}")
        epoch_csv_logger = CSVLogger(fieldnames=['epoch', 'train_loss', 'val_loss', 'val_acc'],
                                     filename=os.path.join(directory, f'epoch_log_{epoch_h}.csv'))

        if (epoch_h) % args.hyper_log_interval == 0:
            eval_train_corr, eval_train_loss = evaluate(global_step, train_loader, 'train')
            eval_val_corr, eval_val_loss = evaluate(epoch_h, val_loader, 'valid')
            eval_test_corr, eval_test_loss = evaluate(epoch_h, test_loader, 'test')
            epoch_row = {'hyper_param': str(hp_k), 'train_loss': eval_train_loss,
                         'train_acc': str(eval_train_corr),
                         'val_loss': str(eval_val_loss), 'val_acc': str(eval_val_corr),
                         'test_loss': str(eval_test_loss), 'test_acc': str(eval_test_corr),
                         'epoch_h': str(epoch_h),
                         'hp_update': str(update)}
            epoch_h_csv_logger.writerow(epoch_row)
            if args.break_perfect_val and eval_val_corr >= 0.999 and eval_train_corr >= 0.999:
                break

        elementary_epochs = args.epochs
        if epoch_h == 0:
            elementary_epochs = args.init_epochs

        optimizer = init_optimizer

        for epoch in range(1, elementary_epochs + 1):
            global_step, epoch_train_loss = train(epoch, global_step)

            epoch_row = {'epoch': str(epoch), 'train_loss': epoch_train_loss}
            epoch_csv_logger.writerow(epoch_row)

            if np.isnan(epoch_train_loss):
                print("Loss is nan, stop the loop")
                break

        hp_k, update = KFAC_optimize()


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default="MNIST", choices=['MNIST'],
                        help='which dataset to train')
    opt_data_num = 50  # Temporary variable to testing
    parser.add_argument('--datasize', type=int, default=opt_data_num, metavar='DS',
                        help='train datasize')
    val_multiple = 3
    parser.add_argument('--valsize', type=int, default=50, metavar='DS',
                        help='valid datasize')
    parser.add_argument('--testsize', type=int, default=50, metavar='DS',
                        help='test datasize')

    # Optimization hyperparameters
    parser.add_argument('--batch_size', type=int, default=opt_data_num, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=250, metavar='N',
                        help='input batch size for testing (default: 1000)')
    elementary_epochs = 5
    parser.add_argument('--epochs', type=int, default=elementary_epochs, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--init_epochs', type=int, default=1, metavar='N',
                        help='number of initial epochs to train (default: 10)')
    parser.add_argument('--hepochs', type=int, default=100000, metavar='HN',
                        help='number of hyperparameter epochs to train (default: 10)')
    parser.add_argument('--train_batch_num', type=int, default=1, metavar='HN',  # 128 full pass
                        help='num of validation batches')
    parser.add_argument('--val_batch_num', type=int, default=1, metavar='HN',
                        help='num of validation batches')
    parser.add_argument('--eval_batch_num', type=int, default=100, metavar='HN',
                        help='num of validation batches')

    # Optimizer Parameters
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--lrh', type=float, default=0.1, metavar='LRH',
                        help='hyperparameter learning rate (default: 0.01)')

    # Non-optimization hyperparameters
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout rate')
    parser.add_argument('--l2', type=float, default=-4,  # -4 worked for resnet
                        help='l2')
    parser.add_argument('--input_dropout', type=float, default=0.0,
                        help='dropout rate on input')

    # Architectural hyperparameters
    parser.add_argument('--model', type=str, default="mlp", choices=['mlp'],
                        help='which model to train')
    parser.add_argument('--num_layers', type=int, default=0,
                        help='number of layers in network')

    # IFT algorithm choices
    parser.add_argument('--restart', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to reset parameter')
    parser.add_argument('--jacobian', type=str, default="direct", choices=['direct', 'product'],
                        help='which method to compute jacobian')
    parser.add_argument('--hessian', type=str, default="identity",
                        choices=['identity'],
                        help='which method to compute hessian')
    parser.add_argument('--hyper_train', type=str, default="all_weight",
                        choices=['weight', 'all_weight', 'dropout', 'opt_data', 'various'],
                        help='which hyperparameter to train')

    # Logging parameters
    parser.add_argument('--elementary_log_interval', type=int, default=elementary_epochs, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--hyper_log_interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save', action='store_true', default=False,
                        help='whether to save current run')

    parser.add_argument('--graph_hessian', action='store_true', default=False,
                        help='whether to save current run')

    # Miscellaneous hyperparameters
    parser.add_argument('--imsize', type=float, default=28, metavar='IMZ',
                        help='image size')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--break-perfect-val', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=100, metavar='S',
                        help='random seed (default: 1)')

    return parser.parse_args()


if __name__ == '__main__':
    torch.manual_seed(0)
    args = parse_args()

    args.break_perfect_val = True
    print(args)
    experiment(args)

    print("Finished with experiments!")



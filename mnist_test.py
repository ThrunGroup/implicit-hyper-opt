import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd import Variable
import copy
from multiprocessing import Pool

# Local imports
from data_loaders import load_mnist, load_cifar10, load_cifar100, load_ham
from models.simple_models import CNN, Net, GaussianDropout
from utils.util import eval_hessian, eval_jacobian, gather_flat_grad, conjugate_gradiant
from kfac import KFACOptimizer
from utils.csv_logger import CSVLogger
from ruamel.yaml import YAML
from models.resnet_cifar import resnet44


def init_hyper_train(args, model):
    init_hyper = None
    if args.hyper_train == 'weight':
        init_hyper = args.l2
        model.weight_decay = Variable(torch.FloatTensor([init_hyper]).cuda(), requires_grad=True)
        # if args.cuda: model.weight_decay = model.weight_decay.cuda()
    elif args.hyper_train == 'all_weight':
        init_hyper = args.l2
        num_p = sum(p.numel() for p in model.parameters())
        weights = np.ones(num_p) * init_hyper
        model.weight_decay = Variable(torch.FloatTensor(weights).cuda(), requires_grad=True)
        # if args.cuda: model.weight_decay = model.weight_decay.cuda()
    elif args.hyper_train == 'opt_data':
        model.num_opt_data = args.batch_size
        # opt_data = torch.zeros(imsize*imsize*in_channel * model.num_opt_data, requires_grad=True)
        init_x = torch.randn(imsize * imsize * in_channel * model.num_opt_data).cuda() * 0.0  # 0.1
        init_y = torch.tensor([i % num_classes for i in range(model.num_opt_data)])
        # for x, y in train_loader:
        #   init_x = gather_flat_grad(x).cuda()
        #   init_y = y
        model.opt_data = Variable(init_x, requires_grad=True)
        # torch.FloatTensor(opt_data), requires_grad=True)
        model.opt_data_y = init_y
        if args.cuda:
            # model.opt_data = model.opt_data.cuda()
            model.opt_data_y = model.opt_data_y.cuda()
    elif args.hyper_train == 'dropout':
        init_hyper = args.dropout
        model.Gaussian.dropout = Variable(torch.FloatTensor([init_hyper]), requires_grad=True)
        if args.cuda: model.Gaussian.dropout = Variable(torch.FloatTensor([init_hyper]).cuda(), requires_grad=True)
    elif args.hyper_train == 'various':
        inits = np.zeros(3) - 3
        model.various = Variable(torch.FloatTensor(inits).cuda(), requires_grad=True)
    return init_hyper

def get_hyper_train(args, model):
    if args.hyper_train == 'weight':
        return model.weight_decay
    elif args.hyper_train == 'all_weight':
        return model.weight_decay
    elif args.hyper_train == 'opt_data':
        return model.opt_data
    elif args.hyper_train == 'dropout':
        return model.Gaussian.dropout
    elif args.hyper_train == 'various':
        return model.various


def train_loss_func(args, model, x, y, network, reduction='mean'):
    predicted_y = None
    reg_loss = 0
    if args.hyper_train == 'weight':
        predicted_y = network(x)
        reg_loss = network.L2_loss()
    elif args.hyper_train == 'all_weight':
        predicted_y = network(x)
        reg_loss = network.all_L2_loss()
    elif args.hyper_train == 'opt_data':
        opt_x = network.opt_data.reshape(args.batch_size, in_channel, imsize, imsize)
        predicted_y = network.forward(opt_x)
        y = model.opt_data_y
        reg_loss = network.L2_loss()
    elif args.hyper_train == 'dropout':
        predicted_y = network(x)
    elif args.hyper_train == 'various':
        x = change_saturation_brightness(x, model.various[0], model.various[1])
        predicted_y = network(x)
        model.weight_decay = -3  # model.various[2]
        reg_loss = network.L2_loss()
    return F.cross_entropy(predicted_y, y, reduction=reduction) + reg_loss, predicted_y


def val_loss_func(args, model, x, y, network, reduction='mean'):
    predicted_y = network(x)
    loss = F.cross_entropy(predicted_y, y, reduction=reduction)
    if args.hyper_train == 'opt_data':
        regularizer = 0  # scale * hyper_sum  # scale * hyper_sum #  - torch.sum(x))
    else:
        regularizer = 0  # 1e-5 * torch.sum(torch.abs(get_hyper_train()))
    return loss + regularizer, predicted_y


def test_loss_func(args, model, x, y, network, reduction='mean'):
    return val_loss_func(args, model, x, y, network, reduction=reduction)  # , predicted_y


def prepare_data(args, x, y):
    if args.cuda: x, y = x.cuda(), y.cuda()
    x, y = Variable(x), Variable(y)
    return x, y


def batch_loss(args, model, x, y, network, loss_func, reduction='mean'):
    loss, predicted_y = loss_func(args, model, x, y, network, reduction=reduction)
    return loss, predicted_y

def train(args, model, train_loader, optimizer, train_loss_func, kfac_opt, elementary_epoch, step):
    model.train()  # _train()
    total_loss = 0.0
    # TODO (JON): Sample a mini-batch
    # TODO (JON): Change x to input
    for batch_idx, (x, y) in enumerate(train_loader):
        # Take a gradient step for this mini-batch
        optimizer.zero_grad()
        if args.hessian == 'KFAC':
            kfac_opt.zero_grad()
        x, y = prepare_data(args, x, y)
        loss, _ = batch_loss(args, model, x, y, model, train_loss_func)
        loss.backward()
        optimizer.step()
        if args.hessian == 'KFAC':
            kfac_opt.fake_step()

        total_loss += loss.item()
        step += 1
        if batch_idx >= args.train_batch_num: break

    if elementary_epoch % args.elementary_log_interval == 0:
        print(f'Train Epoch: {elementary_epoch} \tLoss: {total_loss:.6f}')

    return step, total_loss / (batch_idx + 1)


def evaluate(args, model, step, data_loader, name=None):
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        model.eval()  # TODO (JON): Do I need no_grad is using eval?

        # TODO: Sample a minibatch here?
        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = prepare_data(args, x, y)
            loss, predicted_y = batch_loss(args, model, x, y, model, test_loss_func)
            total_loss += loss.item()

            pred = predicted_y.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
            if batch_idx >= args.eval_batch_num: break
        total_loss /= (batch_idx + 1)

    # TODO (JON): Clean up print and logging?
    data_size = args.batch_size * (batch_idx + 1)
    acc = float(correct) / data_size
    print(f'Evaluate {name}, {step}: Average loss: {total_loss:.4f}, Accuracy: {correct}/{data_size} ({acc}%)')
    return acc, total_loss

def change_saturation_brightness(x, saturation, brightness):
    # print(saturation, brightness)
    saturation_noise = 1.0 + torch.randn(x.shape[0]).cuda() * torch.exp(saturation)
    brightness_noise = torch.randn(x.shape[0]).cuda() * torch.exp(brightness)
    return x * saturation_noise.view(-1, 1, 1, 1) + brightness_noise.view(-1, 1, 1, 1)

################################################################################
def experiment(args):
    print(f"Running experiment with args: {args}")
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args.train_batch_num -= 1
    args.val_batch_num -= 1
    args.eval_batch_num -= 1

    # TODO (JON): What is yaml for right now?
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.boolean_representation = ['False', 'True']

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda: torch.cuda.manual_seed(args.seed)

    ########## Setup dataset
    # TODO (JON): Verify that the loaders are shuffling the validation / test sets.
    if args.dataset == 'MNIST':
        num_train = args.datasize
        if num_train == -1: num_train = 50000
        train_loader, val_loader, test_loader = load_mnist(args.batch_size, subset=[args.datasize, args.valsize, args.testsize], num_train=num_train)
        in_channel = 1
        imsize = 28
        fc_shape = 800
        num_classes = 10
    else:
        raise Exception("Must choose MNIST dataset")
    # TODO (JON): Right now we are not using the test loader for anything.  Should evaluate it occasionally.

    ##################### Setup model
    if args.model == "mlp":
        model = Net(args.num_layers, args.dropout, imsize, in_channel, args.l2, num_classes=num_classes)
    elif args.model == "cnn":
        model = CNN(args.num_layers, args.dropout, fc_shape, args.l2, in_channel, imsize, do_alexnet=False,
                    num_classes=num_classes)
    elif args.model == "alexnet":
        model = CNN(args.num_layers, args.dropout, fc_shape, args.l2, in_channel, imsize, do_alexnet=True,
                    num_classes=num_classes)
    elif args.model == "resnet":
        model = resnet44(dropout=args.dropout, num_classes=num_classes, in_channel=in_channel)
    else:
        raise Exception("bad model")

    hyper = init_hyper_train(args, model)  # We need this when doing all_weight
    if args.cuda:
        model = model.cuda()
        model.weight_decay = model.weight_decay.cuda()
        # model.Gaussian.dropout = model.Gaussian.dropout.cuda()

    ############ Setup Optimizer
    # TODO (JON):  Add argument for other optimizers?
    init_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # , momentum=0.9)
    hyper_optimizer = torch.optim.RMSprop([get_hyper_train(args, model)])  # , lr=args.lrh)  # try 0.1 as lr

    ############## Setup Inversion Algorithms
    KFAC_damping = 1e-2
    kfac_opt = KFACOptimizer(model, damping=KFAC_damping)  # sec_optimizer

    def KFAC_optimize(epoch_h):
        # set up placeholder for the partial derivative in each batch
        total_d_val_loss_d_lambda = torch.zeros(get_hyper_train(args, model).size(0))
        if args.cuda: total_d_val_loss_d_lambda = total_d_val_loss_d_lambda.cuda()

        num_weights = sum(p.numel() for p in model.parameters())
        d_val_loss_d_theta = torch.zeros(num_weights).cuda()
        model.train()
        for batch_idx, (x, y) in enumerate(val_loader):
            model.zero_grad()
            x, y = prepare_data(args, x, y)
            val_loss, _ = batch_loss(args, model, x, y, model, val_loss_func)
            val_loss_grad = grad(val_loss, model.parameters())
            d_val_loss_d_theta += gather_flat_grad(val_loss_grad)
            if batch_idx >= args.val_batch_num: break
        d_val_loss_d_theta /= (batch_idx + 1)

        # get d theta / d lambda
        if args.hessian == 'zero':
            pass
        elif args.hessian == 'identity' or args.hessian == 'direct':
            if args.hessian == 'identity':
                pre_conditioner = d_val_loss_d_theta
                flat_pre_conditioner = pre_conditioner  # 2*pre_conditioner - args.lr*hessian_term
            elif args.hessian == 'direct':
                assert args.dataset == 'MNIST' and args.model == 'mlp' and args.num_layers == 0, "Don't do direct for large problems."
                hessian = torch.zeros(
                    num_weights, num_weights).cuda()  # grad(grad(train_loss, model.parameters()), model.parameters())
                for batch_idx, (x, y) in enumerate(train_loader):
                    x, y = prepare_data(args, x, y)
                    train_loss, _ = batch_loss(args, model, x, y, model, train_loss_func)
                    # TODO (JON): Probably don't recompute - use create_graph and retain_graph?

                    model.zero_grad(), hyper_optimizer.zero_grad()
                    d_train_loss_d_theta = grad(train_loss, model.parameters(), create_graph=True, retain_graph=True)
                    flat_d_train_loss_d_theta = gather_flat_grad(d_train_loss_d_theta)
                    for p_index, p in enumerate(flat_d_train_loss_d_theta):
                        hessian_term = grad(p, model.parameters(), retain_graph=True)
                        flat_hessian_term = gather_flat_grad(hessian_term)
                        hessian[p_index] += flat_hessian_term
                    if batch_idx >= args.train_batch_num: break
                hessian /= (batch_idx + 1)
                inv_hessian = torch.pinverse(hessian)
                pre_conditioner = d_val_loss_d_theta @ inv_hessian
                flat_pre_conditioner = pre_conditioner

            model.train()  # train()
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = prepare_data(args, x, y)
                train_loss, _ = batch_loss(args, model, x, y, model, train_loss_func)
                # TODO (JON): Probably don't recompute - use create_graph and retain_graph?

                model.zero_grad(), hyper_optimizer.zero_grad()
                d_train_loss_d_theta = grad(train_loss, model.parameters(), create_graph=True)
                flat_d_train_loss_d_theta = gather_flat_grad(d_train_loss_d_theta)

                model.zero_grad(), hyper_optimizer.zero_grad()
                flat_d_train_loss_d_theta.backward(flat_pre_conditioner)
                if get_hyper_train(args, model).grad is not None:
                    total_d_val_loss_d_lambda -= get_hyper_train(args, model).grad
                if batch_idx >= args.train_batch_num: break
            total_d_val_loss_d_lambda /= (batch_idx + 1)
        elif args.hessian == 'KFAC':
            # model.zero_grad()
            flat_pre_conditioner = torch.zeros(num_weights).cuda()
            for batch_idx, (x, y) in enumerate(train_loader):
                model.train()
                model.zero_grad(), hyper_optimizer.zero_grad()
                x, y = prepare_data(args, x, y)
                train_loss, _ = batch_loss(args, model, x, y, model, train_loss_func)
                # TODO (JON): Probably don't recompute - use create_graph and retain_graph?
                d_train_loss_d_theta = grad(train_loss, model.parameters(), create_graph=True)
                flat_d_train_loss_d_theta = gather_flat_grad(d_train_loss_d_theta)

                current = 0
                for m in model.modules():
                    if m.__class__.__name__ in ['Linear', 'Conv2d']:
                        # kfac_opt.zero_grad()
                        if m.__class__.__name__ == 'Conv2d':
                            size0, size1 = m.weight.size(0), m.weight.view(m.weight.size(0), -1).size(1)
                        else:
                            size0, size1 = m.weight.size(0), m.weight.size(1)
                        mod_size1 = size1 + 1 if m.bias is not None else size1
                        shape = (size0, (mod_size1))
                        size = size0 * mod_size1
                        pre_conditioner = kfac_opt._get_natural_grad(m,d_val_loss_d_theta[current:current + size].view(shape),KFAC_damping)
                        flat_pre_conditioner[current: current + size] = gather_flat_grad(pre_conditioner)
                        current += size
                model.zero_grad(), hyper_optimizer.zero_grad()
                flat_d_train_loss_d_theta.backward(flat_pre_conditioner)
                total_d_val_loss_d_lambda -= get_hyper_train(args, model).grad
                if batch_idx >= args.train_batch_num: break
            total_d_val_loss_d_lambda /= (batch_idx + 1)
        else:
            print(args.hessian)
            raise Exception(f"Passed {args.hessian}, not a valid choice")

        direct_d_val_loss_d_lambda = torch.zeros(get_hyper_train(args, model).size(0))
        if args.cuda: direct_d_val_loss_d_lambda = direct_d_val_loss_d_lambda.cuda()
        model.train()
        for batch_idx, (x_val, y_val) in enumerate(val_loader):
            model.zero_grad(), hyper_optimizer.zero_grad()
            x_val, y_val = prepare_data(args, x_val, y_val)
            val_loss, _ = batch_loss(args, model, x_val, y_val, model, val_loss_func)
            val_loss_grad = grad(val_loss, get_hyper_train(args, model), allow_unused=True)
            if val_loss_grad is not None and val_loss_grad[0] is not None:
                direct_d_val_loss_d_lambda += gather_flat_grad(val_loss_grad)
            else:
                break
            if batch_idx >= args.val_batch_num: break
        direct_d_val_loss_d_lambda /= (batch_idx + 1)

        get_hyper_train(args, model).grad = direct_d_val_loss_d_lambda + total_d_val_loss_d_lambda
        print("weight={}, update={}".format(get_hyper_train(args, model).norm(), get_hyper_train(args, model).grad.norm()))

        hyper_optimizer.step()
        model.zero_grad(), hyper_optimizer.zero_grad()
        return get_hyper_train(args, model), get_hyper_train(args, model).grad


    ########### Perform the training
    global_step = 0

    hp_k, update = 0, 0
    for epoch_h in range(0, args.hepochs + 1):
        print(f"Hyper epoch: {epoch_h}")
        if (epoch_h) % args.hyper_log_interval == 0:
            if args.hyper_train == 'opt_data':
                if args.dataset == 'MNIST':
                    save_learned(get_hyper_train(args, model).reshape(args.batch_size, imsize, imsize), True, args.batch_size,
                                 args)
                elif args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
                    save_learned(get_hyper_train(args, model).reshape(args.batch_size, in_channel, imsize, imsize), False,
                                 args.batch_size, args)
            elif args.hyper_train == 'various':
                print(f"saturation: {torch.sigmoid(model.various[0])}, brightness: {torch.sigmoid(model.various[1])}, decay: {torch.exp(model.various[2])}")
            eval_train_corr, eval_train_loss = evaluate(args, model, global_step, train_loader, 'train')
            # TODO (JON):  I don't know if we want normal train loss, or eval?
            eval_val_corr, eval_val_loss = evaluate(args, model, epoch_h, val_loader, 'valid')
            eval_test_corr, eval_test_loss = evaluate(args, model, epoch_h, test_loader, 'test')
            if args.break_perfect_val and eval_val_corr >= 0.999 and eval_train_corr >= 0.999:
                break

        min_loss = 10e8

        elementary_epochs = args.epochs
        if epoch_h == 0:
            elementary_epochs = args.init_epochs
        if True:  # epoch_h == 0:
            optimizer = init_optimizer
        # else:
        #    optimizer = sec_optimizer
        for epoch in range(1, elementary_epochs + 1):
            global_step, epoch_train_loss = train(args, model, train_loader, optimizer, train_loss_func, kfac_opt, epoch, global_step)
            if np.isnan(epoch_train_loss):
                print("Loss is nan, stop the loop")
                break
            elif False:  # epoch_train_loss >= min_loss:
                print(f"Breaking on epoch {epoch}. train_loss = {epoch_train_loss}, min_loss = {min_loss}")
                break
            min_loss = epoch_train_loss
        # if epoch_h == 0:
        #     continue

        hp_k, update = KFAC_optimize(epoch_h)


def save_learned(datas, is_mnist, batch_size, args):
    print("saving...")

    saturation_multiple = 5
    if not is_mnist:
        saturation_multiple = 5
    datas = torch.sigmoid(datas.detach() * saturation_multiple).cpu().numpy()
    col_size = 10
    row_size = batch_size // col_size
    if batch_size % row_size != 0:
        row_size += 1
    fig = plt.figure(figsize=(col_size, row_size))

    for i, data in enumerate(datas):
        ax = plt.subplot(row_size, col_size, i + 1)
        # plt.tight_layout()
        if is_mnist:
            plt.imshow(data, cmap='gray', interpolation='gaussian')  # 'none'
        else:
            plt.imshow(np.transpose(data, (1, 2, 0)), interpolation='gaussian')
        # plt.title(f"Ground Truth: {i}", fontsize=4)
        plt.xticks([])
        plt.yticks([])
        ax.set_aspect('auto')

    plt.subplots_adjust(wspace=0.05 * col_size / row_size, hspace=0.05)
    plt.draw()
    fig.savefig('images/learned_images_' + args.dataset + '_' + str(args.batch_size) + '_' + args.model + '.pdf')
    plt.close(fig)


if __name__ == '__main__':
    torch.manual_seed(0)

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default="MNIST", choices=['MNIST', 'CIFAR10', 'CIFAR100', 'HAM'],
                        help='which dataset to train')
    # TODO (JON): Let's test with small train,validation for now.
    opt_data_num = 20  # Temporary variable to testing
    parser.add_argument('--datasize', type=int, default=opt_data_num, metavar='DS',
                        help='train datasize')
    val_multiple = 3
    parser.add_argument('--valsize', type=int, default=-1, metavar='DS',
                        help='valid datasize')
    parser.add_argument('--testsize', type=int, default=100, metavar='DS',
                        help='test datasize')

    # Optimization hyperparameters
    # TODO (JON): Different batch sizes for train vs val?
    parser.add_argument('--batch_size', type=int, default=opt_data_num, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=250, metavar='N',
                        help='input batch size for testing (default: 1000)')
    elementary_epochs = 2
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
    # TODO (JON): We probably want sub-epoch updates on our weights and hyperparameters.total_d_val_loss_d_lambdaa
    # TODO (JON): Add how many elementary batches before a hyper batch
    # TODO (JON): Add how many hyper batches to do before going back to elementary

    # Optimizer Parameters
    # TODO (JON): I changed elementary optimizer to adam, and these aren't used anymore.
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--lrh', type=float, default=0.1, metavar='LRH',
                        help='hyperparameter learning rate (default: 0.01)')
    # TODO (JON): Specify elementary optimizer too?
    # TODO (JON): Specify hyperparameter optimizer?
    # TODO (JON): Generalize code to use a hyperparameter optimizer

    # Non-optimization hyperparameters
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout rate')
    parser.add_argument('--l2', type=float, default=-4,  # -4 worked for resnet
                        help='l2')
    parser.add_argument('--input_dropout', type=float, default=0.0,
                        help='dropout rate on input')

    # Architectural hyperparameters
    parser.add_argument('--model', type=str, default="mlp", choices=['mlp', 'cnn', 'alexnet', 'resnet', 'pretrained', ],
                        help='which model to train')
    parser.add_argument('--num_layers', type=int, default=0,
                        help='number of layers in network')

    # IFT algorithm choices
    parser.add_argument('--restart', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to reset parameter')
    parser.add_argument('--jacobian', type=str, default="direct", choices=['direct', 'product'],
                        help='which method to compute jacobian')
    parser.add_argument('--hessian', type=str, default="identity", choices=['direct', 'KFAC', 'identity', 'zero'],
                        help='which method to compute hessian')
    parser.add_argument('--hyper_train', type=str, default="opt_data",
                        choices=['weight', 'all_weight', 'dropout', 'opt_data', 'various'],
                        help='which hyperparameter to train')

    # Logging parameters
    # TODO (JON): Add how often we want to log info for hyper updates
    parser.add_argument('--elementary_log_interval', type=int, default=elementary_epochs, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--hyper_log_interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save', action='store_true', default=False,
                        help='whether to save current run')

    parser.add_argument('--graph_hessian', action='store_true', default=False,
                        help='whether to save current run')

    # Miscellaneous hyperparameters
    parser.add_argument('--imsize', type=float, default=28, metavar='IMZ',
                        help='image size')  # TODO (JON): Should this be automatically set based on dataset?
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--break-perfect-val', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=100, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()


    def setup_overfit_validation(dataset, model, num_layers):
        cur_args = copy.deepcopy(args)
        cur_args.datasize = 50
        cur_args.valsize = 50
        cur_args.testsize = -1

        cur_args.lr = 1e-4
        # cur_args.lrh = 1e-2

        cur_args.batch_size = cur_args.datasize

        cur_args.train_batch_num = 1
        cur_args.val_batch_num = 1
        cur_args.eval_batch_num = 100

        cur_args.dataset = dataset
        cur_args.model = model
        cur_args.num_layers = num_layers

        cur_args.hyper_train = 'all_weight'
        cur_args.l2 = -4

        cur_args.hessian = 'KFAC'
        cur_args.hepochs = 1000
        cur_args.epochs = 5
        cur_args.init_epochs = 5

        cur_args.elementary_log_interval = 5
        cur_args.hyper_log_interval = 50

        cur_args.graph_hessian = False
        return cur_args


    def setup_overfit_images():
        argss = []
        # TODO: Try other optimizers! Ex. Adam?
        for dataset in ['MNIST']:  #'MNIST', 'CIFAR10']:  # 'MNIST',
            for model in ['mlp']: #'mlp', 'alexnet', 'resnet']:  # 'mlp', 'cnn',
                layer_selection = [1]
#                 if model == 'mlp':
#                     layer_selection = [1, 0]
                for num_layers in layer_selection:
                    args = setup_overfit_validation(dataset, model, num_layers)
                    args.hyper_log_interval = 1
                    args.testsize = 50
                    args.break_perfect_val = True
                    args.hepochs = 500 # TODO: I SHRUNK THIS
                    args.hessian = 'KFAC'
                    if dataset == 'CIFAR10':
                        args.lrh = 1e-2
                    elif dataset == 'MNIST':
                        args.lrh = 1e-1
                    if model == 'alexnet' and dataset == 'MNIST':
                        args.lr = 7e-4  # 1e-3
                    elif model == 'resnet' and dataset == 'CIFAR10':
                        args.lr = 1e-5  # 1e-5

                    # TODO: Higher capacity models need less lr
                    # TODO: Ex. same architecture on MNIST has more capacity than on CIFAR
                    # TODO: Higher capacity hypers need less lr
                    # TODO: model and hyper progress must be balanced
                    # TODO: Idea - identical rmsprop on both?
                    argss += [args]
        return argss

    super_execute_argss = setup_overfit_images()
    # TODO (JON): I put different elementary optimizer and inverter
    do_multiprocess = False
    if do_multiprocess:
        p = Pool(min(4, len(super_execute_argss)))  # Set this to whatever the GPU can handle
        p.map(experiment, super_execute_argss)
    else:
        for execute_args in super_execute_argss:
            print(execute_args)
            experiment(execute_args)

    # TODO: Separate out the part of the code that specifies arguments for experiments!
    # TODO: Also, we should have plot_utils load paths from the arguments we provide?
    print("Finished with experiments!")

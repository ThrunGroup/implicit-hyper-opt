"""python train2.py --tune=wdecay --wdecay_type=per_param --wdecay=1e-6

python train2.py --tune=wdecay --wdecay_type=per_param --wdecay=1e-6 --lr=1e-4 --hyper_lr=1e-3
"""
import os
import sys
import csv
import ipdb
import time
import math
import hashlib
import datetime
import argparse

# YAML setup
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad

# Local imports
import data
import rnn_utils
import model_basic as model

from logger import Logger
from rnn_utils import batchify, get_batch, repackage_hidden

sys.path.insert(0, '..')
from utils.util import gather_flat_grad

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=650,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=650,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'rmsprop', 'adam'],
                    help='Elementary optimizer')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=300000,
                    help='upper epoch limit')

parser.add_argument('--hyper_lr', type=float, default=0.01,
                    help='hyper learning rate')
parser.add_argument('--hyper_optimizer', type=str, default='rmsprop', choices=['rmsprop', 'adam'],
                    help='Hyperparameter optimizer')
parser.add_argument('--warmup_epochs', type=int, default=0,
                    help='Number of iterations to train without tuning hyperparameters')
parser.add_argument('--batch_size', type=int, default=40, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--small_data', action='store_true', default=False,
                    help='Whether to use a small subset of the training set')

parser.add_argument('--dropouto', type=float, default=1e-6,
                    help='dropout applied to output (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=1e-6,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=1e-6,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=1e-6,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=1e-6,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed')
parser.add_argument('--disable_cuda', action='store_true', default=False,
                    help='Flag to DISABLE CUDA (ENABLED by default)')
parser.add_argument('--gpu', type=int, default=0,
                    help='Select which GPU to use (e.g., 0, 1, 2, or 3)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=0.,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=0.,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=0.,
                    help='weight decay applied to all weights')
parser.add_argument('--wdecay_type', type=str, default='global', choices=['global', 'per_layer', 'per_param'],
                    help='Choose the type of weight decay to use (either global, per_layer, or per_param)')

parser.add_argument('--nonmono', type=int, default=5,
                    help='Number of epochs for nonmonotonic lr decay')
parser.add_argument('--use_nonmono_decay', action='store_true', default=False,
                    help='Whether to use nonmonotonic lr decay')
parser.add_argument('--lr_decay', type=float, default=4.0,
                    help='Learning rate decay.')
parser.add_argument('--patience', type=int, default=10000,
                    help='How long to wait for the val loss to improve before early stopping.')
parser.add_argument('--train_log_interval', type=int, default=2,
                    help='Training minibatches between logging')
parser.add_argument('--val_log_interval', type=int, default=1,
                    help='Validation minibatches between logging')
parser.add_argument('--val_steps', type=int, default=1,
                    help='Validation steps')
parser.add_argument('--num_neumann_terms', type=int, default=0,
                    help='The maximum number of neumann terms to use')
parser.add_argument('--tune', type=str, default='dropouto',
                    help='Choose which hyperparameters to tune')

parser.add_argument('--save_dir', type=str, default='saves',
                    help='The base save directory.')
parser.add_argument('--prefix', type=str, default=None,
                    help='An optional prefix for the experiment name -- for uniqueness.')
parser.add_argument('--test', action='store_true', default=False,
                    help="Just run test, don't train.")
parser.add_argument('--overwrite', action='store_true', default=False,
                    help="Run the experiment and overwrite a (possibly existing) result file.")

args = parser.parse_args()
args.tied = True

if args.tune is not None:
    args.tune = args.tune.split(',')

if not args.disable_cuda and torch.cuda.is_available():
    use_device = torch.device('cuda:{}'.format(args.gpu))
else:
    use_device = torch.device('cpu')

# Set the random seed manually for reproducibility.
if args.seed:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)


# Create hyperparameters and logger
# ---------------------------------
files_used = ['train.py', 'rnn_utils.py', 'model_basic.py']
epoch_labels = ['epoch', 'time', 'train_loss', 'val_loss', 'train_ppl', 'val_ppl']
iteration_labels = ['iteration', 'time', 'train_loss', 'val_loss', 'test_loss', 'train_ppl', 'val_ppl', 'test_ppl']
for hparam_name in args.tune:
    if hparam_name == 'wdecay' and args.wdecay_type == 'per_param':
        iteration_labels += ['per_param_wdecay_mean', 'per_param_wdecay_std']
    else:
        iteration_labels += [hparam_name]
stats = { 'epoch': epoch_labels, 'iteration': iteration_labels }
logger = Logger(sys.argv, args, files=files_used, stats=stats)
# ---------------------------------

###############################################################################
# Create save folder
###############################################################################
if not args.test:
    save_dir = logger.log_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Check if the result file exists, and if so, don't run it again.
    if not args.overwrite:
        if os.path.exists(os.path.join(save_dir, 'result')):
            print("The result file {} exists! Not rerunning.".format(os.path.join(save_dir, 'result')))
            sys.exit(0)

    with open(os.path.join(save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)


###############################################################################
# Load data
###############################################################################
def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, param_optimizer], f)

def model_load(fn):
    global model, criterion, param_optimizer
    with open(fn, 'rb') as f:
        model, criterion, param_optimizer = torch.load(f)

fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1

train_data = batchify(corpus.train, args.batch_size, use_device)
hyperval_data = batchify(corpus.valid, args.batch_size, use_device)
val_data = batchify(corpus.valid, eval_batch_size, use_device)
test_data = batchify(corpus.test, test_batch_size, use_device)
ntokens = len(corpus.dictionary)

if args.small_data:
    train_data = train_data[:10]        # (10,40)
    hyperval_data = hyperval_data[:10]  # (10,40)
    val_data = val_data[:10]            # ()
    # test_data = test_data[:len(test_data)//2]
    test_data = test_data[:10]

# Initialize tunable hyperparameters
# ----------------------------------
dropouto = torch.full([1], rnn_utils.logit(args.dropouto), requires_grad=True, device='cuda:0')
dropouth = torch.full([1], rnn_utils.logit(args.dropouth), requires_grad=True, device='cuda:0')
dropouti = torch.full([1], rnn_utils.logit(args.dropouti), requires_grad=True, device='cuda:0')

dropoute = torch.full([1], rnn_utils.logit(args.dropoute), requires_grad=True, device='cuda:0')

model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, dropouto, dropouth, dropouti, dropoute, args.wdrop,
                       args.tied, wdecay=args.wdecay, wdecay_type=args.wdecay_type)
model = model.to(use_device)
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(use_device)

params = list(model.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

hparams = []
if 'wdecay' in args.tune:
    hparams.append(model.weight_decay)
if 'dropouto' in args.tune:
    hparams.append(model.dropouto)
if 'dropouth' in args.tune:
    hparams.append(model.dropouth)
if 'dropouti' in args.tune:
    hparams.append(model.dropouti)
# ----------------------------------


###############################################################################
# Training code
###############################################################################
def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    with torch.no_grad():
        total_loss = 0
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(batch_size)
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args)
            output, hidden = model(data, hidden)
            total_loss += len(data) * criterion(output.view(-1, ntokens), targets).data
            hidden = repackage_hidden(hidden)

    model.train()
    return total_loss.item() / len(data_source)


def get_hyper_train():
    return hparams


def val_loss_func(data_source):
    val_hidden = model.init_hidden(args.batch_size)

    # model.eval()  # Turn on evaluation mode which disables dropout. TODO: Do we want this here?
    ntokens = len(corpus.dictionary)
    seq_len = args.bptt

    data, targets = get_batch(data_source, 0, args, seq_len=seq_len)
    val_hidden = repackage_hidden(val_hidden)

    output, val_hidden, rnn_hs, dropped_rnn_hs = model(data, val_hidden, return_h=True)
    val_loss = criterion(output.view(-1, ntokens), targets)

    return val_loss


def train_loss_func():
    train_hidden = model.init_hidden(args.batch_size)
    ntokens = len(corpus.dictionary)
    seq_len = args.bptt

    # Turn on training mode which enables dropout.
    model.train()
    data, targets = get_batch(train_data, 0, args, seq_len=seq_len)
    output, train_hidden, rnn_hs, dropped_rnn_hs = model(data, train_hidden, return_h=True)
    xentropy_loss = criterion(output.view(-1, ntokens), targets)
    loss = xentropy_loss

    if args.wdecay_type in ['global', 'per_layer', 'per_param']:
        loss = loss + model.L2_loss()

    return xentropy_loss, loss


def zero_hypergrad(get_hyper_train):
    current_index = 0
    for p in get_hyper_train():
        p_num_params = np.prod(p.shape)
        if p.grad is not None:
            p.grad = p.grad * 0
        current_index += p_num_params


def store_hypergrad(get_hyper_train, total_d_val_loss_d_lambda):
    current_index = 0
    for p in get_hyper_train():
        p_num_params = np.prod(p.shape)
        p.grad = total_d_val_loss_d_lambda[current_index:current_index + p_num_params].view(p.shape)
        current_index += p_num_params


def make_hparam_dict():
    hparam_dict = {}
    if 'dropouto' in args.tune:
        hparam_dict['dropouto'] = torch.sigmoid(model.dropouto).item()
    if 'dropouti' in args.tune:
        hparam_dict['dropouti'] = torch.sigmoid(model.dropouti).item()
    if 'dropouth' in args.tune:
        hparam_dict['dropouth'] = torch.sigmoid(model.dropouth).item()
    if 'wdecay' in args.tune and args.wdecay_type == 'global':  # TODO(PV): Not sure how to log the wdecay when it's per-parameter... maybe mean and std dev?
        hparam_dict['wdecay'] = torch.exp(model.weight_decay).item()
    if 'wdecay' in args.tune and args.wdecay_type == 'per_param':
        hparam_dict['per_param_wdecay_mean'] = torch.exp(model.weight_decay).mean().item()
        hparam_dict['per_param_wdecay_std'] = torch.exp(model.weight_decay).std().item()
    return hparam_dict


def hyper_step(train_grad):
# def hyper_step():
    """Estimate the hypergradient, and take an update with it.
    """
    zero_hypergrad(get_hyper_train)

    # xentropy_loss, regularized_loss = train_loss_func()
    # train_grad = grad(regularized_loss, model.parameters(), create_graph=True)
    d_train_loss_d_w = gather_flat_grad(train_grad)

    # Compute gradients of the validation loss w.r.t. the weights/hypers
    num_weights, num_hypers = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in get_hyper_train())
    d_val_loss_d_theta = torch.zeros(num_weights).cuda()
    # model.eval()
    model.zero_grad()

    val_loss = val_loss_func(hyperval_data)  # eval() is used in here
    d_val_loss_d_theta += gather_flat_grad(grad(val_loss, model.parameters()))  # Do we need create_graph=True or retain_graph=True ?

    # compute d / d lambda (partial Lv / partial w * partial Lt / partial w)
    # = (partial Lv / partial w * partial^2 Lt / (partial w partial lambda))
    # indirect_grad = gather_flat_grad(grad(d_train_loss_d_w, get_hyper_train(), grad_outputs=preconditioner.view(-1)))
    hypergrad = gather_flat_grad(grad(d_train_loss_d_w, get_hyper_train(), grad_outputs=d_val_loss_d_theta.detach().view(-1)))
    get_hyper_train()[0].grad = -hypergrad
    # get_hyper_train()[0].grad = hypergrad
    # store_hypergrad(get_hyper_train, hypergrad)
    return val_loss, hypergrad.norm()


def main():
    # Loop over epochs.
    lr = args.lr
    start_time = time.time()

    train_epoch = 0
    global_step = 0

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        # param_optimizer = optim.SGD(model.parameters(), lr=args.lr)
        # param_optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        param_optimizer = optim.Adam(model.parameters(), lr=args.lr)
        hyper_optimizer = optim.Adam(get_hyper_train(), lr=args.hyper_lr)

        while train_epoch < args.epochs:
            epoch_start_time = time.time()

            model.train()
            xentropy_loss, regularized_loss = train_loss_func()

            # ---------Zero grad------------
            current_index = 0
            for p in model.parameters():
                p_num_params = np.prod(p.shape)
                if p.grad is not None:
                    p.grad = p.grad * 0  # Explicitly zeroing the gradients -- why is this required? Why not model.zero_grad() ?
                current_index += p_num_params
            param_optimizer.zero_grad()
            # -----End of zero grad---------

            train_grad = grad(regularized_loss, model.parameters(), create_graph=True)

            hyper_optimizer.zero_grad()
            val_loss, grad_norm = hyper_step(train_grad)
            # val_loss, grad_norm = hyper_step()
            hyper_optimizer.step()
            # val_loss = torch.zeros(1)

            # Replace the original gradient for the elementary optimizer step.
            current_index = 0
            flat_train_grad = gather_flat_grad(train_grad)
            for p in model.parameters():
                p_num_params = np.prod(p.shape)
                p.grad = flat_train_grad[current_index: current_index + p_num_params].view(p.shape)
                current_index += p_num_params

            param_optimizer.step()
            cur_loss = xentropy_loss
            elapsed = time.time() - start_time

            if global_step % 100 == 0:
                val_loss = evaluate(val_data, test_batch_size)
                test_loss = evaluate(test_data, test_batch_size)

                hparam_dict = make_hparam_dict()
                iteration_dict = { 'iteration': global_step, 'time': time.time() - start_time, 'train_loss': cur_loss.item(),
                                   'val_loss': val_loss, 'train_ppl': math.exp(cur_loss.item()), 'val_ppl': math.exp(val_loss),
                                   'test_loss': test_loss, 'test_ppl': math.exp(test_loss),
                                   **hparam_dict }
                logger.write('iteration', iteration_dict)

                hparam_string = ' | '.join(['{}: {}'.format(key, value) for (key, value) in hparam_dict.items()])

                print('| epoch {:3d} | step {} | lr {:05.5f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f} | val_loss: {:6.2f} | val ppl: {:6.2f} | test_loss: {:6.2f} | test ppl: {:6.2f} | {}'.format(
                      train_epoch, global_step, param_optimizer.param_groups[0]['lr'], elapsed * 1000 / args.log_interval,
                      cur_loss, math.exp(cur_loss), val_loss, math.exp(val_loss), test_loss, math.exp(test_loss), hparam_string))

            # print('| epoch {:3d} | batches | lr {:05.5f} | ms/batch {:5.2f} | '
            #       'loss {:5.2f} | ppl {:8.2f} | val_loss: {:6.2f} | val ppl: {:6.2f} | wdecay mean: {:6.4e} | wdecay std: {:6.4e} | wdecay: {}'.format(
            #       train_epoch, param_optimizer.param_groups[0]['lr'], elapsed * 1000 / args.log_interval,
            #       cur_loss, math.exp(cur_loss), val_loss, math.exp(val_loss), torch.exp(model.weight_decay).mean().item(),
            #       torch.exp(model.weight_decay).std().item(), model.weight_decay))

            start_time = time.time()
            global_step += 1
            train_epoch += 1
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
        sys.stdout.flush()


if __name__ == '__main__':
    main()

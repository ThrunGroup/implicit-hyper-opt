"""python train.py --tune=wdecay --wdecay_type=per_param --wdecay=1e-6

python train.py --wdecay=1e-9 --optimizer=sgd --lr=30 --clip=0.25 --emsize=650 --nhid=650 --nlayers=2 --dropouto=0.7 --use_nonmono_decay
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
import torch.nn.functional as F
from torch.autograd import grad

# Local imports
import data
import rnn_utils
import plot_utils
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
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')

parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'rmsprop', 'adam'],
                    help='Elementary optimizer')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=10000,
                    help='upper epoch limit')

parser.add_argument('--hyper_lr', type=float, default=1e-3,
                    help='hyper learning rate')
parser.add_argument('--hyper_optimizer', type=str, default='adam', choices=['rmsprop', 'adam'],
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
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--plot_every', type=int, default=2,
                    help='Plot perplexity & hyperparameter values every N epochs')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=1e-2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1e-2,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1e-9,
                    help='weight decay applied to all weights')
parser.add_argument('--wdecay_type', type=str, default='global', choices=['global', 'per_layer', 'per_param'],
                    help='Choose the type of weight decay to use (either global or per-parameter)')
parser.add_argument('--dropout_type', type=str, default='standard', choices=['standard', 'concrete', 'per_param'],
                    help='Choose the type of dropout (standard or concrete)')

parser.add_argument('--nonmono', type=int, default=5,
                    help='Number of epochs for nonmonotonic lr decay')
parser.add_argument('--use_nonmono_decay', action='store_true', default=False,
                    help='Whether to use nonmonotonic lr decay')
parser.add_argument('--lr_decay', type=float, default=4.0,
                    help='Learning rate decay.')
parser.add_argument('--patience', type=int, default=20,
                    help='How long to wait for the val loss to improve before early stopping.')
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
epoch_labels = ['epoch', 'time', 'train_loss', 'val_loss', 'test_loss', 'train_ppl', 'val_ppl', 'test_ppl']
iteration_labels = ['iteration', 'time', 'train_loss', 'val_loss', 'test_loss', 'train_ppl', 'val_ppl', 'test_ppl']
for hparam_name in args.tune:
    if hparam_name == 'wdecay' and args.wdecay_type == 'per_param':
        iteration_labels += ['per_param_wdecay_mean', 'per_param_wdecay_std']
    elif hparam_name in ['dropouti', 'dropouto', 'dropouth', 'wdrop'] and args.dropout_type == 'per_param':
        iteration_labels += ['{}_mean'.format(hparam_name), '{}_std'.format(hparam_name)]
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
        torch.save(model, f)
        # torch.save([model, criterion, param_optimizer], f)

def model_load(fn):
    # global model, criterion, param_optimizer
    global model
    with open(fn, 'rb') as f:
        model = torch.load(f)
        # model, criterion, param_optimizer = torch.load(f)

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
    train_data = train_data[:10]
    hyperval_data = hyperval_data[:10]
    val_data = val_data[:10]
    test_data = test_data[:len(test_data)//2]

# Initialize tunable hyperparameters
# ----------------------------------
if args.dropout_type == 'per_param':
    num_dropouts = args.nhid
else:
    num_dropouts = 1

dropouto = torch.full([num_dropouts], rnn_utils.logit(args.dropouto), requires_grad=True, device='cuda:0')
dropouth = torch.full([num_dropouts], rnn_utils.logit(args.dropouth), requires_grad=True, device='cuda:0')
dropouti = torch.full([num_dropouts], rnn_utils.logit(args.dropouti), requires_grad=True, device='cuda:0')

dropoute = torch.full([1], rnn_utils.logit(args.dropoute), requires_grad=True, device='cuda:0')

if args.dropout_type == 'per_param':
    wdrop = torch.full([4*num_dropouts, num_dropouts], rnn_utils.logit(args.wdrop), requires_grad=True, device='cuda:0')
else:
    wdrop = torch.full([1], rnn_utils.logit(args.wdrop), requires_grad=True, device='cuda:0')

alpha = torch.full([1], rnn_utils.inv_softplus(args.alpha), requires_grad=True, device='cuda:0')
beta = torch.full([1], rnn_utils.inv_softplus(args.beta), requires_grad=True, device='cuda:0')

model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, dropouto, dropouth, dropouti, dropoute, wdrop,
                       args.tied, wdecay=args.wdecay, wdecay_type=args.wdecay_type, dropout_type=args.dropout_type)
model = model.to(use_device)
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(use_device)

params = list(model.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

hparams = []
if 'dropouto' in args.tune:
    hparams.append(dropouto)
if 'dropouth' in args.tune:
    hparams.append(dropouth)
if 'dropouti' in args.tune:
    hparams.append(dropouti)
if 'dropoute' in args.tune:
    hparams.append(dropoute)
if 'wdrop' in args.tune:
    hparams.append(wdrop)
if 'wdecay' in args.tune:
    hparams.append(model.weight_decay)
if 'alpha' in args.tune:
    hparams.append(alpha)
if 'beta' in args.tune:
    hparams.append(beta)
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


val_hidden = model.init_hidden(args.batch_size)

def val_loss_func(data_source, val_epoch, val_iter, val_seq_pos):
    global global_step, val_hidden

    model.eval()  # Turn on evaluation mode which disables dropout. TODO: Do we want this here?

    ntokens = len(corpus.dictionary)
    seq_len = args.bptt

    if val_seq_pos >= len(data_source):
        val_epoch += 1
        val_seq_pos = 0
        val_hidden = model.init_hidden(args.batch_size)

    data, targets = get_batch(data_source, val_seq_pos, args, seq_len=seq_len)
    val_hidden = repackage_hidden(val_hidden)

    output, val_hidden, rnn_hs, dropped_rnn_hs = model(data, val_hidden, return_h=True)
    val_loss = criterion(output.view(-1, ntokens), targets)

    # I DON'T THINK ALPHA AND BETA SHOULD BE USED WHEN COMPUTING THE VALIDATION LOSS!
    # loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
    # loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])

    val_iter += 1
    val_seq_pos += seq_len
    global_step += 1

    return val_loss, val_epoch, val_iter, val_seq_pos


global_step = 0
epoch_train_loss = 0
train_hidden = model.init_hidden(args.batch_size)

def train_loss_func(train_epoch, train_iter, train_seq_pos):
    global epoch_train_loss, global_step, train_hidden

    train_losses = []
    cur_start_time = time.time()
    ntokens = len(corpus.dictionary)

    num_batches = len(train_data) // args.bptt
    seq_len = args.bptt

    if train_seq_pos >= len(train_data):
        train_epoch += 1
        train_seq_pos = 0
        train_hidden = model.init_hidden(args.batch_size)

    # Turn on training mode which enables dropout.
    model.train()
    data, targets = get_batch(train_data, train_seq_pos, args, seq_len=seq_len)
    train_hidden = repackage_hidden(train_hidden)

    output, train_hidden, rnn_hs, dropped_rnn_hs = model(data, train_hidden, return_h=True)

    xentropy_loss = criterion(output.view(-1, ntokens), targets)
    loss = xentropy_loss

    loss = loss + sum(F.softplus(alpha) * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
    loss = loss + sum(F.softplus(beta) * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])

    if 'wdecay' in args.tune and args.wdecay_type in ['global', 'per_layer', 'per_param']:
        loss = loss + model.L2_loss()

    train_losses.append(loss.item())
    # epoch_train_loss += loss.item()
    epoch_train_loss += xentropy_loss.item()  # non-regularized loss

    train_iter += 1
    train_seq_pos += seq_len
    global_step += 1

    return xentropy_loss, loss, train_epoch, train_iter, train_seq_pos


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


def neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr, num_neumann_terms):
    preconditioner = d_val_loss_d_theta.detach()
    counter = preconditioner
    old_size = torch.sum(counter ** 2)
    # print(f"term {-1}: size = {torch.sum(preconditioner ** 2)}")

    # Do the fixed point iteration to approximate the vector-inverseHessian product
    for i in range(num_neumann_terms):
        old_counter = counter

        # This increments counter to counter * (I - hessian) = counter - counter * hessian
        hessian_term = (counter.view(1, -1) @ d_train_loss_d_w.view(-1, 1) @ d_train_loss_d_w.view(1, -1)).view(-1)
        counter = counter - elementary_lr * hessian_term

        size, diff = torch.sum(counter ** 2), torch.sum((counter - old_counter) ** 2)
        rel_change = size / old_size
        print(f"term {i}: size = {size}, rel_change = {rel_change}, diff={diff}")
        if rel_change > 0.9999: break

        preconditioner = preconditioner + counter
        old_size = size
    return preconditioner


def hyper_step(d_train_loss_d_w):
    """Estimate the hypergradient, and take an update with it.
    """
    global val_epoch, val_iter, val_seq_pos

    zero_hypergrad(get_hyper_train)
    d_train_loss_d_w = gather_flat_grad(d_train_loss_d_w)

    # Compute gradients of the validation loss w.r.t. the weights/hypers
    num_weights, num_hypers = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in get_hyper_train())
    d_val_loss_d_theta = torch.zeros(num_weights).cuda()
    # model.train()
    model.eval()
    model.zero_grad()

    val_loss, val_epoch, val_iter, val_seq_pos = val_loss_func(hyperval_data, val_epoch, val_iter, val_seq_pos)  # eval() is used in here
    d_val_loss_d_theta += gather_flat_grad(grad(val_loss, model.parameters()))  # Do we need create_graph=True or retain_graph=True ?

    # Initialize the preconditioner and counter
    preconditioner = neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w,
                                                      args.lr, args.num_neumann_terms)

    # compute d / d lambda (partial Lv / partial w * partial Lt / partial w)
    # = (partial Lv / partial w * partial^2 Lt / (partial w partial lambda))
    indirect_grad = gather_flat_grad(grad(d_train_loss_d_w, get_hyper_train(), grad_outputs=preconditioner.view(-1)))
    # indirect_grad = gather_flat_grad(grad(d_train_loss_d_w, get_hyper_train(), grad_outputs=d_val_loss_d_theta.detach().view(-1)))
    hypergrad = indirect_grad

    store_hypergrad(get_hyper_train, -hypergrad)
    # get_hyper_train()[0].grad = -hypergrad
    return val_loss, hypergrad.norm()


def clip_grad_norm(gradients, max_norm, norm_type=2):

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    total_norm = 0
    for g in gradients:
        # param_norm = g.data.norm(norm_type)
        param_norm = g.norm(norm_type)
        total_norm += param_norm.item() ** norm_type

    total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)

    gradients = list(gradients)

    if clip_coef < 1:
        for i in range(len(gradients)):
            # g.data.mul_(clip_coef)
            gradients[i] = gradients[i] * clip_coef  # TODO: Check this

    return gradients, total_norm


def make_hparam_dict():
    hparam_dict = {}
    if 'dropouto' in args.tune:
        if args.dropout_type == 'per_param':
            hparam_dict['dropouto_mean'] = torch.sigmoid(dropouto).mean().item()
            hparam_dict['dropouto_std'] = torch.sigmoid(dropouto).std().item()
        else:
            hparam_dict['dropouto'] = torch.sigmoid(dropouto).item()
    if 'dropouti' in args.tune:
        if args.dropout_type == 'per_param':
            hparam_dict['dropouti_mean'] = torch.sigmoid(dropouti).mean().item()
            hparam_dict['dropouti_std'] = torch.sigmoid(dropouti).std().item()            
        else:
            hparam_dict['dropouti'] = torch.sigmoid(dropouti).item()
    if 'dropouth' in args.tune:
        if args.dropout_type == 'per_param':
            hparam_dict['dropouth_mean'] = torch.sigmoid(dropouth).mean().item()
            hparam_dict['dropouth_std'] = torch.sigmoid(dropouth).std().item()            
        else:
            hparam_dict['dropouth'] = torch.sigmoid(dropouth).item()
    if 'dropoute' in args.tune:
        hparam_dict['dropoute'] = torch.sigmoid(dropoute).item()
    if 'wdrop' in args.tune:
        if args.dropout_type == 'per_param':
            hparam_dict['wdrop_mean'] = torch.sigmoid(wdrop).mean().item()
            hparam_dict['wdrop_std'] = torch.sigmoid(wdrop).std().item()
        else:
            hparam_dict['wdrop'] = torch.sigmoid(wdrop).item()
    if 'alpha' in args.tune:
        hparam_dict['alpha'] = F.softplus(alpha).item()
    if 'beta' in args.tune:
        hparam_dict['beta'] = F.softplus(beta).item()
    if 'wdecay' in args.tune and args.wdecay_type == 'global':  # TODO(PV): Not sure how to log the wdecay when it's per-parameter... maybe mean and std dev?
        hparam_dict['wdecay'] = torch.exp(model.weight_decay).item()
    if 'wdecay' in args.tune and args.wdecay_type == 'per_param':
        hparam_dict['per_param_wdecay_mean'] = torch.exp(model.weight_decay).mean().item()
        hparam_dict['per_param_wdecay_std'] = torch.exp(model.weight_decay).std().item()
    return hparam_dict


train_epoch, val_epoch = 0, 0
train_iter, val_iter = 0, 0
train_seq_pos, val_seq_pos = 0, 0

def main():
    global train_epoch, val_epoch, train_iter, val_iter, train_seq_pos, val_seq_pos, epoch_train_loss

    # Loop over epochs.
    lr = args.lr
    best_val_loss = []
    stored_loss = 1e8
    start_time = time.time()

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        if args.optimizer == 'sgd':
            param_optimizer = optim.SGD(model.parameters(), lr=args.lr)
        elif args.optimizer == 'rmsprop':
            param_optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        elif args.optimizer == 'adam':
            param_optimizer = optim.Adam(model.parameters(), lr=args.lr)

        if args.hyper_optimizer == 'rmsprop':
            hyper_optimizer = optim.RMSprop(get_hyper_train(), lr=args.hyper_lr)
        elif args.hyper_optimizer == 'adam':
            hyper_optimizer = optim.Adam(get_hyper_train(), lr=args.hyper_lr)

        patience_elapsed = 0
        old_train_epoch = 0
        total_loss = 0

        epoch_start_time = time.time()

        while train_epoch < args.epochs and patience_elapsed < args.patience:

            model.train()
            xentropy_loss, regularized_loss, train_epoch, train_iter, train_seq_pos = train_loss_func(train_epoch, train_iter, train_seq_pos)

            current_index = 0
            for p in model.parameters():
                p_num_params = np.prod(p.shape)
                if p.grad is not None:
                    p.grad = p.grad * 0  # Explicitly zeroing the gradients -- why is this required? Why not model.zero_grad() ?
                current_index += p_num_params
            param_optimizer.zero_grad()
            train_grad = grad(regularized_loss, model.parameters(), create_graph=True)

            if args.clip:
                train_grad, total_grad_norm = clip_grad_norm(train_grad, args.clip)

            val_loss, grad_norm = hyper_step(train_grad)
            if train_epoch >= args.warmup_epochs:
                hyper_optimizer.step()

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

            start_time = time.time()

            hparam_dict = make_hparam_dict()
            iteration_dict = { 'iteration': global_step, 'time': time.time() - start_time, 'train_loss': cur_loss.item(),
                               'val_loss': val_loss.item(), 'train_ppl': math.exp(cur_loss.item()), 'val_ppl': math.exp(val_loss.item()),
                               **hparam_dict }
            logger.write('iteration', iteration_dict)

            hparam_string = ' | '.join(['{}: {}'.format(key, value) for (key, value) in hparam_dict.items()])

            total_loss += regularized_loss.item()
            if train_iter % args.log_interval == 0 and train_iter > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f} | {}'.format(
                    train_epoch, train_iter, len(train_data) // args.bptt, param_optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), hparam_string))
                sys.stdout.flush()
                total_loss = 0

            # print('| epoch {:3d} | batches | lr {:05.5f} | ms/batch {:5.2f} | '
            #       'loss {:5.2f} | ppl {:8.2f} | val_loss: {:6.2f} | val ppl: {:6.2f} | {}'.format(
            #       train_epoch, param_optimizer.param_groups[0]['lr'], elapsed * 1000 / args.log_interval,
            #       cur_loss, math.exp(cur_loss), val_loss, math.exp(val_loss), hparam_string))

            # print('| epoch {:3d} | batches | lr {:05.5f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
            #       train_epoch, param_optimizer.param_groups[0]['lr'], elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))

            # Once we have completed a whole epoch on the training set
            if train_epoch != old_train_epoch:
                num_train_batches = len(train_data) // args.bptt
                if args.small_data:
                    mean_epoch_train_loss = epoch_train_loss
                else:
                    mean_epoch_train_loss = epoch_train_loss / num_train_batches

                val_loss = evaluate(val_data, eval_batch_size)
                test_loss = evaluate(test_data, test_batch_size)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | lr: {:6.4f} | train loss {:5.2f} | train ppl {:6.2f} | val loss {:5.2f} | val ppl {:5.2f} | test loss {:5.2f} | test ppl {:5.2f}'.format(
                        train_epoch, (time.time() - epoch_start_time), param_optimizer.param_groups[0]['lr'],
                        mean_epoch_train_loss, math.exp(mean_epoch_train_loss), val_loss, math.exp(val_loss),
                        test_loss, math.exp(test_loss)))
                print('-' * 89)
                sys.stdout.flush()

                epoch_start_time = time.time()

                ###########################
                ###    TRAINING LOG     ###
                ###########################
                elapsed_time = time.process_time() - start_time

                epoch_dict = { 'epoch': old_train_epoch, 'time': elapsed_time, 'train_loss': mean_epoch_train_loss, 'val_loss': val_loss, 'test_loss': test_loss,
                               'train_ppl': math.exp(mean_epoch_train_loss), 'val_ppl': math.exp(val_loss), 'test_ppl': math.exp(test_loss) }
                logger.write('epoch', epoch_dict)

                if train_epoch % args.plot_every == 0:
                    # Save plots of train/val perplexity and hyperparameter trajectories
                    plot_utils.plot_epoch_ppl(logger.log_dir, save_to=os.path.join(logger.log_dir, 'ppl.png'))
                    plot_utils.plot_epoch_loss(logger.log_dir, save_to=os.path.join(logger.log_dir, 'loss.png'))
                    plot_utils.plot_hparams(logger.log_dir, save_to=os.path.join(logger.log_dir, 'hparams.png'))

                if args.use_nonmono_decay:
                    if len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono]):
                        print('Decaying the learning rate')
                        sys.stdout.flush()
                        param_optimizer.param_groups[0]['lr'] /= args.lr_decay

                old_train_epoch = train_epoch
                epoch_train_loss = 0

                if val_loss < stored_loss:
                    model_save(os.path.join(logger.log_dir, args.save))
                    print('Saving model (new best validation)')
                    sys.stdout.flush()
                    stored_loss = val_loss
                    patience_elapsed = 0
                else:
                    patience_elapsed += 1

                if param_optimizer.param_groups[0]['lr'] < 0.0001:
                    break

                best_val_loss.append(val_loss)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
        sys.stdout.flush()


    # Load the best saved model.
    model_load(os.path.join(save_dir, args.save))

    # Run on val and test data.
    val_loss = evaluate(val_data, eval_batch_size)
    test_loss = evaluate(test_data, test_batch_size)
    print('=' * 89)
    print('| End of training | val loss {:8.5f} | val ppl {:8.5f} | test loss {:8.5f} | test ppl {:8.5f}'.format(
           val_loss, math.exp(val_loss), test_loss, math.exp(test_loss)))
    print('=' * 89)
    sys.stdout.flush()

    # Save the final val and test performance to a results file
    with open(os.path.join(logger.log_dir, 'result.csv'), 'w') as result_file:
        result_writer = csv.DictWriter(result_file, fieldnames=['val_loss', 'val_ppl', 'test_loss', 'test_ppl'])
        result_writer.writeheader()
        result_writer.writerow({ 'val_loss': val_loss, 'val_ppl': math.exp(val_loss), 'test_loss': test_loss, 'test_ppl': math.exp(test_loss) })
        result_file.flush()


if __name__ == '__main__':
    main()

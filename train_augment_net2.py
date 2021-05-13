import ipdb

import copy
import sys
import time

import argparse
import numpy as np
from tqdm import tqdm

import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
from multiprocessing import Pool

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

# Local imports
import data_loaders
from data_loaders import load_mnist, load_boston
from models.cnn_mlp import CNN_MLP
from utils.util import gather_flat_grad

from models import resnet_cifar
from models.unet import UNet
from models.resnet import ResNet18
from models.simple_models import Net
from models.wide_resnet import WideResNet
from models.simple_models import Net
from models.simple_models import CNN, Net

from train_augment_net_multiple import load_logger, get_id
from train_augment_net_graph import save_images
from train_augment_net_multiple import make_parser, make_argss




def saver(epoch, elementary_model, elementary_optimizer, augment_net, reweighting_net, hyper_optimizer, path):
    """
    Saves torch models

    :param epoch:
    :param elementary_model:
    :param elementary_optimizer:
    :param augment_net:
    :param reweighting_net:
    :param hyper_optimizer:
    :param path:
    :return:
    """
    if 'weight_decay' in elementary_model.__dict__:
        torch.save({
            'epoch': epoch,
            'elementary_model_state_dict': elementary_model.state_dict(),
            'weight_decay': elementary_model.weight_decay,
            'elementary_optimizer_state_dict': elementary_optimizer.state_dict(),
            'augment_model_state_dict': augment_net.state_dict(),
            'reweighting_net_state_dict': reweighting_net.state_dict(),
            'hyper_optimizer_state_dict': hyper_optimizer.state_dict()
        }, path + '/checkpoint.pt')
    else:
        torch.save({
            'epoch': epoch,
            'elementary_model_state_dict': elementary_model.state_dict(),
            'elementary_optimizer_state_dict': elementary_optimizer.state_dict(),
            'augment_model_state_dict': augment_net.state_dict(),
            'reweighting_net_state_dict': reweighting_net.state_dict(),
            'hyper_optimizer_state_dict': hyper_optimizer.state_dict()
        }, path + '/checkpoint.pt')


def load_baseline_model(args):
    """
    Load a simple baseline model AND dataset
    Note that this sets the model to training mode

    :param args:
    :return:
    """
    if args.dataset == 'cifar10':
        imsize, in_channel, num_classes = 32, 3, 10
        train_loader, val_loader, test_loader = data_loaders.load_cifar10(args.batch_size, val_split=True,
                                                                          augmentation=args.data_augmentation,
                                                                          subset=[args.train_size, args.val_size,
                                                                                  args.test_size])
    elif args.dataset == 'cifar100':
        imsize, in_channel, num_classes = 32, 3, 100
        train_loader, val_loader, test_loader = data_loaders.load_cifar100(args.batch_size, val_split=True,
                                                                           augmentation=args.data_augmentation,
                                                                           subset=[args.train_size, args.val_size,
                                                                                   args.test_size])
    elif args.dataset == 'mnist':
        imsize, in_channel, num_classes = 28, 1, 10
        num_train = 50000
        train_loader, val_loader, test_loader = load_mnist(args.batch_size,
                                                           subset=[args.train_size, args.val_size, args.test_size],
                                                           num_train=num_train, only_split_train=False)
    elif args.dataset == 'boston':
        imsize, in_channel, num_classes = 13, 1, 1
        train_loader, val_loader, test_loader = load_boston(args.batch_size)

    init_l2 = -7  # TODO: Important to make sure this is small enough to be unregularized when starting?
    if args.model == 'resnet18':
        cnn = ResNet18(num_classes=num_classes)
    elif args.model == 'wideresnet':
        cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)
    elif args.model[:3] == 'mlp':
        cnn = Net(args.num_layers, 0.0, imsize, in_channel, init_l2, num_classes=num_classes,
                  do_classification=args.do_classification)
    elif args.model == 'cnn_mlp':
        cnn = CNN_MLP(learning_rate=0.0001)

    checkpoint = None
    if args.load_baseline_checkpoint:
        checkpoint = torch.load(args.load_baseline_checkpoint)
        cnn.load_state_dict(checkpoint['model_state_dict'])

    model = cnn.cuda()
    if args.use_weight_decay:
        if args.weight_decay_all:
            num_p = sum(p.numel() for p in model.parameters())
            weights = np.ones(num_p) * init_l2
            model.weight_decay = Variable(torch.FloatTensor(weights).cuda(), requires_grad=True)
        else:
            weights = init_l2
            model.weight_decay = Variable(torch.FloatTensor([weights]).cuda(), requires_grad=True)
        model.weight_decay = model.weight_decay.cuda()
    model.train()
    return model, train_loader, val_loader, test_loader, checkpoint


def load_finetuned_model(args, baseline_model):
    """
    Loads the augmentation net, sample reweighting net, and baseline model
    Note: sets all these models to train mode

    :param args:
    :param baseline_model:
    :return:
    """
    # augment_net = Net(0, 0.0, 32, 3, 0.0, num_classes=32**2 * 3, do_res=True)
    if args.dataset == 'mnist':
        imsize, in_channel, num_classes = 28, 1, 10
    else:
        imsize, in_channel, num_classes = 32, 3, 10

    augment_net = UNet(in_channels=in_channel, n_classes=in_channel, depth=2, wf=3, padding=True, batch_norm=False,
                       do_noise_channel=True,
                       up_mode='upconv', use_identity_residual=True)  # TODO(PV): Initialize UNet properly
    # TODO (JON): DEPTH 1 WORKED WELL.  Changed upconv to upsample.  Use a wf of 2.

    # This ResNet outputs scalar weights to be applied element-wise to the per-example losses
    reweighting_net = Net(1, 0.0, imsize, in_channel, 0.0, num_classes=1)
    # resnet_cifar.resnet20(num_classes=1)

    if args.load_finetune_checkpoint:
        checkpoint = torch.load(args.load_finetune_checkpoint)
        # temp_baseline_model = baseline_model
        # baseline_model.load_state_dict(checkpoint['elementary_model_state_dict'])
        if 'weight_decay' in checkpoint:
            baseline_model.weight_decay = checkpoint['weight_decay']
        # baseline_model.weight_decay = temp_baseline_model.weight_decay
        # baseline_model.load_state_dict(checkpoint['elementary_model_state_dict'])
        augment_net.load_state_dict(checkpoint['augment_model_state_dict'])
        try:
            reweighting_net.load_state_dict(checkpoint['reweighting_model_state_dict'])
        except KeyError:
            pass

    augment_net, reweighting_net, baseline_model = augment_net.cuda(), reweighting_net.cuda(), baseline_model.cuda()
    augment_net.train(), reweighting_net.train(), baseline_model.train()
    return augment_net, reweighting_net, baseline_model


def zero_hypergrad(get_hyper_train):
    """
    :param get_hyper_train:
    :return:
    """
    current_index = 0
    for p in get_hyper_train():
        p_num_params = np.prod(p.shape)
        if p.grad is not None:
            p.grad = p.grad * 0 # TODO (@Mo): Is this necessary? Could just set to 0?
        current_index += p_num_params


def store_hypergrad(get_hyper_train, total_d_val_loss_d_lambda):
    """
    Updates the hypergradients and the number of hyperparameters?

    :param get_hyper_train:
    :param total_d_val_loss_d_lambda:
    :return:
    """
    current_index = 0
    for p in get_hyper_train():
        p_num_params = np.prod(p.shape)
        p.grad = total_d_val_loss_d_lambda[current_index:current_index + p_num_params].view(p.shape)
        current_index += p_num_params


def neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr, num_neumann_terms, model):
    preconditioner = d_val_loss_d_theta.detach()
    counter = preconditioner
    # Do the fixed point iteration to approximate the vector-inverseHessian product
    for i in range(num_neumann_terms):
        old_counter = counter
        # This increments counter to counter * (I - hessian) = counter - counter * hessian
        hessian_term = gather_flat_grad(
            grad(d_train_loss_d_w, model.parameters(), grad_outputs=counter.view(-1), retain_graph=True))
        counter = old_counter - elementary_lr * hessian_term
        preconditioner = preconditioner + counter
    return elementary_lr * preconditioner


def cg_batch(A_bmm, B, M_bmm=None, X0=None, rtol=1e-4, atol=0.0, maxiter=10, verbose=True):
    """
    Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.

    This function solves a batch of matrix linear systems of the form

        A_i X_i = B_i,  i=1,...,K,

    where A_i is a n x n positive definite matrix and B_i is a n x m matrix,
    and X_i is the n x m matrix representing the solution for the ith system.

    Args:
        A_bmm: A callable that performs a batch matrix multiply of A and a K x n x m matrix.
        B: A K x n x m matrix representing the right hand sides.
        M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
            matrices M and a K x n x m matrix. (default=identity matrix)
        X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
        rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
        atol: (optional) Absolute tolerance for norm of residual. (default=0)
        maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
        verbose: (optional) Whether or not to print status messages. (default=False)
    """
    K, n, m = B.shape

    if M_bmm is None:
        M_bmm = lambda x: x
    if X0 is None:
        X0 = M_bmm(B)
    if maxiter is None:
        maxiter = 5 * n

    assert B.shape == (K, n, m)
    assert X0.shape == (K, n, m)
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)

    X_k = X0
    R_k = B - A_bmm(X_k)
    Z_k = M_bmm(R_k)

    P_k = torch.zeros_like(Z_k)

    P_k1 = P_k
    R_k1 = R_k
    R_k2 = R_k
    X_k1 = X0
    Z_k1 = Z_k
    Z_k2 = Z_k

    B_norm = torch.norm(B, dim=1)
    stopping_matrix = torch.max(rtol * B_norm, atol * torch.ones_like(B_norm))

    if verbose:
        residual_norm = torch.norm(A_bmm(X_k) - B, dim=1)
        print("%03s | %010s %06s" % ("it", torch.max(residual_norm - stopping_matrix), "it/s"))

    optimal = False
    start = time.perf_counter()
    cur_error = 1e-8
    epsilon = 1e-2
    for k in range(1, maxiter + 1):
        # epsilon = cur_error ** 3  # 1e-8
        start_iter = time.perf_counter()
        Z_k = M_bmm(R_k)

        if k == 1:
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = (R_k2 * Z_k2).sum(1)
            denominator[denominator < epsilon / 2] = epsilon  # epsilon
            beta = (R_k1 * Z_k1).sum(1) / denominator
            P_k = Z_k1 + beta.unsqueeze(1) * P_k1

        denominator = (P_k * A_bmm(P_k)).sum(1)
        denominator[denominator < epsilon / 2] = epsilon
        alpha = (R_k1 * Z_k1).sum(1) / denominator
        X_k = X_k1 + alpha.unsqueeze(1) * P_k
        R_k = R_k1 - alpha.unsqueeze(1) * A_bmm(P_k)
        end_iter = time.perf_counter()

        residual_norm = torch.norm(A_bmm(X_k) - B, dim=1)

        cur_error = torch.max(residual_norm - stopping_matrix)
        if verbose:
            print("%03d | %8.6e %4.2f" %
                  (k, cur_error,
                   1. / (end_iter - start_iter)))

        if (residual_norm <= stopping_matrix).all():
            optimal = True
            break

    end = time.perf_counter()

    if verbose:
        if optimal:
            print("Terminated in %d steps (optimal). Took %.3f ms." %
                  (k, (end - start) * 1000))
        else:
            print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                  (k, (end - start) * 1000))

    info = {
        "niter": k,
        "optimal": optimal
    }

    return X_k, info


def get_models(args):
    '''
    Loads both the baseline model and the finetuning models in train mode
    '''
    model, train_loader, val_loader, test_loader, checkpoint = load_baseline_model(args)
    augment_net, reweighting_net, model = load_finetuned_model(args, model)
    return model, train_loader, val_loader, test_loader, augment_net, reweighting_net, checkpoint


def experiment(args):
    if args.do_print:
        print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load the baseline model
    args.load_baseline_checkpoint = None  # '/h/lorraine/PycharmProjects/CG_IFT_test/baseline_checkpoints/cifar10_resnet18_sgdm_lr0.1_wd0.0005_aug1.pt'
    args.load_finetune_checkpoint = ''  # TODO: Make it load the augment net if this is provided
    args.only_print_final_vals = False
    model, train_loader, val_loader, test_loader, augment_net, reweighting_net, checkpoint = get_models(args)

    # Load the logger
    csv_logger, test_id = load_logger(args)
    args.save_loc = './finetuned_checkpoints/' + get_id(args)

    # Hyperparameter access functions
    def get_hyper_train():
        # return torch.cat([p.view(-1) for p in augment_net.parameters()])
        if args.use_augment_net and args.use_reweighting_net:
            return list(augment_net.parameters()) + list(reweighting_net.parameters())
        elif args.use_augment_net:
            return augment_net.parameters()
        elif args.use_reweighting_net:
            return reweighting_net.parameters()
        elif args.use_weight_decay:
            return [model.weight_decay]

    def get_hyper_train_flat():
        if args.use_augment_net and args.use_reweighting_net:
            return torch.cat([torch.cat([p.view(-1) for p in augment_net.parameters()]),
                              torch.cat([p.view(-1) for p in reweighting_net.parameters()])])
        elif args.use_reweighting_net:
            return torch.cat([p.view(-1) for p in reweighting_net.parameters()])
        elif args.use_augment_net:
            return torch.cat([p.view(-1) for p in augment_net.parameters()])
        elif args.use_weight_decay:
            return model.weight_decay  # TODO: This correct?

    # Setup the optimizers
    if args.load_baseline_checkpoint is not None:
        args.lr = args.lr * 0.2 * 0.2 * 0.2
    if args.use_weight_decay:
        # optimizer = optim.Adam(model.parameters(), lr=1e-3)
        args.wdecay = 0

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wdecay)
    if args.dataset == 'boston':
        optimizer = optim.Adam(model.parameters())
    use_scheduler = False
    if not args.do_simple:
        use_scheduler = True
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)  # [60, 120, 160]
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    use_hyper_scheduler = False
    hyper_optimizer = optim.RMSprop(get_hyper_train())
    if not args.do_simple:
        hyper_optimizer = optim.SGD(get_hyper_train(), lr=args.lr, momentum=0.9, nesterov=True)
        use_hyper_scheduler = True
    hyper_scheduler = MultiStepLR(hyper_optimizer, milestones=[40, 100, 140], gamma=0.2)

    graph_iter = 0

    def train_loss_func(x, y):
        x, y = x.cuda(), y.cuda()
        reg = 0.

        if args.use_augment_net and (args.num_neumann_terms >= 0 or args.load_finetune_checkpoint != ''):
            x = augment_net(x, class_label=y)

        pred = model(x)
        if args.do_classification:
            xentropy_loss = F.cross_entropy(pred, y, reduction='none')
        else:
            xentropy_loss = F.mse_loss(pred, y)

        if args.use_reweighting_net and args.num_neumann_terms >= 0:
            loss_weights = reweighting_net(x)  # TODO: Or reweighting_net(augment_x) ??
            loss_weights = loss_weights.squeeze()
            loss_weights = F.sigmoid(loss_weights / 10.0)
            # loss_weights = (loss_weights - torch.mean(loss_weights)) / torch.std(loss_weights)
            loss_weights = F.softmax(loss_weights) * args.batch_size
            # TODO: Want loss_weight vs x_entropy_loss

            if args.do_diagnostic:
                nonlocal graph_iter
                if graph_iter % 100 == 0:
                    np_loss = xentropy_loss.data.cpu().numpy()
                    np_weight = loss_weights.data.cpu().numpy()
                    for i in range(10):
                        class_indices = (y == i).cpu().numpy()
                        class_indices = [val * ind for val, ind in enumerate(class_indices) if val != 0]
                        plt.scatter(np_loss[class_indices], np_weight[class_indices], alpha=0.5, label=str(i))
                    # plt.scatter((xentropy_loss*loss_weights).data.cpu().numpy(), loss_weights.data.cpu().numpy(), alpha=0.5, label='weighted')
                    # print(np_loss)
                    plt.ylim([np.min(np_weight) / 2.0, np.max(np_weight) * 2.0])
                    plt.xlim([np.min(np_loss) / 2.0, np.max(np_loss) * 2.0])
                    plt.yscale('log')
                    plt.xscale('log')
                    plt.axhline(1.0, c='k')
                    plt.ylabel("loss_weights")
                    plt.xlabel("xentropy_loss")
                    plt.legend()
                    plt.savefig("images/aaaa_lossWeightvsEntropy.pdf")
                    plt.clf()

            xentropy_loss = xentropy_loss * loss_weights
        graph_iter += 1

        if args.use_weight_decay:
            if args.weight_decay_all:
                reg = model.all_L2_loss()
            else:
                reg = model.L2_loss()
            if args.load_finetune_checkpoint == '' and args.num_neumann_terms < 0:
                reg = 0
        # print(f"reg: {reg}, mean_hyper: {torch.mean(get_hyper_train()[0])}")
        final_loss = xentropy_loss.mean() + reg
        return final_loss, pred

    use_reg = args.use_augment_net and not args.use_reweighting_net
    reg_anneal_epoch = 0
    stop_reg_epoch = 200
    if args.reg_weight == 0:
        use_reg = False

    def val_loss_func(x, y):
        x, y = x.cuda(), y.cuda()
        pred = model(x)
        if args.do_classification:
            xentropy_loss = F.cross_entropy(pred, y)
        else:
            xentropy_loss = F.mse_loss(pred, y)

        reg = 0
        if args.use_augment_net:
            if use_reg:
                num_sample = 10
                xs = torch.zeros(num_sample, x.shape[0], x.shape[1], x.shape[2], x.shape[3]).cuda()
                for i in range(num_sample):
                    xs[i] = augment_net(x, class_label=y)
                xs_diffs = (torch.abs(torch.mean(xs, dim=0) - x))
                diff_loss = torch.mean(xs_diffs)
                stds = torch.std(xs, dim=0)
                entrop_loss = -torch.mean(stds)
                # TODO : Remember to add direct grad back in to hyper_step
                reg = args.reg_weight * (diff_loss + entrop_loss)
            else:
                reg = 0

        # reg *= (args.num_finetune_epochs - reg_anneal_epoch) / (args.num_finetune_epochs + 2)
        if reg_anneal_epoch >= stop_reg_epoch:
            reg *= 0
        return xentropy_loss + reg

    def test(loader, do_test_augment=True, num_augment=5):
        # model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        correct, total = 0., 0.
        losses = []
        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                pred = model(images)
                if do_test_augment:
                    if args.use_augment_net and (args.num_neumann_terms >= 0 or args.load_finetune_checkpoint != ''):
                        shape_0, shape_1 = pred.shape[0], pred.shape[1]
                        pred = pred.view(1, shape_0, shape_1)  # Batch size, num_classes
                        for _ in range(num_augment):
                            pred = torch.cat((pred, model(augment_net(images)).view(1, shape_0, shape_1))) # TODO (@Mo): Should we be augmenting the validation images?
                        pred = torch.mean(pred, dim=0) # TODO (@Mo): This is certainly a weird way of getting a prediction...
                if args.do_classification:
                    xentropy_loss = F.cross_entropy(pred, labels)
                else:
                    xentropy_loss = F.mse_loss(pred, labels)
                losses.append(xentropy_loss.item())

            if args.do_classification:
                pred = torch.max(pred.data, 1)[1]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
            else:
                correct, total = 0, 1

        avg_loss = float(np.mean(losses))
        acc = correct / total
        model.train()
        return avg_loss, acc

    def hyper_step(elementary_lr, do_true_inverse=False):
        # hyper_step(get_hyper_train, model, val_loss_func, val_loader, old_d_train_loss_d_w, elementary_lr, use_reg, args, train_loader, train_loss_func, elementary_optimizer):
        """
        Estimate the hypergradient, and take an update with it.

        :param get_hyper_train:  A function which returns the hyperparameters we want to tune.
        :param model:  A function which returns the elementary parameters we want to tune.
        :param val_loss_func:  A function which takes input x and output y, then returns the scalar valued loss.
        :param val_loader: A generator for input x, output y tuples.
        :param d_train_loss_d_w:  The derivative of the training loss with respect to elementary parameters.
        :param hyper_optimizer: The optimizer which updates the hyperparameters.
        :return: The scalar valued validation loss, the hyperparameter norm, and the hypergradient norm.
        """
        zero_hypergrad(get_hyper_train)
        num_weights, num_hypers = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in get_hyper_train())
        print(f"num_weights : {num_weights}, num_hypers : {num_hypers}")

        # d_train_loss_d_w = gather_flat_grad(d_train_loss_d_w)  # TODO: Commented this out!
        d_train_loss_d_w = torch.zeros(num_weights).cuda()
        model.train(), model.zero_grad()
        for batch_idx, (x, y) in enumerate(train_loader):
            train_loss, _ = train_loss_func(x, y)
            optimizer.zero_grad()
            d_train_loss_d_w += gather_flat_grad(grad(train_loss, model.parameters(), create_graph=True))
            break # TODO (@Mo): Huh?
        optimizer.zero_grad()

        # Compute gradients of the validation loss w.r.t. the weights/hypers
        d_val_loss_d_theta, direct_grad = torch.zeros(num_weights).cuda(), torch.zeros(num_hypers).cuda()
        model.train(), model.zero_grad()
        for batch_idx, (x, y) in enumerate(val_loader):
            val_loss = val_loss_func(x, y)
            optimizer.zero_grad()
            d_val_loss_d_theta += gather_flat_grad(grad(val_loss, model.parameters(), retain_graph=use_reg))
            if use_reg:
                direct_grad += gather_flat_grad(grad(val_loss, get_hyper_train(), allow_unused=True))
                direct_grad[direct_grad != direct_grad] = 0
            break # TODO (@Mo): Huh?

        # Initialize the preconditioner and counter
        preconditioner = d_val_loss_d_theta
        if do_true_inverse:
            hessian = torch.zeros(num_weights, num_weights).cuda()
            for i in range(num_weights):
                hess_row = gather_flat_grad(grad(d_train_loss_d_w[i], model.parameters(), retain_graph=True))
                hessian[i] = hess_row
                # hessian[-i] = hess_row
            '''
            hessian = hessian.t()
            final_hessian = torch.zeros(num_weights, num_weights).cuda()
            for i in range(num_weights):
                final_hessian[-i] = hessian[i]
            hessian = final_hessian
            '''
            # hessian = hessian  #hessian @ hessian
            # chol = torch.cholesky(hessian.view(1, num_weights, num_weights))[0] + 1e-3*torch.eye(num_weights).cuda()
            inv_hessian = torch.pinverse(hessian)
            # inv_hessian = inv_hessian @ inv_hessian
            preconditioner = d_val_loss_d_theta @ inv_hessian
        elif not args.use_cg:
            preconditioner = neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr,
                                                              args.num_neumann_terms, model)
        else:
            def A_vector_multiply_func(vec):
                val = gather_flat_grad(grad(d_train_loss_d_w, model.parameters(),
                                            grad_outputs=vec.view(-1), retain_graph=True))
                # val_2 = gather_flat_grad(grad(d_train_loss_d_w, model.parameters(), grad_outputs=val.view(-1), retain_graph=True))
                # return val_2.view(1, -1, 1)
                return val.view(1, -1, 1)

            if args.num_neumann_terms > 0:
                preconditioner, _ = cg_batch(A_vector_multiply_func, d_val_loss_d_theta.view(1, -1, 1),
                                             maxiter=args.num_neumann_terms)

        if args.save_hessian and do_true_inverse:
            def save_hessian(hessian, name):
                print("saving...")
                fig = plt.figure()
                # clamp_min, clamp_max = -5.0, 5.0
                hessian = torch.tanh(hessian)  # torch.clamp(hessian, clamp_min, clamp_max)
                # hessian[hessian < -1e-4] = clamp_min
                # hessian[hessian > 1e-4] = clamp_max
                # hessian = torch.log(torch.abs(hessian))#torch.clamp(hessian, -1e1, 1e1)
                # hessian[hessian == hessian] = torch.clamp(hessian[hessian == hessian], -10, 10)
                # hessian[torch.abs(hessian) < 1e-0] = 0
                data = hessian.detach().cpu().numpy()
                im = plt.imshow(data, cmap='bwr', interpolation='none', vmin=-1, vmax=1)
                # plt.title(f"Ground Truth: {i}", fontsize=4)
                plt.xticks([])
                plt.yticks([])
                plt.draw()
                cbar = plt.colorbar(im)
                cbar.ax.tick_params(labelsize=20)
                fig.savefig('images/hessian_' + str(name) + '.pdf')
                plt.close(fig)

            '''
            if do_true_inverse:
                name = 'true_inv'
            elif args.use_cg:
                name = 'cg'
            else:
                name = 'neumann_' + str(args.num_neumann_terms)
            '''

            save_hessian(inv_hessian, name='true_inv')
            new_hessian = torch.zeros(inv_hessian.shape).cuda()
            for param_group in optimizer.param_groups:
                cur_step_size = param_group['step_size']
                cur_bias_correction = param_group['bias_correction']
                print(f'size: {cur_step_size}')
                break
            for i in range(10):
                hess_term = torch.eye(inv_hessian.shape[0]).cuda()
                norm_1, norm_2 = torch.norm(torch.eye(inv_hessian.shape[0]).cuda(), p=2), torch.norm(hessian, p=2)
                for j in range(i):
                    # norm_2 = torch.norm(hessian@hessian, p=2)
                    hess_term = hess_term @ (torch.eye(inv_hessian.shape[0]).cuda() - norm_1 / norm_2 * hessian)
                new_hessian += hess_term  # (torch.eye(inv_hessian.shape[0]).cuda() - elementary_lr*0.1*hessian)
                # if (i+1) % 10 == 0 or i == 0:
                save_hessian(new_hessian, name='neumann_' + str(i))
        # conjugate_grad(A_vector_multiply_func, d_val_loss_d_theta)

        # compute d / d lambda (partial Lv / partial w * partial Lt / partial w)
        # = (partial Lv / partial w * partial^2 Lt / (partial w partial lambda))
        indirect_grad = gather_flat_grad(
            grad(d_train_loss_d_w, get_hyper_train(), grad_outputs=preconditioner.view(-1)))
        hypergrad = direct_grad + indirect_grad

        zero_hypergrad(get_hyper_train)
        store_hypergrad(get_hyper_train, -hypergrad)
        # get_hyper_train()[0].grad = hypergrad
        return val_loss, hypergrad.norm()

    init_time = time.time()
    val_loss, val_acc = test(val_loader)
    test_loss, test_acc = test(test_loader)
    if args.do_print:
        print(f"Initial Val Loss: {val_loss, val_acc}")
        print(f"Initial Test Loss: {test_loss, test_acc}")
    iteration = 0
    hypergradient_cos_diff, hypergradient_l2_diff = -1, -1
    for epoch in range(0, args.num_finetune_epochs):
        reg_anneal_epoch = epoch
        xentropy_loss_avg = 0.
        total_val_loss, val_loss = 0., 0.
        correct = 0.
        total = 0.
        weight_norm, grad_norm = .0, .0

        if args.do_print:
            progress_bar = tqdm(train_loader)
        else:
            progress_bar = train_loader
        num_tune_hyper = 45000 / 5000  # 1/5th the val data as train data
        if args.do_simple:
            num_tune_hyper = 1
        hyper_num = 0
        for i, (images, labels) in enumerate(progress_bar):
            if args.do_print:
                progress_bar.set_description('Finetune Epoch ' + str(epoch))

            images, labels = images.cuda(), labels.cuda()
            # pred = model(images)
            optimizer.zero_grad()  # TODO: ADDED
            xentropy_loss, pred = train_loss_func(images, labels)  # F.cross_entropy(pred, labels)
            xentropy_loss.backward()  # TODO: ADDED
            optimizer.step()  # TODO: ADDED
            optimizer.zero_grad()  # TODO: ADDED
            xentropy_loss_avg += xentropy_loss.item()

            if epoch > args.warmup_epochs and args.num_neumann_terms >= 0 and args.load_finetune_checkpoint == '':  # if this is less than 0, then don't do hyper_steps
                if i % num_tune_hyper == 0:
                    cur_lr = 1.0
                    for param_group in optimizer.param_groups:
                        cur_lr = param_group['lr']
                        break
                    train_grad = None  # TODO: ADDED
                    val_loss, grad_norm = hyper_step(cur_lr)

                    if args.do_inverse_compare:
                        approx_hypergradient = get_hyper_train_flat().grad
                        # TODO: Call hyper_step with the true inverse
                        _, _ = hyper_step(cur_lr, do_true_inverse=True)
                        true_hypergradient = get_hyper_train_flat().grad
                        hypergradient_l2_norm = torch.norm(true_hypergradient - approx_hypergradient, p=2)
                        norm_1, norm_2 = torch.norm(true_hypergradient, p=2), torch.norm(approx_hypergradient, p=2)
                        hypergradient_cos_norm = (true_hypergradient @ approx_hypergradient) / (norm_1 * norm_2)
                        hypergradient_cos_diff = hypergradient_cos_norm.item()
                        hypergradient_l2_diff = hypergradient_l2_norm.item()
                        print(f"hypergrad_diff, l2: {hypergradient_l2_norm}, cos: {hypergradient_cos_norm}")
                    # get_hyper_train, model, val_loss_func, val_loader, train_grad, cur_lr, use_reg, args, train_loader, train_loss_func, optimizer)
                    hyper_optimizer.step()

                    weight_norm = get_hyper_train_flat().norm()
                    total_val_loss += val_loss.item()
                    hyper_num += 1

            # Replace the original gradient for the elementary optimizer step.
            '''
            current_index = 0
            flat_train_grad = gather_flat_grad(train_grad)
            for p in model.parameters():
                p_num_params = np.prod(p.shape)
                # if p.grad is not None:
                p.grad = flat_train_grad[current_index: current_index + p_num_params].view(p.shape)
                current_index += p_num_params
            optimizer.step()
            '''

            iteration += 1

            # Calculate running average of accuracy
            if args.do_classification:
                pred = torch.max(pred.data, 1)[1]
                total += labels.size(0)
                correct += (pred == labels.data).sum().item()
                accuracy = correct / total
            else:
                total = 1
                accuracy = 0

            if args.do_print:
                progress_bar.set_postfix(
                    train='%.4f' % (xentropy_loss_avg / (i + 1)),
                    val='%.4f' % (total_val_loss / max(hyper_num, 1)),
                    acc='%.4f' % accuracy,
                    weight='%.3f' % weight_norm,
                    update='%.3f' % grad_norm
                )
            if i % (num_tune_hyper ** 2) == 0:
                if args.use_augment_net:
                    if args.do_diagnostic:
                        save_images(images, labels, augment_net, args)
                if not args.do_simple or args.do_inverse_compare:
                    if not args.do_simple:
                        saver(epoch, model, optimizer, augment_net, reweighting_net, hyper_optimizer, args.save_loc)
                    val_loss, val_acc = test(val_loader)
                    csv_logger.writerow({'epoch': str(epoch),
                                         'train_loss': str(xentropy_loss_avg / (i + 1)), 'train_acc': str(accuracy),
                                         'val_loss': str(val_loss), 'val_acc': str(val_acc),
                                         'test_loss': str(test_loss), 'test_acc': str(test_acc),
                                         'run_time': time.time() - init_time,
                                         'hypergradient_cos_diff': hypergradient_cos_diff,
                                         'hypergradient_l2_diff': hypergradient_l2_diff,
                                         'iteration': iteration})

        if use_scheduler:
            scheduler.step(epoch)
        if use_hyper_scheduler:
            hyper_scheduler.step(epoch)
        train_loss = xentropy_loss_avg / (i + 1)

        if not args.only_print_final_vals:
            val_loss, val_acc = test(val_loader)
            # if val_acc >= 0.99 and accuracy >= 0.99 and epoch >= 50: break
            test_loss, test_acc = test(test_loader)
            tqdm.write('val loss: {:6.4f} | val acc: {:6.4f} | test loss: {:6.4f} | test_acc: {:6.4f}'.format(
                val_loss, val_acc, test_loss, test_acc))

            csv_logger.writerow({'epoch': str(epoch),
                                 'train_loss': str(train_loss), 'train_acc': str(accuracy),
                                 'val_loss': str(val_loss), 'val_acc': str(val_acc),
                                 'test_loss': str(test_loss), 'test_acc': str(test_acc),
                                 'hypergradient_cos_diff': hypergradient_cos_diff,
                                 'hypergradient_l2_diff': hypergradient_l2_diff,
                                 'run_time': time.time() - init_time, 'iteration': iteration})
        else:
            if args.do_print:
                val_loss, val_acc = test(val_loader, do_test_augment=False)
                tqdm.write('val loss: {:6.4f} | val acc: {:6.4f}'.format(val_loss, val_acc))


    val_loss, val_acc = test(val_loader)
    test_loss, test_acc = test(test_loader)
    saver(args.num_finetune_epochs, model, optimizer, augment_net, reweighting_net, hyper_optimizer, args.save_loc)
    return train_loss, accuracy, val_loss, val_acc, test_loss, test_acc


def make_test_arg():
    '''
    Instantiates a set of arguments for a test experiment
    '''

    test_args = make_parser().parse_args()  # make_argss()[0]
    test_args.reg_weight = .5
    test_args.num_neumann_terms = 1
    test_args.use_cg = False
    test_args.seed = 3333
    test_args.do_diagnostic = True
    test_args.data_augmentation = True
    test_args.use_reweighting_net = False
    test_args.use_augment_net = True
    test_args.use_weight_decay = False
    return test_args


def make_inverse_compare_arg():
    test_args = make_parser().parse_args()  # make_argss()[0]
    test_args.reg_weight = 1.0
    # TODO: What am I tuning?

    test_args.seed = 8888
    test_args.do_diagnostic = True
    test_args.data_augmentation = True
    test_args.use_reweighting_net = False
    test_args.use_augment_net = True

    test_args.batch_size = 50
    test_args.train_size, test_args.val_size, test_args.test_size = test_args.batch_size, 1000, 100

    test_args.num_finetune_epochs = 10000
    test_args.model = 'resnet18'  # 'resnet18', 'mlp'
    test_args.use_weight_decay = False  # TODO: Add weight_decay to saveinfo?
    test_args.dataset = 'mnist'  # 'mnist', 'cifar10'  # TODO: Need to add dataset to the save info?

    test_args.num_neumann_terms = -1
    test_args.use_cg = False
    return test_args


def make_val_size_compare(hyperparam, val_prop, data_size, dataset, model):
    '''
    Not sure
    '''

    test_args = make_parser().parse_args()  # make_argss()[0]
    test_args.reg_weight = 0.0
    # TODO: What am I tuning?

    test_args.seed = 9998
    test_args.data_augmentation = False
    test_args.batch_size = data_size  # TODO: Do i want a variable batch size?
    assert 0 <= val_prop <= 1.0, 'Train proportion in [0, 1]'
    test_args.val_prop = val_prop
    test_args.train_size = int(data_size * (1.0 - test_args.val_prop))
    test_args.val_size = int(data_size * (test_args.val_prop))
    if test_args.val_size <= 0:
        test_args.val_size = 1
    if test_args.train_size <= 0:
        test_args.train_size = 1

    # TODO: For long running, boost test_size and num_epochs
    test_args.test_size = -1
    test_args.num_finetune_epochs = 10 # @Mo: 250
    test_args.model = model

    if hyperparam == 'weightDecayParams':
        test_args.use_weight_decay = True
        test_args.weight_decay_all = True
        test_args.use_reweighting_net = False
        test_args.use_augment_net = False
    elif hyperparam == 'weightDecayGlobal':
        test_args.use_weight_decay = True
        test_args.weight_decay_all = False
        test_args.use_reweighting_net = False
        test_args.use_augment_net = False
    elif hyperparam == 'dataAugment':
        test_args.use_weight_decay = False
        test_args.weight_decay_all = False
        test_args.use_reweighting_net = False
        test_args.use_augment_net = True
    elif hyperparam == 'lossReweight':
        test_args.use_weight_decay = False
        test_args.weight_decay_all = False
        test_args.use_reweighting_net = True
        test_args.use_augment_net = False

    test_args.dataset = dataset  # 'mnist', 'cifar10'  # TODO: Need to add dataset to the save info?
    test_args.do_simple = True
    test_args.do_diagnostic = False
    test_args.do_print = False

    test_args.num_neumann_terms = 3
    if test_args.val_size == 1:
        test_args.num_neumann_terms = -1
    test_args.use_cg = False
    return test_args


def run_val_prop_compare(hyperparams, data_sizes, val_props, seeds, datasets, model):
    # TODO (@Mo): Use itertools' product
    for seed in seeds:
        for dataset in datasets:
            for hyperparam in hyperparams:
                for data_size in data_sizes:
                    data_to_save = {'val_losses': [], 'val_accs': [], 'test_losses': [], 'test_accs': [],
                                    'val_losses_re': [], 'val_accs_re': [], 'test_losses_re': [], 'test_accs_re': [],
                                    'info': ''}
                    for val_prop in val_props:
                        print(f"seed:{seed}, dataset:{dataset}, hyperparam:{hyperparam}, data_size:{data_size}, prop:{val_prop}")
                        args = make_val_size_compare(hyperparam, val_prop, data_size, dataset, model)
                        args.seed = seed
                        print("Running experiment:", args)
                        train_loss, accuracy, val_loss, val_acc, test_loss, test_acc = experiment(args)
                        data_to_save['val_losses'] += [val_loss]
                        data_to_save['val_accs'] += [val_acc]
                        data_to_save['test_losses'] += [test_loss]
                        data_to_save['test_accs'] += [test_acc]

                        second_args = make_val_size_compare(hyperparam, 0, data_size, dataset)
                        second_args.seed = seed
                        second_args.num_neumann_terms = -1
                        loc = '/sailhome/motiwari/data-augmentation/implicit-hyper-opt/CG_IFT_test/finetuned_checkpoints/'
                        loc += get_id(args) + '/'
                        loc += 'checkpoint.pt'
                        second_args.load_finetune_checkpoint = loc
                        print("Running experiment:", args)
                        train_loss_re, accuracy_re, val_loss_re, val_acc_re, test_loss_re, test_acc_re = experiment(
                            second_args)
                        data_to_save['val_losses_re'] += [val_loss_re]
                        data_to_save['val_accs_re'] += [val_acc_re]
                        data_to_save['test_losses_re'] += [test_loss_re]
                        data_to_save['test_accs_re'] += [test_acc_re]

                    '''print(f"Data size = {data_size}")
                    print(f"Proportions: {val_props}")
                    print(f"val_losses: {data_to_save['val_losses']}")
                    print(f"val_accuracies: {data_to_save['val_accs']}")
                    print(f"test_losses: {data_to_save['test_losses']}")
                    print(f"test_accuracies: {data_to_save['test_accs']}")'''
                    with open(
                            f'finetuned_checkpoints/dataset:{dataset}_datasize:{data_size}_hyperparam:{hyperparam}_seed:{seed}.pkl',
                            'wb') as f:
                        pickle.dump(data_to_save, f)
                        # TODO: Add the result for experiment that loads final hypers, then puts all data in train
                        #   so change args.num_neumann to -1, val_prop to 0


def graph_val_prop_compare(hyperparams, data_sizes, val_props, seeds, dataset, exclude_sizes=[], retrain=False,
                           do_legend=True, fontsize=12):
    font = {'family': 'Times New Roman'}
    mpl.rc('font', **font)
    mpl.rcParams['legend.fontsize'] = fontsize
    mpl.rcParams['axes.labelsize'] = fontsize
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    mpl.rcParams['axes.grid'] = False
    linewidth = 4
    mpl.rcParams['figure.figsize'] = 5.0, 6.0

    title = f'{dataset.upper()} with Logistic Regression'
    if retrain:
        title += ' with Re-Training'
    linestyles = ['-', ':', '--']
    y_axis = 'test_accs'  # 'test_losses'
    if retrain:
        y_axis += '_re'

    for data_size in data_sizes[::-1]:
        if data_size in exclude_sizes:
            continue

        color = None
        for i, hyperparam in enumerate(hyperparams):
            data_to_graph = []
            for seed in seeds:
                pickle_name = f'finetuned_checkpoints/dataset:{dataset}_datasize:{data_size}_hyperparam:{hyperparam}_seed:{seed}.pkl'
                try:
                    with open(pickle_name, 'rb') as f:
                        data_for_seed = pickle.load(f)
                except FileNotFoundError:
                    print(f"Could not open {pickle_name}")
                    break
                data_to_graph += [data_for_seed[y_axis]]

            label = 'Size:' + str(data_size)

            if hyperparam == 'weightDecayParams' and data_size == data_sizes[0]:
                hyperlabel = 'WD per weight'
                label += ',Hyper:' + hyperlabel
            elif hyperparam == 'weightDecayGlobal' and data_size == data_sizes[0]:
                hyperlabel = hyperparam
                hyperlabel = 'Global WD'
                label += ',Hyper:' + hyperlabel  # + ',Data:' + dataset
            # if seed != seeds[0]: label = None

            if data_size != data_sizes[0] and hyperparam != hyperparams[0]: label = None

            plot = plt.errorbar(val_props,
                                np.mean(data_to_graph, axis=0),
                                .5 * np.std(data_to_graph, axis=0),
                                label=label, linestyle=linestyles[i], c=color, alpha=1.0,
                                linewidth=linewidth)
            color = plot[0].get_color()

    if not retrain:
        plt.axvline(0.1, color='black', linewidth=2.0, alpha=1.0, zorder=1)

    plt.xlabel('Proportion data in valid')
    plt.ylabel('Test Accuracy')
    plt.title(title)
    if do_legend:
        plt.legend(fancybox=True, borderaxespad=0.0, framealpha=.5, fontsize=fontsize,
                   handlelength=1.0)
    if dataset == 'mnist':
        plt.ylim([0.65, 0.9])
    plt.tight_layout()
    retrain_title = ''
    if retrain: retrain_title = '_re'
    plt.savefig(f"images/valProp_vs_testAcc_{dataset}{retrain_title}.pdf")
    plt.clf()

    np.clip(np.round(np.array(val_props) * data_size), 1, 10e32)
    for data_size in data_sizes:
        if data_size in exclude_sizes:
            continue

        color = None
        for i, hyperparam in enumerate(hyperparams):
            data_to_graph = []
            for seed in seeds:
                pickle_name = f'finetuned_checkpoints/dataset:{dataset}_datasize:{data_size}_hyperparam:{hyperparam}_seed:{seed}.pkl'
                try:
                    with open(pickle_name, 'rb') as f:
                        data_for_seed = pickle.load(f)
                except FileNotFoundError:
                    print(f"Could not open {pickle_name}")
                    break
                data_to_graph += [data_for_seed[y_axis]]

            label = 'Size:' + str(data_size) + ',Hyper:' + hyperparam + ',Data:' + dataset

            # if seed != seeds[0]: label = None

            if data_size != data_sizes[0] and hyperparam != hyperparams[0]: label = None
            plot = plt.errorbar(np.clip(np.round(np.array(val_props) * data_size), 1, 10e32),
                                np.mean(data_to_graph, axis=0),
                                .5 * np.std(data_to_graph, axis=0),
                                label=label, linestyle=linestyles[i], c=color, alpha=1.0,
                                linewidth=linewidth)
            color = plot[0].get_color()

    if not retrain:
        plt.axvline(0.1, color='black', linewidth=2.0, alpha=1.0, zorder=1)
    plt.xlabel('Number data in valid')
    plt.ylabel('Test Accuracy')
    plt.xscale('log')
    # plt.title(title)
    if do_legend: plt.legend(fontsize='x-small')
    # plt.ylim([0.4, 0.9])
    plt.savefig(f"images/valNum_vs_testAcc_for_{dataset}{retrain_title}.pdf")
    plt.clf()


# TODO: Make a function to create multiple args to deploy
def do_boston(hyperparam, num_layer, num_neumann):

    test_args = make_parser().parse_args()  # make_argss()[0]
    test_args.reg_weight = 0.0
    # TODO: What am I tuning?

    test_args.seed = 1

    test_args.data_augmentation = False
    test_args.batch_size = 128 * 4

    test_args.model = 'mlp'  # 'resnet18', 'mlp'
    test_args.model += str(num_layer)

    if hyperparam == 'weightDecayParams':
        test_args.use_weight_decay = True
        test_args.weight_decay_all = True
        test_args.use_reweighting_net = False
        test_args.use_augment_net = False
    elif hyperparam == 'weightDecayGlobal':
        test_args.use_weight_decay = True
        test_args.weight_decay_all = False
        test_args.use_reweighting_net = False
        test_args.use_augment_net = False
    elif hyperparam == 'dataAugment':
        test_args.use_weight_decay = False
        test_args.weight_decay_all = False
        test_args.use_reweighting_net = False
        test_args.use_augment_net = True
    elif hyperparam == 'lossReweight':
        test_args.use_weight_decay = False
        test_args.weight_decay_all = False
        test_args.use_reweighting_net = True
        test_args.use_augment_net = False
    test_args.num_layers = num_layer

    test_args.dataset = 'boston'
    test_args.do_classification = False
    test_args.do_simple = True
    test_args.do_diagnostic = False
    test_args.do_print = True

    test_args.num_neumann_terms = num_neumann
    test_args.use_cg = False

    test_args.warmup_epochs = 200
    test_args.num_finetune_epochs = test_args.warmup_epochs + 400
    test_args.do_inverse_compare = True
    test_args.save_hessian = False
    return test_args


def multi_boston_args():
    num_neumanns = [10, 20, 5, 1, 0]
    hyperparams = ['weightDecayParams']  # , 'weightDecayGlobal']
    num_layers = [0]  # , 1]
    argss = []
    for num_neumann in num_neumanns:
        for hyperparam in hyperparams:
            for num_layer in num_layers:
                args = do_boston(hyperparam, num_layer, num_neumann)
                if num_neumann == 10:
                    args.use_cg = True
                    args.num_neumann_terms = 20
                argss += [args]
    return argss


def multi_boston_how_many_steps():
    num_neumanns = range(50)  # [0, 1, 20]
    hyperparams = ['weightDecayParams']  # , 'weightDecayGlobal']
    num_layers = [0]  # , 1]
    argss = []
    for num_neumann in num_neumanns:
        for hyperparam in hyperparams:
            for num_layer in num_layers:
                args = do_boston(hyperparam, num_layer, num_neumann)
                # args.warmup_epochs = 200
                # args.num_finetune_epochs = args.warmup_epochs + 40
                argss += [args]
                cg_args = copy.deepcopy(args)
                cg_args.use_cg = True
                argss += [cg_args]
    return argss


# def curried_run_val(seed):
#     return run_val_prop_compare(hyperparams, data_sizes, val_props, [seed], datasets)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Data Augmentation Example")
    parser.add_argument('--seeds', type=int, default=[1], metavar='S', nargs='+',
                        help='Random seed list (default: [1])')
    parser.add_argument('--hyperparams', type=str, default=['dataAugment'], metavar='H', nargs='+',
                        choices=['weightDecayParams', 'weightDecayGlobal', 'dataAugment', 'lossReweight'],
                        help='Hyperparameter list (default: [dataAugment])')
    parser.add_argument('--data-sizes', type=int, default=[100, 200, 1600], metavar='DSZ', nargs='+',
                        help='Data size list (default: [100, 200, 1600])')
    parser.add_argument('--val-props', type=float, default=[.1, .25, .5, .75, .9], metavar='VP', nargs='+',
                        help='Validation proportion list (default: [.1, .25, .5, .75, .9])')
    parser.add_argument('--datasets', type=str, default=['mnist'], metavar='DS', nargs='+',
                        choices=['cifar10', 'cifar100', 'mnist', 'boston'],
                        help='Choose dataset list (default: [mnist])')
    parser.add_argument('--model', type=str, default='mlp', metavar='DS', nargs='+',
                        choices=['resnet18', 'wideresnet', 'mlp', 'cnn_mlp'],
                        help='Choose a model (default: mlp)')
    return parser.parse_args()


if __name__ == '__main__':
    # experiment(make_test_arg())
    # experiment(make_inverse_compare_arg())
    # experiment(make_val_size_compare(0.5, 100))

    '''
    seeds = [1, 2, 3]
    hyperparams = ['weightDecayParams', 'weightDecayGlobal']
    data_sizes = [50, 100, 250, 500, 1000]  # TODO: Generalize to other variables - ex. hyper choice
    val_props = [.0, .1, .25, .5, .75, .9]
    '''

    # curried_run_val(seeds[0])
    # p = Pool(len(seeds))
    # p.map(curried_run_val, seeds)

    args = parse_args()

    if torch.cuda.is_available():
        print("GPU is available to use in this machine. Using GPU...")
        # TODO: change torch device like this.
        # device = torch.device('cuda')
    else:
        print("GPU is not available to use in this machine. Using CPU...")

    # Uncomment this line to actually run the experiments
    run_val_prop_compare(args.hyperparams, args.data_sizes, args.val_props, args.seeds, args.datasets, args.model)

    for dataset in args.datasets:
        exclude_sizes = []  # 50]
        fontsize = 16
        graph_val_prop_compare(args.hyperparams, args.data_sizes, args.val_props, args.seeds, dataset,
                               exclude_sizes=exclude_sizes, do_legend=False, fontsize=fontsize)
        graph_val_prop_compare(args.hyperparams, args.data_sizes, args.val_props, args.seeds, dataset,
                               exclude_sizes=exclude_sizes, retrain=True, do_legend=True, fontsize=fontsize)

    '''
    # TODO: THE TRICK IS TO TRIAN FOR A LOT OF ITERATIONS!!!
    inverse_argss = multi_boston_args()
    #inverse_argss = multi_boston_how_many_steps()
    for args in inverse_argss:
        experiment(args)

    # inverse_argss = multi_boston_args()
    inverse_argss = multi_boston_how_many_steps()
    for args in inverse_argss:
        experiment(args)
    '''

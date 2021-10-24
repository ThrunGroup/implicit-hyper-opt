import argparse
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
from torch.autograd import grad
from torch.autograd import Variable
import copy
from typing import Callable

# Local imports
from data_loaders import DataLoaders
from models.simple_models import CNN, Net, GaussianDropout
from models.unet import UNet
from utils.util import eval_hessian, eval_jacobian, gather_flat_grad, conjugate_gradiant


def prepare_data(use_cuda, x, y):
    if use_cuda:
        x, y = x.cuda(), y.cuda()
    return x, y


def augmented_loss_func(x, y, model, aug_model=None, reduction='elementwise_mean'):
    assert aug_model
    x = aug_model(x)
    predicted_y = model(x)
    return F.cross_entropy(predicted_y, y, reduction=reduction), predicted_y


def unaugmented_loss_func(x, y, model, aug_model=None, reduction='elementwise_mean'):
    predicted_y = model(x)
    return F.cross_entropy(predicted_y, y, reduction=reduction), predicted_y


def aug_and_unaug_loss_func(x, y, model, aug_model=None, reduction='elementwise_mean'):
    assert aug_model
    loss1, predicted_y1 = augmented_loss_func(x, y, model, aug_model, reduction=reduction)
    loss2, predicted_y2 = unaugmented_loss_func(x, y, model, aug_model, reduction=reduction)
    loss = loss1 + loss2
    predicted_y = torch.cat([predicted_y1, predicted_y2])
    return loss, predicted_y


def train(use_cuda, model, train_loader, optimizer, train_loss_func, aug_model=None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    if aug_model:
        for param in aug_model.parameters():
            param.requires_grad = False
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = prepare_data(use_cuda, x, y)
        loss, _ = train_loss_func(x, y, model, aug_model)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if aug_model:
        for param in aug_model.parameters():
            param.requires_grad = True
    return total_loss / (batch_idx + 1)


def evaluate(use_cuda, model, data_loader):
    total_loss, correct = .0, 0
    with torch.no_grad():
        model.eval()
        data_size = 0
        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = prepare_data(use_cuda, x, y)
            loss, predicted_y = unaugmented_loss_func(x, y, model)
            total_loss += loss.item()
            pred = predicted_y.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
            data_size += x.size(0)
        total_loss /= (batch_idx + 1)
    acc = float(correct) / data_size
    return total_loss, acc


def hyperoptimize(use_cuda, model, aug_model, train_loader, val_loader, optimizer,
                  hyper_optimizer, hessian, neumann_converge_factor=0.1, num_neumann=0):
    hyper_optimizer.zero_grad()
    # Calculate v1 = dLv / dw
    num_weights = sum(p.numel() for p in model.parameters())
    dLv_dw = torch.zeros(num_weights).cuda()
    model.train()
    for batch_idx, (x, y) in enumerate(val_loader):
        optimizer.zero_grad()
        x, y = prepare_data(use_cuda, x, y)
        val_loss, _ = unaugmented_loss_func(x, y, model)
        val_loss_grad = grad(val_loss, model.parameters())
        dLv_dw += gather_flat_grad(val_loss_grad)
    dLv_dw /= (batch_idx + 1)

    # Calculate preconditioner  v1*(inverse Hessian approximation) [orange term in Figure 2]

    # get dw / dlambda
    if hessian == 'identity':  # TODO (@Mo): Warning!!! This may not work for 'direct' hessian. See https://github.com/ThrunGroup/implicit-hyper-opt/blob/weight_decay_overfit/mnist_test.py#L450-L499; the code may not be equivalent
        if hessian == 'identity':
            pre_conditioner = dLv_dw.detach()  # num_neumann = 0
            flat_pre_conditioner = pre_conditioner  # 2*pre_conditioner - args.lr*hessian_term
        model.train()  # train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = prepare_data(use_cuda, x, y)
            train_loss, _ = augmented_loss_func(x, y, model, aug_model=aug_model)
            optimizer.zero_grad()
            dLt_dw = grad(train_loss, model.parameters(), create_graph=True)
            flat_dLt_dw = gather_flat_grad(dLt_dw)
            optimizer.zero_grad()
            flat_dLt_dw.backward(flat_pre_conditioner)  # hyperparams.grad = flat_pre_conditioner * d(dLt/dw)/ dlambda

    elif hessian == 'neumann':
        flat_dLt_dw = torch.zeros(num_weights).cuda()
        model.train(), aug_model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = prepare_data(use_cuda, x, y)
            train_loss, _ = augmented_loss_func(x, y, model, aug_model)
            # TODO (JON): Probably don't recompute - use create_graph and retain_graph?
            optimizer.zero_grad()
            hyper_optimizer.zero_grad()
            dLt_dw = grad(train_loss, model.parameters(), create_graph=True)
            flat_dLt_dw += gather_flat_grad(dLt_dw)
        flat_dLt_dw /= (batch_idx + 1)

        pre_conditioner = dLv_dw.detach()  # dLv_dw is already a flat tensor
        counter = pre_conditioner
        i = 0
        while i < num_neumann:
            old_counter = counter
            hessian_term = gather_flat_grad(
                grad(flat_dLt_dw, model.parameters(), grad_outputs=counter.view(-1), retain_graph=True))
            counter = old_counter - neumann_converge_factor * hessian_term
            pre_conditioner += counter
            i += 1
        pre_conditioner *= neumann_converge_factor

        model.train(), aug_model.train()  # train()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = prepare_data(use_cuda, x, y)
            train_loss, _ = augmented_loss_func(x, y, model, aug_model)
            # TODO (JON): Probably don't recompute - use create_graph and retain_graph?
            # for name, param in model.named_parameters():
            #     print("model----------", name, param.grad)
            #     break
            # for name, param in aug_model.named_parameters():
            #     print("----------", name, param.grad)
            #     break


            optimizer.zero_grad()  # , hyper_optimizer.zero_grad()
            dLt_dw = grad(train_loss, model.parameters(), create_graph=True)
            flat_dLt_dw = gather_flat_grad(dLt_dw)
            optimizer.zero_grad()
            flat_dLt_dw.backward(pre_conditioner)
    for hyper_params in aug_model.parameters():
        hyper_params.grad /= -(batch_idx + 1)





    flatten_hyper_params = gather_flat_grad(aug_model.parameters())
    flatten_hyper_grads = torch.cat([p.grad.view(-1) for p in aug_model.parameters()])

    print("weight={}, update={}".format(flatten_hyper_params.norm(), flatten_hyper_grads.norm()))
    hyper_optimizer.step()  # TODO (@Mo): Understand hyper_optimizer.step(), get_hyper_train(), kfac_opt.fake_step()?
    optimizer.zero_grad()
    hyper_optimizer.zero_grad()
    return flatten_hyper_params, flatten_hyper_grads


def get_optimizer(name: str) -> Callable:
    if name == 'adam':
        return torch.optim.Adam
    elif name == 'rmsprop':
        return torch.optim.RMSprop
    elif name == 'adagrad':
        return torch.optim.Adagrad
    elif name == 'sgd':
        return torch.optim.SGD


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)


def experiment(config: dict = None, use_wandb: bool = True):
    """

    :param config contains dataset, model, aug_model, num_layers, dropout, fc_shape, wf, depth, use_cuda, loss_criterion
    hessian, neumann_converge_factor, num_neumann, optimizer, hyper_optimizer, epochs, hepochs, model_lr, hyper_model_lr
    batch_size, datasize, train_prop, test_size, seed, patience
    """

    if use_wandb:
        wandb.init()
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

    ###############################################################################
    # Setup dataset
    ###############################################################################
    set_seed(config.seed)
    train_size = int(config.datasize * config.train_prop)
    val_size = config.datasize - train_size
    batch_size = min(config.batch_size, config.datasize)
    train_loader, val_loader, test_loader = DataLoaders.get_data_loaders(dataset=config.dataset,
                                                                         batch_size=batch_size,
                                                                         train_size=train_size,
                                                                         val_size=val_size,
                                                                         test_size=config.test_size,
                                                                         num_train=config.datasize,
                                                                         data_augment=True)
    if config.dataset == 'mnist':
        in_channel = 1
        imsize = 28
        num_classes = 10
    else:
        in_channel = 3
        imsize = 32
        if config.dataset == 'cifar10':
            num_classes = 10
        elif config.dataset == 'cifar100':
            num_classes = 100
        else:
            raise Exception("bad dataset")

    ###############################################################################
    # Setup model
    ###############################################################################
    if config.model == "mlp":
        model = Net(config.num_layers, config.dropout, imsize, in_channel, weight_decay=0,
                    num_classes=num_classes)
        # For later test
        unaugmented_model = Net(config.num_layers, config.dropout, imsize, in_channel, weight_decay=0,
                                num_classes=num_classes)
        augmented_model = Net(config.num_layers, config.dropout, imsize, in_channel, weight_decay=0,
                              num_classes=num_classes)
    elif config.model == "cnn":
        model = CNN(config.num_layers, config.dropout, config.fc_shape, 0,
                    in_channel, imsize, do_alexnet=False,
                    num_classes=num_classes)
        # For later test
        unaugmented_model = CNN(config.num_layers, config.dropout, config.fc_shape, 0,
                                in_channel, imsize, do_alexnet=False,
                                num_classes=num_classes)
        augmented_model = CNN(config.num_layers, config.dropout, config.fc_shape, 0,
                              in_channel, imsize, do_alexnet=False,
                              num_classes=num_classes)
    elif config.model == "alexnet":
        model = CNN(config.num_layers, config.dropout, config.fc_shape, 0, in_channel, imsize, do_alexnet=True,
                    num_classes=num_classes)
        # For later test
        unaugmented_model = CNN(config.num_layers, config.dropout, config.fc_shape, 0, in_channel, imsize,
                                do_alexnet=True, num_classes=num_classes)
        augmented_model = CNN(config.num_layers, config.dropout, config.fc_shape, 0, in_channel, imsize,
                              do_alexnet=True, num_classes=num_classes)
    else:
        raise Exception("bad model")

    if config.aug_model == "unet":
        aug_model = UNet(in_channels=in_channel, n_classes=in_channel, wf=config.wf, padding=1, depth=config.depth)
    else:
        raise Exception("bad augmentation model")

    use_cuda = torch.cuda.is_available() and config.use_cuda
    if use_cuda:
        model = model.cuda()
        aug_model = aug_model.cuda()
    model_state_dict = copy.deepcopy(model.state_dict())
    best_aug_model_state_dict = copy.deepcopy(aug_model.state_dict())

    ###############################################################################
    # Setup optimizer and hyper_optimizer
    ###############################################################################
    optimizer = get_optimizer(config.optimizer)(model.parameters(), config.model_lr)
    hyper_optimizer = get_optimizer(config.hyper_optimizer)(aug_model.parameters(), config.hyper_model_lr)

    ###############################################################################
    # Perform the training
    ###############################################################################
    patience = 0
    best_val_loss = float("inf")
    for epoch_h in range(1, config.hepochs + 1):
        if epoch_h != 1:
            _1, _2 = hyperoptimize(config.use_cuda, model, aug_model, train_loader, val_loader, optimizer,
                               hyper_optimizer, config.hessian, config.neumann_converge_factor,
                               config.num_neumann)
        model.load_state_dict(model_state_dict)  # @Jay: Initialize model for every hyper epochs -- needs to be
        # discussed
        for epoch in range(1, config.epochs+1):
            loss = train(config.use_cuda, model, train_loader, optimizer, augmented_loss_func, aug_model)
            if loss < config.loss_criterion:
                break
            if epoch % 100 == 0 or epoch == 1:
                print('step', epoch, ': ', loss)

        val_loss, val_acc = evaluate(config.use_cuda, model, val_loader)
        print("hyper_epoch", epoch_h, "val_loss", val_loss, "val_acc", val_acc)

        # log the results
        if use_wandb:
            wandb.log({"hyper_epoch": epoch_h, "val_loss": val_loss, "val_acc": val_acc})

        # Early stopping
        if best_val_loss < val_loss:
            best_val_loss = val_loss
            patience += 1
            print(patience)
        else:
            patience = 0
            best_aug_model_state_dict = copy.deepcopy(aug_model.state_dict())
        if patience >= config.patience:
            break

    # Test if the augmentation network is well trained
    aug_model.load_state_dict(best_aug_model_state_dict)
    unaugmented_model.load_state_dict(model_state_dict)
    augmented_model.load_state_dict(model_state_dict)

    if use_cuda and torch.cuda.is_available():
        unaugmented_model = unaugmented_model.cuda()
        augmented_model = augmented_model.cuda()

    unaug_optimizer = get_optimizer(config.optimizer)(unaugmented_model.parameters(), config.model_lr)
    aug_optimizer = get_optimizer(config.optimizer)(augmented_model.parameters(), config.model_lr)


    unaug_best_acc = .0
    aug_best_acc = .0
    for step in range(1, config.epochs + 1):
        # train model with training dataset
        unaug_loss = train(config.use_cuda, unaugmented_model, train_loader, unaug_optimizer, unaugmented_loss_func,
                           aug_model=None)
        # train model with training dataset U augmented training dataset with aug_model
        aug_loss = train(config.use_cuda, augmented_model, train_loader, aug_optimizer, aug_and_unaug_loss_func,
                         aug_model)

        # Evaluate model accuracy for validation dataset
        unaug_loss, unaug_acc = evaluate(config.use_cuda, unaugmented_model, val_loader)
        aug_loss, aug_acc = evaluate(config.use_cuda, augmented_model, val_loader)

        if unaug_acc >= unaug_best_acc:
            _, unaug_test_acc = evaluate(use_cuda, unaugmented_model, test_loader)
            unaug_best_acc = unaug_acc
        if aug_acc >= aug_best_acc:
            _, aug_test_acc = evaluate(use_cuda, augmented_model, test_loader)
            aug_best_acc = aug_acc

    print("unaug_test_acc", unaug_test_acc, "aug_test_acc", aug_test_acc, "diff", aug_test_acc - unaug_test_acc)
    if use_wandb:
        wandb.log({"unaug_test_acc": unaug_test_acc, "aug_test_acc": aug_test_acc,
                   "diff": aug_test_acc - unaug_test_acc})


if __name__ == '__main__':
    """ "config" contains dataset, model, aug_model, num_layers, dropout, fc_shape, wf, depth, use_cuda, loss_criterion
   hessian, neumann_converge_factor, num_neumann, optimizer, hyper_optimizer, epochs, hepochs, model_lr, hyper_model_lr
   batch_size, datasize, train_prop, test_size, seed, patience   
   """

    # To check if experiment(config) runs well
    config = argparse.ArgumentParser(description='PyTorch MNIST Example')
    config.dataset = 'mnist'
    config.model = 'mlp'
    config.aug_model = 'unet'
    config.num_layers = 2
    config.dropout = 0.14439652649920856
    config.fc_shape = 800
    config.wf = 2
    config.depth = 1
    config.use_cuda = True
    config.loss_criterion = 1e-7
    config.hessian = 'neumann'
    config.neumann_converge_factor = 1e-4
    config.num_neumann = 3
    config.optimizer = 'adam'
    config.hyper_optimizer = 'rmsprop'
    config.epochs = 10
    config.hepochs = 50
    config.model_lr = 1e-4
    config.hyper_model_lr = 1e-2
    config.batch_size = 11
    config.datasize = 232
    config.train_prop = 0.7349830546612787
    config.test_size = -1
    config.seed = 28
    config.patience = 5

    print(torch.cuda.is_available())
    experiment(config, use_wandb=False)

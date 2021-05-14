import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad

from utils.util import gather_flat_grad, save_hessian


def hyper_step(elementary_lr, args, model, train_loader, val_loader, augment_net, reweighting_net, optimizer, use_reg, reg_anneal_epoch, stop_reg_epoch, graph_iter, device, do_true_inverse=False):
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
    zero_hypergrad(get_hyper_train, args, model, augment_net, reweighting_net)
    num_weights, num_hypers = sum(p.numel() for p in model.parameters()), sum(p.numel() for p in get_hyper_train(args, model, augment_net, reweighting_net))
    print(f"num_weights : {num_weights}, num_hypers : {num_hypers}")

    # d_train_loss_d_w = gather_flat_grad(d_train_loss_d_w)  # TODO: Commented this out!
    d_train_loss_d_w = torch.zeros(num_weights).to(device)
    model.train(), model.zero_grad()
    for batch_idx, (x, y) in enumerate(train_loader):
        train_loss, _, graph_iter = train_loss_func(x, y, args, model, augment_net, reweighting_net, graph_iter, device)
        optimizer.zero_grad()
        d_train_loss_d_w += gather_flat_grad(grad(train_loss, model.parameters(), create_graph=True))
        break # TODO (@Mo): Huh?
    optimizer.zero_grad()

    # Compute gradients of the validation loss w.r.t. the weights/hypers
    d_val_loss_d_theta, direct_grad = torch.zeros(num_weights).to(device), torch.zeros(num_hypers).to(device)
    model.train(), model.zero_grad()
    for batch_idx, (x, y) in enumerate(val_loader):
        val_loss = val_loss_func(x, y, args, model, augment_net, use_reg, reg_anneal_epoch, stop_reg_epoch, device)
        optimizer.zero_grad()
        d_val_loss_d_theta += gather_flat_grad(grad(val_loss, model.parameters(), retain_graph=use_reg))
        if use_reg:
            direct_grad += gather_flat_grad(grad(val_loss, get_hyper_train(), allow_unused=True))
            direct_grad[direct_grad != direct_grad] = 0
        break # TODO (@Mo): Huh?

    # Initialize the preconditioner and counter
    preconditioner = d_val_loss_d_theta
    if do_true_inverse:
        hessian = torch.zeros(num_weights, num_weights).to(device)
        for i in range(num_weights):
            hess_row = gather_flat_grad(grad(d_train_loss_d_w[i], model.parameters(), retain_graph=True))
            hessian[i] = hess_row
            # hessian[-i] = hess_row
        '''
        hessian = hessian.t()
        final_hessian = torch.zeros(num_weights, num_weights).to(device)
        for i in range(num_weights):
            final_hessian[-i] = hessian[i]
        hessian = final_hessian
        '''
        # hessian = hessian  #hessian @ hessian
        # chol = torch.cholesky(hessian.view(1, num_weights, num_weights))[0] + 1e-3*torch.eye(num_weights).to(device)
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
        '''
        if do_true_inverse:
            name = 'true_inv'
        elif args.use_cg:
            name = 'cg'
        else:
            name = 'neumann_' + str(args.num_neumann_terms)
        '''

        save_hessian(inv_hessian, name='true_inv')
        new_hessian = torch.zeros(inv_hessian.shape).to(device)
        for param_group in optimizer.param_groups:
            cur_step_size = param_group['step_size']
            cur_bias_correction = param_group['bias_correction']
            print(f'size: {cur_step_size}')
            break
        for i in range(10):
            hess_term = torch.eye(inv_hessian.shape[0]).to(device)
            norm_1, norm_2 = torch.norm(torch.eye(inv_hessian.shape[0]).to(device), p=2), torch.norm(hessian, p=2)
            for j in range(i):
                # norm_2 = torch.norm(hessian@hessian, p=2)
                hess_term = hess_term @ (torch.eye(inv_hessian.shape[0]).to(device) - norm_1 / norm_2 * hessian)
            new_hessian += hess_term  # (torch.eye(inv_hessian.shape[0]).to(device) - elementary_lr*0.1*hessian)
            # if (i+1) % 10 == 0 or i == 0:
            save_hessian(new_hessian, name='neumann_' + str(i))
    # conjugate_grad(A_vector_multiply_func, d_val_loss_d_theta)

    # compute d / d lambda (partial Lv / partial w * partial Lt / partial w)
    # = (partial Lv / partial w * partial^2 Lt / (partial w partial lambda))
    indirect_grad = gather_flat_grad(
        grad(d_train_loss_d_w, get_hyper_train(args, model, augment_net, reweighting_net), grad_outputs=preconditioner.view(-1)))
    hypergrad = direct_grad + indirect_grad

    zero_hypergrad(get_hyper_train, args, model, augment_net, reweighting_net)
    store_hypergrad(get_hyper_train, -hypergrad, args, model, augment_net, reweighting_net)
    # get_hyper_train()[0].grad = hypergrad
    return val_loss, hypergrad.norm(), graph_iter


def zero_hypergrad(get_hyper_train, args, model, augment_net, reweighting_net):
    current_index = 0
    for p in get_hyper_train(args, model, augment_net, reweighting_net):
        p_num_params = np.prod(p.shape)
        if p.grad is not None:
            p.grad = p.grad * 0 # TODO (@Mo): Is this necessary? Could just set to 0?
        current_index += p_num_params


def get_hyper_train(args, model, augment_net, reweighting_net):
    # return torch.cat([p.view(-1) for p in augment_net.parameters()])
    if args.use_augment_net and args.use_reweighting_net:
        return list(augment_net.parameters()) + list(reweighting_net.parameters())
    elif args.use_augment_net:
        return augment_net.parameters()
    elif args.use_reweighting_net:
        return reweighting_net.parameters()
    elif args.use_weight_decay:
        return [model.weight_decay]


def get_hyper_train_flat(args, model, augment_net, reweighting_net):
    if args.use_augment_net and args.use_reweighting_net:
        return torch.cat([torch.cat([p.view(-1) for p in augment_net.parameters()]),
                          torch.cat([p.view(-1) for p in reweighting_net.parameters()])])
    elif args.use_reweighting_net:
        return torch.cat([p.view(-1) for p in reweighting_net.parameters()])
    elif args.use_augment_net:
        return torch.cat([p.view(-1) for p in augment_net.parameters()])
    elif args.use_weight_decay:
        return model.weight_decay  # TODO: This correct?


def train_loss_func(x, y, args, model, augment_net, reweighting_net, graph_iter, device):
    x, y = x.to(device), y.to(device)
    reg = 0.

    if args.use_augment_net and (args.num_neumann_terms >= 0 or args.load_finetune_checkpoint != ''):
        # augment_net = augment_net.to(f'cuda:{model.device_ids[0]}')
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
    return final_loss, pred, graph_iter


def val_loss_func(x, y, args, model, augment_net, use_reg, reg_anneal_epoch, stop_reg_epoch, device):
    x, y = x.to(device), y.to(device)
    pred = model(x)
    if args.do_classification:
        xentropy_loss = F.cross_entropy(pred, y)
    else:
        xentropy_loss = F.mse_loss(pred, y)

    reg = 0
    if args.use_augment_net:
        if use_reg:
            num_sample = 10
            xs = torch.zeros(num_sample, x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(device)
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


def neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, elementary_lr, num_neumann_terms, model):
    preconditioner = d_val_loss_d_theta.detach()
    counter = preconditioner
    # Do the fixed point iteration to approximate the vector-inverseHessian product
    for i in range(num_neumann_terms):
        old_counter = counter
        # This increments counter to counter * (I - hessian) = counter - counter * hessian
        hessian_term = gather_flat_grad(grad(d_train_loss_d_w, model.parameters(), grad_outputs=counter.contiguous().view(-1), retain_graph=True))
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


def store_hypergrad(get_hyper_train, total_d_val_loss_d_lambda, args, model, augment_net, reweighting_net):
    """
    Updates the hypergradients and the number of hyperparameters?
    """
    current_index = 0
    for p in get_hyper_train(args, model, augment_net, reweighting_net):
        p_num_params = np.prod(p.shape)
        p.grad = total_d_val_loss_d_lambda[current_index:current_index + p_num_params].view(p.shape)
        current_index += p_num_params

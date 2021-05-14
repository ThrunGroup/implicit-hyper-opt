import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from constants import DATASET_BOSTON
from data_loaders import DataLoaders
from hyper_train import get_hyper_train, train_loss_func, hyper_step, get_hyper_train_flat
from train_augment_net_graph import save_images
from train_augment_net_multiple import load_logger, get_id
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from utils.model_loader import ModelLoader
from utils.util import save_models


def experiment(args, device):
    if args.do_print:
        print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load the baseline model
    args.load_baseline_checkpoint = None  # '/h/lorraine/PycharmProjects/CG_IFT_test/baseline_checkpoints/cifar10_resnet18_sgdm_lr0.1_wd0.0005_aug1.pt'
    args.load_finetune_checkpoint = ''  # TODO: Make it load the augment net if this is provided

    train_loader, val_loader, test_loader = DataLoaders.get_data_loaders(dataset=args.dataset,
                                                                         batch_size=args.batch_size,
                                                                         train_size=args.train_size,
                                                                         val_size=args.val_size,
                                                                         test_size=args.test_size,
                                                                         num_train=50000,
                                                                         data_augment=args.data_augmentation)
    model_loader = ModelLoader(args, device)
    model, augment_net, reweighting_net, checkpoint = model_loader.get_models()

    # Load the logger
    csv_logger, test_id = load_logger(args)
    args.save_loc = './finetuned_checkpoints/' + get_id(args)

    # Setup the optimizers
    if args.load_baseline_checkpoint is not None:
        args.lr = args.lr * 0.2 * 0.2 * 0.2 # TODO (@Mo): oh my god no
    if args.use_weight_decay:
        # optimizer = optim.Adam(model.parameters(), lr=1e-3)
        args.wdecay = 0

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wdecay)
    if args.dataset == DATASET_BOSTON:
        optimizer = optim.Adam(model.parameters())
    use_scheduler = False
    if not args.do_simple:
        use_scheduler = True
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)  # [60, 120, 160]
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    use_hyper_scheduler = False
    hyper_optimizer = optim.RMSprop(get_hyper_train(args, model, augment_net, reweighting_net))
    if not args.do_simple:
        hyper_optimizer = optim.SGD(get_hyper_train(args, model, augment_net, reweighting_net), lr=args.lr, momentum=0.9, nesterov=True)
        use_hyper_scheduler = True
    hyper_scheduler = MultiStepLR(hyper_optimizer, milestones=[40, 100, 140], gamma=0.2)

    graph_iter = 0
    use_reg = args.use_augment_net and not args.use_reweighting_net
    reg_anneal_epoch = 0
    stop_reg_epoch = 200
    if args.reg_weight == 0:
        use_reg = False

    init_time = time.time()
    val_loss, val_acc = test(val_loader, args, model, augment_net, device)
    test_loss, test_acc = test(test_loader, args, model, augment_net, device)
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

            images, labels = images.to(device), labels.to(device)
            # pred = model(images)
            optimizer.zero_grad()  # TODO: ADDED
            xentropy_loss, pred, graph_iter = train_loss_func(images, labels, args, model, augment_net, reweighting_net, graph_iter, device)  # F.cross_entropy(pred, labels)
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
                    val_loss, grad_norm, graph_iter = hyper_step(cur_lr, args, model, train_loader, val_loader, augment_net, reweighting_net, optimizer, use_reg, reg_anneal_epoch, stop_reg_epoch, graph_iter, device)

                    if args.do_inverse_compare:
                        approx_hypergradient = get_hyper_train_flat(args, model, augment_net, reweighting_net).grad
                        # TODO: Call hyper_step with the true inverse
                        _, _, graph_iter = hyper_step(cur_lr, args, model, train_loader, val_loader, augment_net, reweighting_net, optimizer, use_reg, reg_anneal_epoch, stop_reg_epoch, graph_iter, device, do_true_inverse=True)
                        true_hypergradient = get_hyper_train_flat(args, model, augment_net, reweighting_net).grad
                        hypergradient_l2_norm = torch.norm(true_hypergradient - approx_hypergradient, p=2)
                        norm_1, norm_2 = torch.norm(true_hypergradient, p=2), torch.norm(approx_hypergradient, p=2)
                        hypergradient_cos_norm = (true_hypergradient @ approx_hypergradient) / (norm_1 * norm_2)
                        hypergradient_cos_diff = hypergradient_cos_norm.item()
                        hypergradient_l2_diff = hypergradient_l2_norm.item()
                        print(f"hypergrad_diff, l2: {hypergradient_l2_norm}, cos: {hypergradient_cos_norm}")
                    # get_hyper_train, model, val_loss_func, val_loader, train_grad, cur_lr, use_reg, args, train_loader, train_loss_func, optimizer)
                    hyper_optimizer.step()

                    weight_norm = get_hyper_train_flat(args, model, augment_net, reweighting_net).norm()
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
                        save_models(epoch, model, optimizer, augment_net, reweighting_net, hyper_optimizer, args.save_loc)
                    val_loss, val_acc = test(val_loader, args, model, augment_net, device)
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
            val_loss, val_acc = test(val_loader, args, model, augment_net, device)
            # if val_acc >= 0.99 and accuracy >= 0.99 and epoch >= 50: break
            test_loss, test_acc = test(test_loader, args, model, augment_net, device)
            tqdm.write('epoch: {:d} | val loss: {:6.4f} | val acc: {:6.4f} | test loss: {:6.4f} | test_acc: {:6.4f}'.format(
                epoch, val_loss, val_acc, test_loss, test_acc))

            csv_logger.writerow({'epoch': str(epoch),
                                 'train_loss': str(train_loss), 'train_acc': str(accuracy),
                                 'val_loss': str(val_loss), 'val_acc': str(val_acc),
                                 'test_loss': str(test_loss), 'test_acc': str(test_acc),
                                 'hypergradient_cos_diff': hypergradient_cos_diff,
                                 'hypergradient_l2_diff': hypergradient_l2_diff,
                                 'run_time': time.time() - init_time, 'iteration': iteration})
        elif args.do_print:
            val_loss, val_acc = test(val_loader, args, model, augment_net, device, do_test_augment=False)
            tqdm.write('val loss: {:6.4f} | val acc: {:6.4f}'.format(val_loss, val_acc))


    val_loss, val_acc = test(val_loader, args, model, augment_net, device)
    test_loss, test_acc = test(test_loader, args, model, augment_net, device)
    save_models(args.num_finetune_epochs, model, optimizer, augment_net, reweighting_net, hyper_optimizer, args.save_loc)
    return train_loss, accuracy, val_loss, val_acc, test_loss, test_acc


def test(loader, args, model, augment_net, device, do_test_augment=True, num_augment=5):
    # model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct, total = 0., 0.
    losses = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            pred = model(images)
            if do_test_augment:
                if args.use_augment_net and (args.num_neumann_terms >= 0 or args.load_finetune_checkpoint != ''):
                    shape_0, shape_1 = pred.shape[0], pred.shape[1]
                    pred = pred.view(1, shape_0, shape_1)  # Batch size, num_classes
                    for _ in range(num_augment):
                        pred = torch.cat((pred, model(augment_net(images)).view(1, shape_0, shape_1)))
                    pred = torch.mean(pred, dim=0)
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

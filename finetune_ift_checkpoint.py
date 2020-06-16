import os
import ipdb
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd import Variable

# Local imports
import data_loaders
from csv_logger import CSVLogger
from resnet import ResNet18
from wide_resnet import WideResNet
from unet import UNet


def experiment():
    parser = argparse.ArgumentParser(description='CNN Hyperparameter Fine-tuning')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'],
                        help='Choose a dataset')
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'wideresnet'],
                        help='Choose a model')
    parser.add_argument('--num_finetune_epochs', type=int, default=200,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgdm',
                        help='Choose an optimizer')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Mini-batch size')
    parser.add_argument('--data_augmentation', action='store_true', default=True,
                        help='Whether to use data augmentation')
    parser.add_argument('--wdecay', type=float, default=5e-4,
                        help='Amount of weight decay')
    parser.add_argument('--load_checkpoint', type=str,
                        help='Path to pre-trained checkpoint to load and finetune')
    parser.add_argument('--save_dir', type=str, default='finetuned_checkpoints',
                        help='Save directory for the fine-tuned checkpoint')
    args = parser.parse_args()
    args.load_checkpoint = '/h/lorraine/PycharmProjects/CG_IFT_test/baseline_checkpoints/cifar10_resnet18_sgdm_lr0.1_wd0.0005_aug0.pt'

    if args.dataset == 'cifar10':
        num_classes = 10
        train_loader, val_loader, test_loader = data_loaders.load_cifar10(args.batch_size, val_split=True,
                                                                          augmentation=args.data_augmentation)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_loader, val_loader, test_loader = data_loaders.load_cifar100(args.batch_size, val_split=True,
                                                                           augmentation=args.data_augmentation)

    if args.model == 'resnet18':
        cnn = ResNet18(num_classes=num_classes)
    elif args.model == 'wideresnet':
        cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    test_id = '{}_{}_{}_lr{}_wd{}_aug{}'.format(args.dataset, args.model, args.optimizer, args.lr, args.wdecay,
                                                int(args.data_augmentation))
    filename = os.path.join(args.save_dir, test_id + '.csv')
    csv_logger = CSVLogger(
        fieldnames=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc'],
        filename=filename)

    checkpoint = torch.load(args.load_checkpoint)
    init_epoch = checkpoint['epoch']
    cnn.load_state_dict(checkpoint['model_state_dict'])
    model = cnn.cuda()
    model.train()

    args.hyper_train = 'augment'  # 'all_weight'  # 'weight'

    def init_hyper_train(model):
        """

        :return:
        """
        init_hyper = None
        if args.hyper_train == 'weight':
            init_hyper = np.sqrt(args.wdecay)
            model.weight_decay = Variable(torch.FloatTensor([init_hyper]).cuda(), requires_grad=True)
            model.weight_decay = model.weight_decay.cuda()
        elif args.hyper_train == 'all_weight':
            num_p = sum(p.numel() for p in model.parameters())
            weights = np.ones(num_p) * np.sqrt(args.wdecay)
            model.weight_decay = Variable(torch.FloatTensor(weights).cuda(), requires_grad=True)
            model.weight_decay = model.weight_decay.cuda()
        model = model.cuda()
        return init_hyper

    if args.hyper_train == 'augment':  # Dont do inside the prior function, else scope is wrong
        augment_net = UNet(in_channels=3,
                           n_classes=3,
                           depth=5,
                           wf=6,
                           padding=True,
                           batch_norm=False,
                           up_mode='upconv')  # TODO(PV): Initialize UNet properly
        augment_net = augment_net.cuda()

    def get_hyper_train():
        """

        :return:
        """
        if args.hyper_train == 'weight' or args.hyper_train == 'all_weight':
            return [model.weight_decay]
        if args.hyper_train == 'augment':
            return augment_net.parameters()

    def get_hyper_train_flat():
        return torch.cat([p.view(-1) for p in get_hyper_train()])

    # TODO: Check this size

    init_hyper_train(model)

    if args.hyper_train == 'all_weight':
        wdecay = 0.0
    else:
        wdecay = args.wdecay
    optimizer = optim.SGD(model.parameters(), lr=args.lr * 0.2 * 0.2, momentum=0.9, nesterov=True,
                          weight_decay=wdecay)  # args.wdecay)
    # print(checkpoint['optimizer_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = MultiStepLR(optimizer, milestones=[60, 120], gamma=0.2)  # [60, 120, 160]
    hyper_optimizer = torch.optim.Adam(get_hyper_train(), lr=1e-3)  # try 0.1 as lr

    # Set random regularization hyperparameters
    # data_augmentation_hparams = {}  # Random values for hue, saturation, brightness, contrast, rotation, etc.
    if args.dataset == 'cifar10':
        num_classes = 10
        train_loader, val_loader, test_loader = data_loaders.load_cifar10(args.batch_size, val_split=True,
                                                                          augmentation=args.data_augmentation)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_loader, val_loader, test_loader = data_loaders.load_cifar100(args.batch_size, val_split=True,
                                                                           augmentation=args.data_augmentation)

    def test(loader):
        model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
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

    def prepare_data(x, y):
        """

        :param x:
        :param y:
        :return:
        """
        x, y = x.cuda(), y.cuda()

        # x, y = Variable(x), Variable(y)
        return x, y

    def train_loss_func(x, y):
        """

        :param x:
        :param y:
        :return:
        """
        x, y = prepare_data(x, y)

        reg_loss = 0.0
        if args.hyper_train == 'weight':
            pred = model(x)
            xentropy_loss = F.cross_entropy(pred, y)
            # print(f"weight_decay: {torch.exp(model.weight_decay).shape}")
            for p in model.parameters():
                # print(f"weight_decay: {torch.exp(model.weight_decay).shape}")
                # print(f"shape: {p.shape}")
                reg_loss = reg_loss + .5 * (model.weight_decay ** 2) * torch.sum(p ** 2)
                # print(f"reg_loss: {reg_loss}")
        elif args.hyper_train == 'all_weight':
            pred = model(x)
            xentropy_loss = F.cross_entropy(pred, y)
            count = 0
            for p in model.parameters():
                reg_loss = reg_loss + .5 * torch.sum(
                    (model.weight_decay[count: count + p.numel()] ** 2) * torch.flatten(p ** 2))
                count += p.numel()
        elif args.hyper_train == 'augment':
            augmented_x = augment_net(x)
            pred = model(augmented_x)
            xentropy_loss = F.cross_entropy(pred, y)
        return xentropy_loss + reg_loss, pred

    def val_loss_func(x, y):
        """

        :param x:
        :param y:
        :return:
        """
        x, y = prepare_data(x, y)
        pred = model(x)
        xentropy_loss = F.cross_entropy(pred, y)
        return xentropy_loss

    for epoch in range(init_epoch, init_epoch + args.num_finetune_epochs):
        xentropy_loss_avg = 0.
        total_val_loss = 0.
        correct = 0.
        total = 0.

        progress_bar = tqdm(train_loader)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description('Finetune Epoch ' + str(epoch))

            # TODO: Take a hyperparameter step here
            optimizer.zero_grad(), hyper_optimizer.zero_grad()
            val_loss, weight_norm, grad_norm = hyper_step(1, 1, get_hyper_train, get_hyper_train_flat,
                                                                model, val_loss_func,
                                                                val_loader, train_loss_func, train_loader,
                                                                hyper_optimizer)
            # del val_loss
            # print(f"hyper: {get_hyper_train()}")

            images, labels = images.cuda(), labels.cuda()
            # pred = model(images)
            # xentropy_loss = F.cross_entropy(pred, labels)
            xentropy_loss, pred = train_loss_func(images, labels)

            optimizer.zero_grad(), hyper_optimizer.zero_grad()
            xentropy_loss.backward()
            optimizer.step()

            xentropy_loss_avg += xentropy_loss.item()

            # Calculate running average of accuracy
            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels.data).sum().item()
            accuracy = correct / total

            progress_bar.set_postfix(
                train='%.5f' % (xentropy_loss_avg / (i + 1)),
                val='%.4f' % (total_val_loss / (i + 1)),
                acc='%.4f' % accuracy,
                weight='%.2f' % weight_norm,
                update='%.3f' % grad_norm)

        val_loss, val_acc = test(val_loader)
        test_loss, test_acc = test(test_loader)
        tqdm.write('val loss: {:6.4f} | val acc: {:6.4f} | test loss: {:6.4f} | test_acc: {:6.4f}'.format(
            val_loss, val_acc, test_loss, test_acc))

        scheduler.step(epoch)

        row = {'epoch': str(epoch),
               'train_loss': str(xentropy_loss_avg / (i + 1)), 'train_acc': str(accuracy),
               'val_loss': str(val_loss), 'val_acc': str(val_acc),
               'test_loss': str(test_loss), 'test_acc': str(test_acc)}
        csv_logger.writerow(row)


"""def hyper_step(train_batch_num, val_batch_num, get_hyper_train, unshaped_get_hyper_train, model, val_loss_func,
               val_loader, train_loss_func, train_loader, hyper_optimizer):
    '''
    
    :param train_batch_num: 
    :param val_batch_num: 
    :param get_hyper_train: 
    :param unshaped_get_hyper_train: 
    :param model: 
    :param val_loss_func: 
    :param val_loader: 
    :param train_loss_func: 
    :param train_loader: 
    :param hyper_optimizer: 
    :return: 
    '''
    from util import gather_flat_grad
    train_batch_num -= 1
    val_batch_num -= 1
    '''import gc
    print("Printing objects...")
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass
    print("Done printing objects.")'''

    # set up placeholder for the partial derivative in each batch
    total_d_val_loss_d_lambda = torch.zeros(get_hyper_train().size(0)).cuda()

    num_weights = sum(p.numel() for p in model.parameters())
    d_val_loss_d_theta = torch.zeros(num_weights).cuda()
    model.train()
    for batch_idx, (x, y) in enumerate(val_loader):
        model.zero_grad()
        val_loss = val_loss_func(x, y)
        # val_loss_grad = grad(val_loss, model.parameters())
        d_val_loss_d_theta = d_val_loss_d_theta + gather_flat_grad(grad(val_loss, model.parameters()))
        if batch_idx >= val_batch_num: break
    d_val_loss_d_theta = d_val_loss_d_theta / (batch_idx + 1)

    # pre_conditioner = d_val_loss_d_theta  # TODO - where the preconditioner should be
    # flat_pre_conditioner = pre_conditioner

    model.train()  # train()
    for batch_idx, (x, y) in enumerate(train_loader):
        train_loss, _ = train_loss_func(x, y)
        # TODO (JON): Probably don't recompute - use create_graph and retain_graph?

        model.zero_grad(), hyper_optimizer.zero_grad()
        d_train_loss_d_theta = grad(train_loss, model.parameters(), create_graph=True)
        # flat_d_train_loss_d_theta = gather_flat_grad(d_train_loss_d_theta)

        flat_d_train_loss_d_theta = d_val_loss_d_theta.detach().reshape(1, -1) @ gather_flat_grad(
            d_train_loss_d_theta).reshape(-1, 1)

        model.zero_grad(), hyper_optimizer.zero_grad()
        # flat_d_train_loss_d_theta.backward()  #flat_pre_conditioner)
        # if get_hyper_train().grad is not None:
        total_d_val_loss_d_lambda = total_d_val_loss_d_lambda - gather_flat_grad(
            grad(flat_d_train_loss_d_theta.reshape(1), unshaped_get_hyper_train()))
        # get_hyper_train().grad
        # del d_train_loss_d_theta, flat_d_train_loss_d_theta
        if batch_idx >= train_batch_num: break
    total_d_val_loss_d_lambda = total_d_val_loss_d_lambda / (batch_idx + 1)

    direct_d_val_loss_d_lambda = torch.zeros(get_hyper_train().size(0)).cuda()
    '''model.train()
    for batch_idx, (x_val, y_val) in enumerate(val_loader):
        model.zero_grad(), hyper_optimizer.zero_grad()
        val_loss = val_loss_func(x_val, y_val)
        val_loss_grad = grad(val_loss, get_hyper_train(), allow_unused=True)
        if val_loss_grad is not None and val_loss_grad[0] is not None:
            direct_d_val_loss_d_lambda = direct_d_val_loss_d_lambda + gather_flat_grad(val_loss_grad)
            del val_loss_grad
        else:
            del val_loss_grad
            break
        if batch_idx >= val_batch_num: break
    direct_d_val_loss_d_lambda = direct_d_val_loss_d_lambda / (batch_idx + 1)'''

    target_grad = direct_d_val_loss_d_lambda + total_d_val_loss_d_lambda
    current_index = 0
    for p in unshaped_get_hyper_train():
        p_num_params = np.prod(p.shape)
        p.grad = target_grad[current_index:current_index + p_num_params].view(p.shape)
        current_index += p_num_params
    # del direct_d_val_loss_d_lambda, total_d_val_loss_d_lambda
    weight_norm, grad_norm = get_hyper_train().norm(), target_grad.norm()
    #print("weight={}, update={}".format(weight_norm, grad_norm))

    hyper_optimizer.step()
    model.zero_grad(), hyper_optimizer.zero_grad()
    # print(torch.cuda.memory_allocated(), torch.cuda.memory_cached(), torch.cuda.memory_cached() - torch.cuda.memory_allocated())
    # torch.cuda.empty_cache()
    return None, None, val_loss.detach(), weight_norm.detach(), grad_norm.detach()"""


def hyper_step(train_batch_num, val_batch_num, get_hyper_train, get_hyper_train_flat, model, val_loss_func, val_loader, train_loss_func,
               train_loader, hyper_optimizer):
    """

    :param train_batch_num:
    :param val_batch_num:
    :return:
    """
    from util import gather_flat_grad
    train_batch_num -= 1
    val_batch_num -= 1

    # set up placeholder for the partial derivative in each batch
    total_d_val_loss_d_lambda = torch.zeros(get_hyper_train_flat().size(0)).cuda()

    num_weights = sum(p.numel() for p in model.parameters())
    d_val_loss_d_theta = torch.zeros(num_weights).cuda()
    model.train()
    for batch_idx, (x, y) in enumerate(val_loader):
        model.zero_grad()
        val_loss = val_loss_func(x, y)
        # val_loss_grad = grad(val_loss, model.parameters())
        d_val_loss_d_theta = d_val_loss_d_theta + gather_flat_grad(grad(val_loss, model.parameters()))
        if batch_idx >= val_batch_num: break
    d_val_loss_d_theta = d_val_loss_d_theta / (batch_idx + 1)

    model.train()  # train()
    for batch_idx, (x, y) in enumerate(train_loader):
        train_loss, _ = train_loss_func(x, y)
        # TODO (JON): Probably don't recompute - use create_graph and retain_graph?

        model.zero_grad()
        # hyper_optimizer.zero_grad()
        d_train_loss_d_theta = grad(train_loss, model.parameters(), create_graph=True)
        # flat_d_train_loss_d_theta = gather_flat_grad(d_train_loss_d_theta)

        flat_d_train_loss_d_theta = d_val_loss_d_theta.detach().reshape(1, -1) @ gather_flat_grad(
            d_train_loss_d_theta).reshape(-1, 1)

        model.zero_grad()
        # hyper_optimizer.zero_grad()

        # flat_d_train_loss_d_theta.backward()  #flat_pre_conditioner)
        # if get_hyper_train().grad is not None:
        #if gather_flat_grad(get_hyper_train()) is not None:
        total_d_val_loss_d_lambda = total_d_val_loss_d_lambda - gather_flat_grad(
                grad(flat_d_train_loss_d_theta.reshape(1), get_hyper_train()))

        if batch_idx >= train_batch_num: break

    total_d_val_loss_d_lambda = total_d_val_loss_d_lambda / (batch_idx + 1)

    direct_d_val_loss_d_lambda = torch.zeros(get_hyper_train_flat().size(0)).cuda()

    grad_to_assign = direct_d_val_loss_d_lambda + total_d_val_loss_d_lambda
    current_index = 0
    for p in get_hyper_train():
        p_num_params = np.prod(p.shape)
        p.grad = grad_to_assign[current_index:current_index + p_num_params].view(p.shape)
        current_index += p_num_params
    # get_hyper_train().grad = (direct_d_val_loss_d_lambda + total_d_val_loss_d_lambda)

    weight_norm, grad_norm = get_hyper_train_flat().norm(), grad_to_assign.norm()  # get_hyper_train().grad.norm()
    print("weight={}, update={}".format(weight_norm, grad_norm))
    # print("weight={}, update={}".format(get_hyper_train_flat().norm(), gather_flat_grad(get_hyper_train()).norm()))

    hyper_optimizer.step()
    model.zero_grad()
    # hyper_optimizer.zero_grad()

    # return get_hyper_train(), get_hyper_train().grad, val_loss
    return val_loss, weight_norm, grad_norm


if __name__ == '__main__':
    experiment()

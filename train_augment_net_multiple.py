import argparse
import copy
import os
from utils.csv_logger import CSVLogger


def make_parser():
    """
    :return:
    """
    parser = argparse.ArgumentParser(description='CNN Hyperparameter Fine-tuning')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'], help='Choose a dataset')
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'wideresnet'], help='Choose a model')

    parser.add_argument('--num_finetune_epochs', type=int, default=200, help='Number of fine-tuning epochs')

    parser.add_argument('--load_checkpoint', type=str, help='Path to pre-trained checkpoint to load and finetune')
    parser.add_argument('--save_dir', type=str, default='finetuned_checkpoints',
                        help='Save directory for the fine-tuned checkpoint')

    parser.add_argument('--train_size', type=int, default=-1, help='The training size')
    parser.add_argument('--val_size', type=int, default=-1, help='The training size')
    parser.add_argument('--test_size', type=int, default=-1, help='The training size')

    parser.add_argument('--data_augmentation', action='store_true', default=True,
                        help='Whether to use data augmentation')
    parser.add_argument('--use_augment_net', action='store_true', default=True, help='Use augmentation network')
    parser.add_argument('--use_reweighting_net', action='store_true', default=False,
                        help='Use loss reweighting network')
    parser.add_argument('--use_weight_decay', action='store_true', default=False, help='Use weight_decay')
    parser.add_argument('--weight_decay_all', action='store_true', default=True, help='Use weight_decay')

    parser.add_argument('--num_neumann_terms', type=int, default=0, help='The maximum number of neumann terms to use')
    parser.add_argument('--use_cg', action='store_true', default=False, help='If we should use CG')
    parser.add_argument('--reg_weight', type=float, default=0.0, help='The weighting for the regularization')
    parser.add_argument('--seed', type=int, default=1, help='The random seed to use')

    parser.add_argument('--batch_size', type=int, default=128, help='The batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--wdecay', type=float, default=5e-4, help='Amount of weight decay')
    parser.add_argument('--do_diagnostic', action='store_true', default=False,
                        help='If we should do diagnostic functions')
    parser.add_argument('--do_simple', action='store_true', default=False,
                        help='If we should do diagnostic functions')
    parser.add_argument('--do_print', action='store_true', default=True,
                        help='If we should do diagnostic functions')
    parser.add_argument('--load_finetune_checkpoint', default='', help='Choose a model')
    parser.add_argument('--do_classification', action='store_false', default=True,
                        help='If we use cross-entropy loss')
    parser.add_argument('--do_inverse_compare', action='store_true', default=False,
                        help='If we use cross-entropy loss')
    parser.add_argument('--save_hessian', action='store_true', default=False,
                        help='If we use cross-entropy loss')

    parser.add_argument('--num_layers', type=int, default=0, help='How many mlp_layers')
    parser.add_argument('--warmup_epochs', type=int, default=-1, help='How many mlp_layers')
    return parser


def get_id(args):
    """
    :param args:
    :return:
    """
    id = ''
    id += f'data:{args.dataset}'
    id += f'_model:{args.model}'
    id += f'_reweight:{int(args.use_reweighting_net)}'
    id += f'_presetAug:{int(args.data_augmentation)}'
    id += f'_learnAug:{int(args.use_augment_net)}'
    id += f'_cg:{args.use_cg}'
    id += f'_neumann:{int(args.num_neumann_terms)}'
    id += f'_reg:{float(args.reg_weight)}'
    id += f'_seed:{int(args.seed)}'
    return id


def load_logger(args):
    """
    :param args:
    :return:
    """
    # Setup saving information
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    sub_dir = args.save_dir + '/' + get_id(args)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    test_id = get_id(args)
    filename = os.path.join(sub_dir, 'log.csv')
    csv_logger = CSVLogger(
        fieldnames=['epoch', 'run_time', 'iteration',
                    'train_loss', 'train_acc',
                    'val_loss', 'val_acc',
                    'test_loss', 'test_acc',
                    'hypergradient_cos_diff', 'hypergradient_l2_diff'],
        filename=filename)
    return csv_logger, test_id


def make_argss():
    """
    :return:
    """
    parser = make_parser()
    argss = []

    args, _ = parser.parse_known_args()
    # args.use_augment_net = False

    # TODO: Remember to add this to naming
    seed, reg_weight, num_neumann_terms = 1, .0, 0
    reg_weights = [0, 1.0]  # , .0]
    for do_preset_aug in [True]:  # [True, False]:
        for seed in [1, 2, 3]:
            for reg_weight in reg_weights:
                for num_neumann_terms in [1, 0, -1]:  # , 10]:  # -1 means we don't do any hyper_step
                    for do_reweight in [False]:  # [True, False]:
                        for do_augment in [True, False]:
                            if not do_reweight and not do_augment:
                                continue  # TODO: Will crash if we don't pass in any hyperparameters to optimize
                            new_args = copy.deepcopy(args)
                            new_args.data_augmentation = do_preset_aug
                            new_args.num_neumann_terms = num_neumann_terms
                            new_args.seed, new_args.reg_weight = seed, reg_weight
                            new_args.use_reweighting_net = do_reweight
                            new_args.use_augment_net = do_augment
                            if num_neumann_terms != -1:
                                argss += [new_args]
                            elif num_neumann_terms == -1:  # and reg_weight == 0
                                new_args.seed += 100 * (
                                    reg_weights.index(reg_weight))  # Every reg_weight is the same, so change seed
                                new_args.reg_weight = -1
                                new_args.do_augment = False
                                new_args.do_reweight = False
                                argss += [new_args]  # Don't need to graph the no training for different reg_weights
    return argss


def deploy_make_argss():
    """
    :return:
    """
    argss = make_argss()

    from train_augment_net2 import experiment
    for args in argss:
        experiment(args)


if __name__ == '__main__':
    deploy_make_argss()
    print("Finished!")

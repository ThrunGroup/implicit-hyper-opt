import ipdb
from pprint import pprint

import argparse
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch


# Local imports
from augment_net_experiment import experiment
from constants import DATASET_BOSTON, DATASET_MNIST, DATASET_CIFAR_100, DATASET_CIFAR_10, MODEL_RESNET18, \
    MODEL_WIDERESNET, MODEL_MLP, MODEL_CNN_MLP, HYPERPARAM_WEIGHT_DECAY, HYPERPARAM_WEIGHT_DECAY_GLOBAL, \
    HYPERPARAM_DATA_AUGMENT, HYPERPARAM_LOSS_REWEIGHT
from finetune_hyperparameters import make_val_size_compare_finetune_params, \
    make_boston_dataset_finetune_params
from train_augment_net_multiple import get_id


class AugmentNetTrainer(object):
    def __init__(self,
                 seeds=None,
                 hyperparams=None,
                 data_sizes=None,
                 val_props=None,
                 datasets=None,
                 model=None,
                 num_finetune_epochs=200,
                 lr=0.1,
                 hyper_lr = 0.1,
                 ):
        self.seeds = seeds or [1]
        self.hyperparams = hyperparams or [HYPERPARAM_DATA_AUGMENT]
        self.data_sizes = data_sizes or [100, 200, 1600]
        self.val_props = val_props or [.1, .25, .5, .75, .9]
        self.datasets = datasets or ['mnist']
        self.model = model or 'cnn_mlp'
        self.num_finetune_epochs = num_finetune_epochs
        self.lr = lr
        self.hyper_lr = hyper_lr

        if torch.cuda.is_available():
            print("GPU is available to use in this machine. Using", torch.cuda.device_count(), "GPUs...")
            self.device = torch.device('cuda')
        else:
            print("GPU is not available to use in this machine. Using CPU...")
            self.device = torch.device('cpu')

    def process(self):
        self.run_val_prop_compare()

        for dataset in args.datasets:
            exclude_sizes = []  # 50]
            fontsize = 16
            self.graph_val_prop_compare(dataset, exclude_sizes=exclude_sizes, do_legend=False, fontsize=fontsize)
            self.graph_val_prop_compare(dataset, exclude_sizes=exclude_sizes, retrain=True, do_legend=True, fontsize=fontsize)

    def run_val_prop_compare(self):
        # TODO (@Mo): Use itertools' product
        for seed in self.seeds:
            for dataset in self.datasets:
                for hyperparam in self.hyperparams:
                    for data_size in self.data_sizes:
                        data_to_save = {'val_losses': [], 'val_accs': [], 'test_losses': [], 'test_accs': [],
                                        'val_losses_re': [], 'val_accs_re': [], 'test_losses_re': [], 'test_accs_re': [],
                                        'info': ''}
                        for val_prop in self.val_props:
                            print(f"seed:{seed}, dataset:{dataset}, hyperparam:{hyperparam}, data_size:{data_size}, prop:{val_prop}")
                            args = make_val_size_compare_finetune_params(hyperparam, val_prop, data_size, dataset, self.model, self.num_finetune_epochs, self.lr)
                            args.seed = seed
                            train_loss, accuracy, val_loss, val_acc, test_loss, test_acc = experiment(args, self.device)
                            data_to_save['val_losses'] += [val_loss]
                            data_to_save['val_accs'] += [val_acc]
                            data_to_save['test_losses'] += [test_loss]
                            data_to_save['test_accs'] += [test_acc]

                            second_args = make_val_size_compare_finetune_params(hyperparam, 0, data_size, dataset, self.model, self.num_finetune_epochs, self.lr)
                            second_args.seed = seed
                            second_args.num_neumann_terms = -1
                            loc = '/sailhome/motiwari/data-augmentation/implicit-hyper-opt/CG_IFT_test/finetuned_checkpoints/'
                            loc += get_id(args) + '/'
                            loc += 'checkpoint.pt'
                            second_args.load_finetune_checkpoint = loc
                            train_loss_re, accuracy_re, val_loss_re, val_acc_re, test_loss_re, test_acc_re = experiment(second_args, self.device)
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

    def graph_val_prop_compare(self, dataset, exclude_sizes=[], retrain=False, do_legend=True, fontsize=12):
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

        for data_size in self.data_sizes[::-1]:
            if data_size in exclude_sizes:
                continue
            color = None
            for i, hyperparam in enumerate(self.hyperparams):
                data_to_graph = []
                for seed in self.seeds:
                    pickle_name = f'finetuned_checkpoints/dataset:{dataset}_datasize:{data_size}_hyperparam:{hyperparam}_seed:{seed}.pkl'
                    try:
                        with open(pickle_name, 'rb') as f:
                            data_for_seed = pickle.load(f)
                    except FileNotFoundError:
                        print(f"Could not open {pickle_name}")
                        break
                    data_to_graph += [data_for_seed[y_axis]]

                label = 'Size:' + str(data_size)
                if hyperparam == HYPERPARAM_WEIGHT_DECAY and data_size == self.data_sizes[0]:
                    hyperlabel = 'WD per weight'
                    label += ',Hyper:' + hyperlabel
                elif hyperparam == HYPERPARAM_WEIGHT_DECAY_GLOBAL and data_size == self.data_sizes[0]:
                    hyperlabel = hyperparam
                    hyperlabel = 'Global WD'
                    label += ',Hyper:' + hyperlabel  # + ',Data:' + dataset
                # if seed != seeds[0]: label = None

                if data_size != self.data_sizes[0] and hyperparam != self.hyperparams[0]: label = None
                plot = plt.errorbar(self.val_props,
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

        np.clip(np.round(np.array(self.val_props) * data_size), 1, 10e32)
        for data_size in self.data_sizes:
            if data_size in exclude_sizes:
                continue
            color = None
            for i, hyperparam in enumerate(self.hyperparams):
                data_to_graph = []
                for seed in self.seeds:
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
                if data_size != self.data_sizes[0] and hyperparam != self.hyperparams[0]: label = None
                plot = plt.errorbar(np.clip(np.round(np.array(self.val_props) * data_size), 1, 10e32),
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

    def multi_boston_args(self):
        num_neumanns = [10, 20, 5, 1, 0]
        hyperparams = [HYPERPARAM_WEIGHT_DECAY]  # , HYPERPARAM_WEIGHT_DECAY_GLOBAL]
        num_layers = [0]  # , 1]
        argss = []
        for num_neumann in num_neumanns:
            for hyperparam in hyperparams:
                for num_layer in num_layers:
                    args = make_boston_dataset_finetune_params(hyperparam, num_layer, num_neumann,
                                                               self.num_finetune_epochs, self.lr)
                    if num_neumann == 10:
                        args.use_cg = True
                        args.num_neumann_terms = 20
                    argss += [args]
        return argss

    def multi_boston_how_many_steps(self):
        num_neumanns = range(50)  # [0, 1, 20]
        hyperparams = [HYPERPARAM_WEIGHT_DECAY]  # , HYPERPARAM_WEIGHT_DECAY_GLOBAL]
        num_layers = [0]  # , 1]
        argss = []
        for num_neumann in num_neumanns:
            for hyperparam in hyperparams:
                for num_layer in num_layers:
                    args = make_boston_dataset_finetune_params(hyperparam, num_layer, num_neumann,
                                                               self.num_finetune_epochs, self.lr)
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
    parser.add_argument('--hyperparams', type=str, default=[HYPERPARAM_DATA_AUGMENT], metavar='H', nargs='+',
                        choices=[HYPERPARAM_WEIGHT_DECAY, HYPERPARAM_WEIGHT_DECAY_GLOBAL, HYPERPARAM_DATA_AUGMENT, HYPERPARAM_LOSS_REWEIGHT],
                        help=f"Hyperparameter list (default: [{HYPERPARAM_DATA_AUGMENT}])")
    parser.add_argument('--data-sizes', type=int, default=[50000], metavar='DSZ', nargs='+',
                        help='Data size list (default: [50000])')
    parser.add_argument('--val-props', type=float, default=[0.5], metavar='VP', nargs='+',
                        help='Validation proportion list (default: [0.5])')
    parser.add_argument('--datasets', type=str, default=['mnist'], metavar='DS', nargs='+',
                        choices=[DATASET_CIFAR_10, DATASET_CIFAR_100, DATASET_MNIST, DATASET_BOSTON],
                        help=f"Choose dataset list (default: [{DATASET_MNIST}])")
    parser.add_argument('--model', type=str, default='mlp', metavar='M',
                        choices=[MODEL_RESNET18, MODEL_WIDERESNET, MODEL_MLP, MODEL_CNN_MLP],
                        help=f"Choose a model (default: {MODEL_MLP})")
    parser.add_argument('--num-finetune-epochs', type=int, default=10000, metavar='NFE',
                        help='Number of fine-tuning epochs (default: 10000)')
    parser.add_argument('--lr', type=int, default=0.1, metavar='LR',
                        help='Learning rate (default: 0.1)')
    parser.add_argument('--hyper_lr', type=float, default=0.01, metavar='HYPERLR',
                        help='Hyperparameter learning rate (default: 0.01)')
    return parser.parse_args()


if __name__ == '__main__':
    # experiment(make_test_finetune_params())
    # experiment(make_inverse_compare_finetune_params())
    # experiment(make_val_size_compare(0.5, 100))

    '''
    seeds = [1, 2, 3]
    hyperparams = [HYPERPARAM_WEIGHT_DECAY, HYPERPARAM_WEIGHT_DECAY_GLOBAL]
    data_sizes = [50, 100, 250, 500, 1000]  # TODO: Generalize to other variables - ex. hyper choice
    val_props = [.0, .1, .25, .5, .75, .9]
    '''

    # curried_run_val(seeds[0])
    # p = Pool(len(seeds))
    # p.map(curried_run_val, seeds)
    args = parse_args()

    augment_net_trainer = AugmentNetTrainer(seeds=args.seeds,
                                            hyperparams=args.hyperparams,
                                            data_sizes=args.data_sizes,
                                            val_props=args.val_props,
                                            datasets=args.datasets,
                                            model=args.model,
                                            num_finetune_epochs=args.num_finetune_epochs,
                                            lr=args.lr,
                                            hyper_lr=args.hyper_lr,
                                            )
    augment_net_trainer.process()

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

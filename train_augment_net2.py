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
from finetune_hyperparameters import FinetuneHyperparameters
from train_augment_net_multiple import get_id


class AugmentNetTrainer(object):
    def __init__(self,
                 seeds=None,
                 hyperparams=None,
                 data_sizes=None,
                 val_props=None,
                 datasets=None,
                 model=None):
        self.seeds = seeds or [1]
        self.hyperparams = hyperparams or ['dataAugment']
        self.data_sizes = data_sizes or [100, 200, 1600]
        self.val_props = val_props or [.1, .25, .5, .75, .9]
        self.datasets = datasets or ['mnist']
        self.model = model or 'cnn_mlp'

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

    def make_test_arg(self):
        '''
        Instantiates a set of arguments for a test experiment
        '''

        test_args = FinetuneHyperparameters()
        test_args.update_hyperparameters({
            'reg_weight': .5,
            'num_neumann_terms': 1,
            'use_cg': False,
            'seed': 3333,
            'do_diagnostic': True,
            'data_augmentation': True,
            'use_reweighting_net': False,
            'use_augment_net': True,
            'use_weight_decay': False
        })
        return test_args

    def make_inverse_compare_arg(self):
        test_args = FinetuneHyperparameters()
        test_args.update_hyperparameters({
            'reg_weight': 1.0,
            'seed': 8888,
            'do_diagnostic': True,
            'data_augmentation': True,
            'use_reweighting_net': False,
            'use_augment_net': True,
            'batch_size': 50,
            'train_size': 50,
            'val_size': 1000,
            'test_size': 100,
            'num_finetune_epochs': 10000,
            'model': 'resnet18',            # 'resnet18', 'mlp'
            'use_weight_decay': False,      # TODO: Add weight_decay to saveinfo?
            'dataset': 'mnist',             # 'mnist', 'cifar10'  # TODO: Need to add dataset to the save info?
            'num_neumann_terms': -1,
            'use_cg': False
        })
        return test_args


    def make_val_size_compare(self, hyperparam, val_prop, data_size, dataset, model):
        '''
        Not sure
        '''
        assert 0 <= val_prop <= 1.0, 'Train proportion in [0, 1]'

        train_size = int(data_size * (1.0 - val_prop))
        train_size = 1 if train_size <= 0 else train_size
        val_size = int(data_size * val_prop)
        val_size = 1 if val_size <= 0 else val_size

        use_weight_decay = False
        weight_decay_all = False
        use_reweighting_net = False
        use_augment_net = False

        if hyperparam == 'weightDecayParams':
            use_weight_decay = True
            weight_decay_all = True
        elif hyperparam == 'weightDecayGlobal':
            use_weight_decay = True
        elif hyperparam == 'dataAugment':
            use_augment_net = True
        elif hyperparam == 'lossReweight':
            use_reweighting_net = True

        test_args = FinetuneHyperparameters()
        test_args.update_hyperparameters({
            'reg_weight': 0.0,
            'seed': 1,
            'data_augmentation': False,
            'batch_size': data_size,    # TODO: Do i want a variable batch size?
            'val_prop': val_prop,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': -1,            # TODO: For long running, boost test_size and num_epochs
            'num_finetune_epochs': 250,
            'model': model,
            'use_weight_decay': use_weight_decay,
            'weight_decay_all': weight_decay_all,
            'use_reweighting_net': use_reweighting_net,
            'use_augment_net': use_augment_net,
            'dataset': dataset,         # 'mnist', 'cifar10'  # TODO: Need to add dataset to the save info?
            'do_simple': True,
            'do_diagnostic': False,
            'do_print': False,
            'num_neumann_terms': -1 if val_size == 1 else 3,
            'use_cg': False,
            'only_print_final_vals': False,
            'load_finetune_checkpoint': '',
        })
        return test_args


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
                            args = self.make_val_size_compare(hyperparam, val_prop, data_size, dataset, self.model)
                            args.seed = seed
                            train_loss, accuracy, val_loss, val_acc, test_loss, test_acc = experiment(args, self.device)
                            data_to_save['val_losses'] += [val_loss]
                            data_to_save['val_accs'] += [val_acc]
                            data_to_save['test_losses'] += [test_loss]
                            data_to_save['test_accs'] += [test_acc]

                            second_args = self.make_val_size_compare(hyperparam, 0, data_size, dataset)
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
            if data_size in exclude_sizes: continue
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
                if hyperparam == 'weightDecayParams' and data_size == self.data_sizes[0]:
                    hyperlabel = 'WD per weight'
                    label += ',Hyper:' + hyperlabel
                elif hyperparam == 'weightDecayGlobal' and data_size == self.data_sizes[0]:
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
            if data_size in exclude_sizes: continue
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


    # TODO: Make a function to create multiple args to deploy
    def do_boston(self, hyperparam, num_layer, num_neumann):
        use_weight_decay = False
        weight_decay_all = False
        use_reweighting_net = False
        use_augment_net = False

        if hyperparam == 'weightDecayParams':
            use_weight_decay = True
            weight_decay_all = True
        elif hyperparam == 'weightDecayGlobal':
            use_weight_decay = True
        elif hyperparam == 'dataAugment':
            use_augment_net = True
        elif hyperparam == 'lossReweight':
            use_reweighting_net = True

        test_args = FinetuneHyperparameters()
        test_args.update_hyperparameters({
            'reg_weight': 0.0,
            'seed': 1,
            'data_augmentation': False,
            'batch_size': 128 * 4,
            'model': 'mlp' + str(num_layer),    # 'resnet18', 'mlp'
            'use_weight_decay': use_weight_decay,
            'weight_decay_all': weight_decay_all,
            'use_reweighting_net': use_reweighting_net,
            'use_augment_net': use_augment_net,
            'num_layers': num_layer,
            'dataset': 'boston',
            'do_classification': False,
            'do_simple': True,
            'do_diagnostic': False,
            'do_print': True,
            'num_neumann_terms': num_neumann,
            'use_cg': False,
            'warmup_epochs': 200,
            'num_finetune_epochs': 600,
            'do_inverse_compare': True,
            'save_hessian': False
        })
        return test_args


    def multi_boston_args(self):
        num_neumanns = [10, 20, 5, 1, 0]
        hyperparams = ['weightDecayParams']  # , 'weightDecayGlobal']
        num_layers = [0]  # , 1]
        argss = []
        for num_neumann in num_neumanns:
            for hyperparam in hyperparams:
                for num_layer in num_layers:
                    args = self.do_boston(hyperparam, num_layer, num_neumann)
                    if num_neumann == 10:
                        args.use_cg = True
                        args.num_neumann_terms = 20
                    argss += [args]
        return argss


    def multi_boston_how_many_steps(self):
        num_neumanns = range(50)  # [0, 1, 20]
        hyperparams = ['weightDecayParams']  # , 'weightDecayGlobal']
        num_layers = [0]  # , 1]
        argss = []
        for num_neumann in num_neumanns:
            for hyperparam in hyperparams:
                for num_layer in num_layers:
                    args = self.do_boston(hyperparam, num_layer, num_neumann)
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
    parser.add_argument('--data-sizes', type=int, default=[50000], metavar='DSZ', nargs='+',
                        help='Data size list (default: [50000])')
    parser.add_argument('--val-props', type=float, default=[0.1], metavar='VP', nargs='+',
                        help='Validation proportion list (default: [0.1])')
    parser.add_argument('--datasets', type=str, default=['mnist'], metavar='DS', nargs='+',
                        choices=['cifar10', 'cifar100', 'mnist', 'boston'],
                        help='Choose dataset list (default: [mnist])')
    parser.add_argument('--model', type=str, default='cnn_mlp', metavar='M',
                        choices=['resnet18', 'wideresnet', 'mlp', 'cnn_mlp'],
                        help='Choose a model (default: cnn_mlp)')
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

    augment_net_trainer = AugmentNetTrainer(seeds=args.seeds,
                                            hyperparams=args.hyperparams,
                                            data_sizes=args.data_sizes,
                                            val_props=args.val_props,
                                            datasets=args.datasets,
                                            model=args.model)
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

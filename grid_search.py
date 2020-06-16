import os
import sys
import csv
import ipdb
import time
import yaml
import pickle
import random
import argparse
import itertools
from collections import OrderedDict

import numpy as np

from train_baseline import cnn_val_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='A YAML file specifying the grid search/random search configuration.')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Set the name of the experiment.')
    parser.add_argument('--seed', type=int, default=11,
                        help='Set random seed')
    args = parser.parse_args()

    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load the YAML configuration file
    with open(args.config, 'r') as f:
        config = yaml.load(f)

    fixed_hparam_dict = OrderedDict()
    tune_hparam_dict = OrderedDict()
    search_over = []

    for hparam in config['fixed_hparams']:
        fixed_hparam_dict[hparam] = config['fixed_hparams'][hparam]

    for hparam in config['tune_hparams']:
        hparam_values = []

        if 'sampling' in config['tune_hparams'][hparam]:
            if config['tune_hparams'][hparam]['sampling'] == 'grid':
                min_val = round(config['tune_hparams'][hparam]['min_val'], 2)
                max_val = round(config['tune_hparams'][hparam]['max_val'], 2)
                num_samples = config['tune_hparams'][hparam]['num_samples']
                search_over.append('{}:grid'.format(hparam))
                interval = round((max_val - min_val) / (num_samples - 1), 2)
                hparam_values = [round(min_val + i * interval, 2) for i in range(num_samples)]
                tune_hparam_dict[hparam] = hparam_values

    if args.exp_name is None:
        args.exp_name = 'cnn-{}'.format('-'.join(search_over))

    args.exp_name = '{}_seed_{}'.format(args.exp_name, args.seed)

    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)

    fixed_hparam_dict['save_dir'] = args.exp_name

    callback_file = open(os.path.join(args.exp_name, 'callback.csv'), 'w')
    callback_writer = csv.DictWriter(callback_file, fieldnames=['elapsed_time', 'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'] + list(reversed(list(tune_hparam_dict.keys()))))
    callback_writer.writeheader()

    def callback(epoch, avg_xentropy, train_acc, val_loss, val_acc, config):
        global curr_hparam_dict
        elapsed_time = time.time() - start_time
        result_dict = { 'elapsed_time': elapsed_time, 'epoch': epoch, 'train_loss': avg_xentropy, 'train_acc': train_acc,
                        'val_loss': val_loss, 'val_acc': val_acc }
        result_dict.update(curr_hparam_dict)
        callback_writer.writerow(result_dict)
        callback_file.flush()


    # Save the final val and test performance to a results CSV file
    result_file = open(os.path.join(args.exp_name, 'progress.csv'), 'w')
    result_writer = csv.DictWriter(result_file, fieldnames=['elapsed_time', 'train_loss', 'train_acc',
                                                            'val_loss', 'val_acc', 'test_loss', 'test_acc'] + list(tune_hparam_dict.keys()))
    result_writer.writeheader()

    start_time = time.time()

    try:
        for hparam_tuple in itertools.product(*tune_hparam_dict.values()):
            curr_hparam_dict = { key: value for (key,value) in zip(list(tune_hparam_dict.keys()), list(hparam_tuple)) }
            params = {**fixed_hparam_dict, **curr_hparam_dict}

            train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = cnn_val_loss(params, callback=callback, return_all=True)

            elapsed_time = time.time() - start_time
            result_dict = { 'elapsed_time': elapsed_time,
                            'train_loss': train_loss, 'train_acc': train_acc,
                            'val_loss': val_loss, 'val_acc': val_acc,
                            'test_loss': test_loss, 'test_acc': test_acc }
            result_dict.update(curr_hparam_dict)
            result_writer.writerow(result_dict)
            result_file.flush()
    except KeyboardInterrupt:
        print('Exiting out of grid search')

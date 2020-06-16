"""
Usage:
    python create_command_script.py YAML_CONFIG_FILE

    This will create a file like run_scripts/wgan-gp. You can run the commands
    in this file in parallel using xargs as follows:

        xargs -P 4 -I {} sh -c 'eval "$1"' - {} < run_scripts/wgan-gp
"""
import os
import ipdb
import yaml
import argparse
import itertools

import numpy as np


def uniform_in_range(min_val, max_val):
    """Helper function to generate uniform random numbers in the range [min_val, max_val].
    """
    return min_val + np.random.rand() * (max_val - min_val)


def create_commands(config_fname, name_prefix=''):
    # Load the YAML configuration file
    with open(config_fname, 'r') as f:
        config = yaml.load(f)

    # ---------------------------------------------------------------------
    d = {}

    for hparam in config['tune_hparams']:

        hparam_values = []

        if ',' in hparam:
            hparam_keys = hparam.split(',')
            hparam_values = config['tune_hparams'][hparam]['values']
            d[tuple(hparam_keys)] = hparam_values
        else:

            if 'sampling' in config['tune_hparams'][hparam]:
                min_val = round(config['tune_hparams'][hparam]['min_val'], 2)
                max_val = round(config['tune_hparams'][hparam]['max_val'], 2)
                num_samples = config['tune_hparams'][hparam]['num_samples']

                if config['tune_hparams'][hparam]['sampling'] == 'grid':
                    interval = round((max_val - min_val) / (num_samples - 1), 2)
                    hparam_values = [round(min_val + i * interval, 2) for i in range(num_samples)]

                elif config['tune_hparams'][hparam]['sampling'] == 'random':
                    hparam_values = [round(uniform_in_range(min_val, max_val), 4) for _ in range(num_samples)]
            else:  # Should have a fixed list of values that the hyperparameter can take on, like [10, 20, 30]
                hparam_values = config['tune_hparams'][hparam]['values']

            d[hparam] = hparam_values
    # ---------------------------------------------------------------------


    # --------------------------------------------------------------------
    list_of_tuples = []
    for hparam_name in d:
        list_of_tuples.append([(hparam_name, value) for value in d[hparam_name]])
    # --------------------------------------------------------------------


    # --------------------------------------------------------------------
    base_command = config['base_command']

    for param_name, param_value in config['fixed_hparams'].items():
        if param_value is not False:
            base_command += ' --{}'.format(param_name) if param_value is True else ' --{}={}'.format(param_name, param_value)

    command_list = []
    for cidx, combo in enumerate(itertools.product(*list_of_tuples)):
        hparam_args = ""
        for hparam_name, hparam_value in combo:
            if isinstance(hparam_name, tuple):  # Then we also require that the hparam_values are tuples
                for hname, hvalue in zip(hparam_name, hparam_value):
                    hparam_args += ' --{}={}'.format(hname, hvalue)
            else:
                if hparam_value is not False:
                    hparam_args += ' --{}'.format(hparam_name) if hparam_value is True else ' --{}={}'.format(hparam_name, hparam_value)
        command = '{} {}'.format(base_command, hparam_args)

        # command += ' --exp_name {}_{}'.format(name_prefix, cidx)
        command_list.append(command)
    # --------------------------------------------------------------------


    # Save the commands to a file
    if not os.path.exists(config['command_dir']):
        os.makedirs(config['command_dir'])

    with open(os.path.join(config['command_dir'], config['filename']), 'w') as f:
        for command in command_list:
            f.write('{}\n'.format(command))
            f.flush()

    return os.path.join(config['command_dir'], config['filename'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Command Script Generator (for grid search and random search)')
    parser.add_argument('--config', type=str,
                        help='A YAML file specifying the grid search/random search configuration.')
    parser.add_argument('--name_prefix', type=str, default='')
    args = parser.parse_args()

    create_commands(args.config, args.name_prefix)

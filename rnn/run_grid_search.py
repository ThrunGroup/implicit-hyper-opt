"""
Usage: python run_grid_search.py YAML_CONFIG NUM_JOBS_IN_PARALLEL
Example: python run_grid_search.py config_scripts/wgan-gp.yaml 4

What this does:
    1. Call `python create_command_script.py YAML_CONFIG` and get the output filename
    2. Call the shell command `xargs -P NUM_JOBS_IN_PARALLEL -I {} sh -c 'eval "$1"' - {} < GRID_COMMAND_FILE`
"""

import os
import ipdb
import argparse

# Local imports
import create_command_script


parser = argparse.ArgumentParser(description='Command Script Generator (for grid search and random search)')
parser.add_argument('--config', type=str,
                    help='A YAML file specifying the grid search/random search configuration.')
parser.add_argument('--num', type=int,
                    help='How many jobs to run in parallel (e.g., how many GPUs to use simultaneously)')
parser.add_argument('--name_prefix', type=str, default='')
args = parser.parse_args()


if __name__ == '__main__':

    command_filename = create_command_script.create_commands(args.config, args.name_prefix)
    system_command = '''xargs -P {} -I {{}} sh -c 'eval "$1"' - {{}} < {}'''.format(args.num, command_filename)
    os.system(system_command)

import os
import csv
import ipdb
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style='white')
sns.set_palette('bright')


def load_results(fname):
    result_dict = defaultdict(list)
    with open(fname, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key in row:
                try:
                    value = float(row[key])
                    result_dict[key].append(value)
                except:
                    pass
    return result_dict


def plot_epoch_loss(exp_dir, save_to=None):
    # exp_dir = '/scratch/gobi1/pvicol/CG_IFT_test/rnn/logs/saves_wdecay/2019-09-30-tune:[\'wdecay\']-lr:0.0001-hopt:adam-hlr:0.0001-small:0-di:0.05-dh:0.05-do:0.05-a:0.0-b:0.0-wd:1e-06-wdtype:per_layer'
    fpath = os.path.join(exp_dir, 'epoch.csv')
    results = load_results(fpath)

    fig = plt.figure(figsize=(5,4))
    plt.plot(results['epoch'], results['train_loss'], linewidth=2)
    plt.plot(results['epoch'], results['val_loss'], linewidth=2)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.legend(['Train', 'Val'], fontsize=18)
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_epoch_ppl(exp_dir, save_to=None):
    # exp_dir = '/scratch/gobi1/pvicol/CG_IFT_test/rnn/logs/saves_wdecay/2019-09-30-tune:[\'wdecay\']-lr:0.0001-hopt:adam-hlr:0.0001-small:0-di:0.05-dh:0.05-do:0.05-a:0.0-b:0.0-wd:1e-06-wdtype:per_layer'
    fpath = os.path.join(exp_dir, 'epoch.csv')
    results = load_results(fpath)

    fig = plt.figure(figsize=(5,4))
    plt.plot(results['epoch'], results['train_ppl'], linewidth=2)
    plt.plot(results['epoch'], results['val_ppl'], linewidth=2)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(10, 200)
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Perplexity', fontsize=20)
    plt.legend(['Train', 'Val'], fontsize=18)
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_hparams(exp_dir, save_to=None):
    # exp_dir = '/scratch/gobi1/pvicol/CG_IFT_test/rnn/logs/saves_wdecay/2019-09-30-tune:[\'wdecay\']-lr:0.0001-hopt:adam-hlr:0.0001-small:0-di:0.05-dh:0.05-do:0.05-a:0.0-b:0.0-wd:1e-06-wdtype:per_layer'
    fpath = os.path.join(exp_dir, 'iteration.csv')
    results = load_results(fpath)

    hparam_keys = []

    fig = plt.figure(figsize=(5,4))
    for key in results.keys():
        if key not in ['iteration', 'time', 'train_loss', 'val_loss', 'train_ppl', 'val_ppl']:
            plt.plot(results['iteration'], results[key], linewidth=2)
            hparam_keys.append(key)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Hparam', fontsize=20)
    plt.legend(hparam_keys, fontsize=18)
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_item(result_dict,
              xkey,
              ykey,
              xlabel='',
              ylabel='',
              xlabel_fontsize=22,
              ylabel_fontsize=22,
              xtick_fontsize=18,
              ytick_fontsize=18,
              yscale='linear',
              linewidth=2,
              save_to=None):
    
    fig = plt.figure(figsize=(5,4))

    max_iter = 1e6

    num_values = min(len(result_dict[xkey]), len(result_dict[ykey]))
    num_values = min(num_values, max([i for (i, value) in enumerate(result_dict['global_iteration']) if value < max_iter]))

    plt.plot(result_dict[xkey][:num_values], result_dict[ykey][:num_values], linewidth=linewidth)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    # plt.xticks([0, 60000, 120000, 180000], fontsize=xtick_fontsize)
    plt.xticks(fontsize=xtick_fontsize)
    plt.yticks(fontsize=ytick_fontsize)
    plt.yscale(yscale)

    if max(result_dict[xkey]) > 3000:
        plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'k'))
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_item(result_dict,
              xkey,
              ykey,
              xlabel='',
              ylabel='',
              xlabel_fontsize=22,
              ylabel_fontsize=22,
              xtick_fontsize=18,
              ytick_fontsize=18,
              yscale='linear',
              linewidth=2,
              save_to=None):
    
    fig = plt.figure(figsize=(5,4))

    max_iter = 1e6

    num_values = min(len(result_dict[xkey]), len(result_dict[ykey]))
    num_values = min(num_values, max([i for (i, value) in enumerate(result_dict['global_iteration']) if value < max_iter]))

    plt.plot(result_dict[xkey][:num_values], result_dict[ykey][:num_values], linewidth=linewidth)
    plt.xlabel(xlabel, fontsize=xlabel_fontsize)
    plt.ylabel(ylabel, fontsize=ylabel_fontsize)
    # plt.xticks([0, 60000, 120000, 180000], fontsize=xtick_fontsize)
    plt.xticks(fontsize=xtick_fontsize)
    plt.yticks(fontsize=ytick_fontsize)
    plt.yscale(yscale)

    if max(result_dict[xkey]) > 3000:
        plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x/1000) + 'k'))
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str,
                        help='Path to the experiment directory')
    args = parser.parse_args()

    plot_hparams(args.exp_dir)
    plot_epoch_ppl(args.exp_dir)

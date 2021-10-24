import wandb
import math
from train_augment_experiment import experiment

sweep_config = {'method': 'random'}
metric = {'name': 'diff', 'goal': 'maximize'}

# The keys of "parameters_dict" : dataset, model, aug_model, num_layers, dropout, fc_shape, wf, depth, use_cuda,
# loss_criterion, hessian, num_neumann, neumann_converge_factor, optimizer, hyper_optimizer, epochs, hepochs,
# model_lr, hyper_model_lr, batch_size, datasize, train_prop, test_size, seed, patience

# Tutorial of hyperparams sweep using wandb:
# https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch
# /Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb

parameters_dict = {
    # refer to https://docs.wandb.ai/guides/sweeps/configuration#distributions for distribution of hyperparams sweep
    'dataset': {
        # uniformly select between two datasets
        'distribution': 'categorical',
        'values': ['mnist']  # , 'cifar10']
    },
    'model': {
        # uniformly select between two models
        'distribution': 'categorical',
        'values': ['mlp'] # , 'cnn']
    },
    'num_layers': {
        'distribution': 'categorical',
        'values': [1,2]
    },
    'aug_model': {
        'distribution': 'constant',
        'value': 'unet'
    },
    'fc_shape': {
        # randomly select real number in [min, max] from uniform distribution
        'distribution': 'int_uniform',
        'min': 700,
        'max': 800
    },
    'dropout': {
        # randomly select real number x in [min, max] and return exp(x)
        'distribution': 'log_uniform',
        'min': math.log(0.1),
        'max': math.log(0.15)
    },
    'wf': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 2
    },
    'depth': {
        'distribution': 'constant',
        'value': 1
    },
    'use_cuda': {
        'distribution': 'constant',
        'value': True
    },
    'loss_criterion': {
        'distribution': 'log_uniform',
        'min': math.log(1e-8),
        'max': math.log(1e-4)
    },
    'hessian': {
        'distribution': 'constant',
        'value': 'neumann'
    },
    'num_neumann': {
        'distribution': 'int_uniform',
        'min': 3,
        'max': 4
    },
    'neumann_converge_factor': {
        'distribution': 'log_uniform',
        'min': math.log(1e-4),
        'max': math.log(1e-3)
    },
    'optimizer': {
        'distribution': 'categorical',
        'values': ['adam']
    },
    'hyper_optimizer': {
        'distribution': 'categorical',
        'values': ['rmsprop'] # ['adam', 'adagrad', 'rmsprop']
    },
    'epochs': {
        'distribution': 'int_uniform',
        'min': 300,
        'max': 400
    },
    'hepochs': {
        'distribution': 'int_uniform',
        'min': 30,
        'max': 33,
    },
    'model_lr': {
        'distribution': 'log_uniform',
        'min': math.log(1e-4),
        'max': math.log(1e-1)
    },
    'hyper_model_lr': {
        'distribution': 'log_uniform',
        'min': math.log(1e-4),
        'max': math.log(1e-2)
    },
    'datasize': {
        # round log_uniform
        'distribution': 'q_log_uniform',
        'min': math.log(300),
        'max': math.log(350)
    },
    'batch_size': {
        'distribution': 'q_log_uniform',
        'min': math.log(10),
        'max': math.log(15)
    },
    'train_prop': {
        'distribution': 'uniform',
        'min': 0.5,
        'max': 0.75
    },
    'test_size': {
        'distribution': 'constant',
        'value': -1
    },
    'seed': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 100
    },
    'patience': {
        'distribution': 'int_uniform',
        'min': 9,
        'max': 10
    }
}
sweep_config['metric'] = metric
sweep_config['parameters'] = parameters_dict
if __name__ == '__main__':
    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
    wandb.agent(sweep_id, experiment, count=5)

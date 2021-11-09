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
        'values': ['resnet']
    },
    'num_layers': {
        'distribution': 'categorical',
        'values': [1, 2]
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
        'max': math.log(0.11)
    },
    'wf': {
        'distribution': 'constant',
        'value': 5
    },
    'depth': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 4
    },
    'use_cuda': {
        'distribution': 'constant',
        'value': True
    },
    'loss_criterion': {
        'distribution': 'constant',
        'value': 0.000004105
    },
    'hessian': {
        'distribution': 'constant',
        'value': 'neumann'
    },
    'num_neumann': {
        'distribution': 'constant',
        'value': 3
    },
    'neumann_converge_factor': {
        'distribution': 'constant',
        'value': 0.0009768
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
        'min': 380,
        'max': 600
    },
    'hepochs': {
        'distribution': 'int_uniform',
        'min': 30,
        'max': 100,
    },
    'model_lr': {
        'distribution': 'log_uniform',
        'min': math.log(1e-4),
        'max': math.log(5e-3)
    },
    'hyper_model_lr': {
        'distribution': 'log_uniform',
        'min': math.log(0.0002),
        'max': math.log(0.01)
    },
    'datasize': {
        # round log_uniform
        'distribution': 'int_uniform',
        'min': 1000,
        'max': 1050
    },
    'batch_size': {
        'distribution': 'q_log_uniform',
        'min': math.log(20),
        'max': math.log(21)
    },
    'train_prop': {
        'distribution': 'uniform',
        'min': 0.55,
        'max': 0.56
    },
    'test_size': {
        'distribution': 'constant',
        'value': -1
    },
    'val2_size': {
        'distribution': 'constant',
        'value': -1
    },
    'seed': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 1000
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
    wandb.agent(sweep_id, experiment, count=50)

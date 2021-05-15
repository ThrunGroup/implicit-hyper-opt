from constants import HYPERPARAM_WEIGHT_DECAY, HYPERPARAM_WEIGHT_DECAY_GLOBAL, HYPERPARAM_DATA_AUGMENT, \
    HYPERPARAM_LOSS_REWEIGHT


class FinetuneHyperparameters(object):
    def __init__(self, hyperparameters=None):
        # Choose a dataset
        self.dataset = 'cifar10'    # 'cifar10', 'cifar100'

        # Choose a dataset
        self.model = 'resnet18'     # 'resnet18', 'wideresnet'

        # Number of fine-tuning epochs
        self.num_finetune_epochs = 200

        # Path to pre-trained checkpoint to load and finetune
        self.load_checkpoint = None

        # Save directory for the fine-tuned checkpoint
        self.save_dir = 'finetuned_checkpoints'

        # The training size
        self.train_size = -1

        # The training size
        self.val_size = -1

        # The training size
        self.test_size = -1

        # Whether to use data augmentation
        self.data_augmentation = True

        # Use augmentation network
        self.use_augment_net = True

        # Use loss reweighting network
        self.use_reweighting_net = False

        # Use weight_decay
        self.use_weight_decay = False

        # Use weight_decay all
        self.weight_decay_all = False

        # The maximum number of neumann terms to use
        self.num_neumann_terms = 0

        # If we should use CG
        self.use_cg = False

        # The weighting for the regularization
        self.reg_weight = 0.0

        # The random seed to use
        self.seed = 1

        # The batch size
        self.batch_size = 128

        # Learning rate
        self.lr = 0.01

        # Hyperparameter learning rate
        self.hyper_lr = 0.01

        # Amount of weight decay
        self.wdecay = 5e-4

        # If we should do diagnostic functions
        self.do_diagnostic = False

        # If we should do diagnostic functions
        self.do_simple = False

        # If we should do diagnostic functions
        self.do_print = True

        # Choose a model
        self.load_finetune_checkpoint = ''

        # If we use cross-entropy loss
        self.do_classification = True

        # If we use cross-entropy loss
        self.do_inverse_compare = False

        # If we use cross-entropy loss
        self.save_hessian = False

        # How many mlp_layers
        self.num_layers = 0

        # How many mlp_layers
        self.warmup_epochs = -1

        # Update the given hyper parameters
        if hyperparameters:
            for name, value in hyperparameters.items():
                setattr(self, name, value)


def make_test_finetune_params(self):
    '''
    Instantiates a set of arguments for a test experiment
    '''

    return FinetuneHyperparameters({
        'reg_weight': .5,
        'num_neumann_terms': 1,
        'use_cg': False,
        'seed': 3333,
        'do_diagnostic': True,
        'data_augmentation': True,
        'use_reweighting_net': False,
        'use_augment_net': True,
        'use_weight_decay': False,
        'num_finetune_epochs': self.num_finetune_epochs,
        'lr': self.lr,
    })


def make_inverse_compare_finetune_params(num_finetune_epochs, lr):
    return FinetuneHyperparameters({
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
        'num_finetune_epochs': num_finetune_epochs,
        'model': 'resnet18',            # 'resnet18', 'mlp'
        'use_weight_decay': False,      # TODO: Add weight_decay to saveinfo?
        'dataset': 'mnist',             # 'mnist', 'cifar10'  # TODO: Need to add dataset to the save info?
        'num_neumann_terms': -1,
        'use_cg': False,
        'lr': lr,
    })


def make_val_size_compare_finetune_params(hyperparam, val_prop, data_size, dataset, model, num_finetune_epochs, lr):
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

    if hyperparam == HYPERPARAM_WEIGHT_DECAY:
        use_weight_decay = True
        weight_decay_all = True
    elif hyperparam == HYPERPARAM_WEIGHT_DECAY_GLOBAL:
        use_weight_decay = True
    elif hyperparam == HYPERPARAM_DATA_AUGMENT:
        use_augment_net = True
    elif hyperparam == HYPERPARAM_LOSS_REWEIGHT:
        use_reweighting_net = True

    return FinetuneHyperparameters({
        'reg_weight': 0.0,
        'seed': 1,
        'data_augmentation': False,
        'batch_size': data_size,    # TODO: Do i want a variable batch size?
        'val_prop': val_prop,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': -1,            # TODO: For long running, boost test_size and num_epochs
        'num_finetune_epochs': num_finetune_epochs,
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
        'lr': lr,
    })


# TODO: Make a function to create multiple args to deploy
def make_boston_dataset_finetune_params(hyperparam, num_layer, num_neumann, num_finetune_epochs, lr):
    use_weight_decay = False
    weight_decay_all = False
    use_reweighting_net = False
    use_augment_net = False

    if hyperparam == HYPERPARAM_WEIGHT_DECAY:
        use_weight_decay = True
        weight_decay_all = True
    elif hyperparam == HYPERPARAM_WEIGHT_DECAY_GLOBAL:
        use_weight_decay = True
    elif hyperparam == HYPERPARAM_DATA_AUGMENT:
        use_augment_net = True
    elif hyperparam == HYPERPARAM_LOSS_REWEIGHT:
        use_reweighting_net = True

    return FinetuneHyperparameters({
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
        'num_finetune_epochs': num_finetune_epochs,
        'do_inverse_compare': True,
        'save_hessian': False,
        'lr': lr,
    })

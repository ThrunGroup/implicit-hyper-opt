class FinetuneHyperparameters(object):
    def __init__(self):
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

    def update_hyperparameters(self, hyperparameters):
        for name, value in hyperparameters.items():
            setattr(self, name, value)

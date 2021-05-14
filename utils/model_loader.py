import numpy as np
import torch
from torch.autograd import Variable

from constants import DATASET_CIFAR_10, DATASET_CIFAR_100, DATASET_MNIST, DATASET_BOSTON, MODEL_RESNET18, \
    MODEL_WIDERESNET, MODEL_MLP, MODEL_CNN_MLP
from models.cnn_mlp import CNN_MLP
from models.resnet import ResNet18
from models.simple_models import Net
from models.unet import UNet
from models.wide_resnet import WideResNet

INIT_L2 = -7  # TODO: Important to make sure this is small enough to be unregularized when starting?


class ModelLoader(object):
    def __init__(self, args, device):
        self.args = args
        self.device = device

    def get_models(self):
        '''
        Loads both the baseline model and the finetuning models in train mode
        '''
        model, checkpoint = self.load_baseline_model()
        augment_net, reweighting_net, model = self.load_finetuned_model(model)
        # model = nn.DataParallel(model)
        # augment_net = nn.DataParallel(model)
        return model, augment_net, reweighting_net, checkpoint

    def load_baseline_model(self):
        """
        Load a simple baseline model AND dataset
        Note that this sets the model to training mode
        """
        if self.args.dataset == DATASET_CIFAR_10:
            imsize, in_channel, num_classes = 32, 3, 10
        elif self.args.dataset == DATASET_CIFAR_100:
            imsize, in_channel, num_classes = 32, 3, 100
        elif self.args.dataset == DATASET_MNIST:
            imsize, in_channel, num_classes = 28, 1, 10
        elif self.args.dataset == DATASET_BOSTON:
            imsize, in_channel, num_classes = 13, 1, 1

        # init_l2 = -7  # TODO: Important to make sure this is small enough to be unregularized when starting?
        if self.args.model == MODEL_RESNET18:
            cnn = ResNet18(num_classes=num_classes)
        elif self.args.model == MODEL_WIDERESNET:
            cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)
        elif self.args.model[:3] == MODEL_MLP:
            cnn = Net(self.args.num_layers, 0.0, imsize, in_channel, INIT_L2, num_classes=num_classes,
                      do_classification=self.args.do_classification)
        elif self.args.model == MODEL_CNN_MLP:
            cnn = CNN_MLP(learning_rate=0.0001)

        checkpoint = None
        if self.args.load_baseline_checkpoint:
            checkpoint = torch.load(self.args.load_baseline_checkpoint)
            cnn.load_state_dict(checkpoint['model_state_dict'])

        model = cnn.to(self.device)
        if self.args.use_weight_decay:
            if self.args.weight_decay_all:
                num_p = sum(p.numel() for p in model.parameters())
                weights = np.ones(num_p) * INIT_L2
                model.weight_decay = Variable(torch.FloatTensor(weights).to(self.device), requires_grad=True)
            else:
                weights = INIT_L2
                model.weight_decay = Variable(torch.FloatTensor([weights]).to(self.device), requires_grad=True)
            model.weight_decay = model.weight_decay.to(self.device)
        model.train()
        return model, checkpoint

    def load_finetuned_model(self, baseline_model):
        """
        Loads the augmentation net, sample reweighting net, and baseline model
        Note: sets all these models to train mode
        """
        # augment_net = Net(0, 0.0, 32, 3, 0.0, num_classes=32**2 * 3, do_res=True)
        if self.args.dataset == DATASET_MNIST:
            imsize, in_channel, num_classes = 28, 1, 10
        else:
            imsize, in_channel, num_classes = 32, 3, 10

        augment_net = UNet(in_channels=in_channel, n_classes=in_channel, depth=2, wf=3, padding=True, batch_norm=False,
                           do_noise_channel=True,
                           up_mode='upconv', use_identity_residual=True)  # TODO(PV): Initialize UNet properly
        # TODO (JON): DEPTH 1 WORKED WELL.  Changed upconv to upsample.  Use a wf of 2.

        # This ResNet outputs scalar weights to be applied element-wise to the per-example losses
        reweighting_net = Net(1, 0.0, imsize, in_channel, 0.0, num_classes=1)
        # resnet_cifar.resnet20(num_classes=1)

        if self.args.load_finetune_checkpoint:
            checkpoint = torch.load(self.args.load_finetune_checkpoint)
            # temp_baseline_model = baseline_model
            # baseline_model.load_state_dict(checkpoint['elementary_model_state_dict'])
            if 'weight_decay' in checkpoint:
                baseline_model.weight_decay = checkpoint['weight_decay']
            # baseline_model.weight_decay = temp_baseline_model.weight_decay
            # baseline_model.load_state_dict(checkpoint['elementary_model_state_dict'])
            augment_net.load_state_dict(checkpoint['augment_model_state_dict'])
            try:
                reweighting_net.load_state_dict(checkpoint['reweighting_model_state_dict'])
            except KeyError:
                pass

        augment_net, reweighting_net, baseline_model = augment_net.to(self.device), reweighting_net.to(self.device), baseline_model.to(self.device)
        augment_net.train(), reweighting_net.train(), baseline_model.train()
        return augment_net, reweighting_net, baseline_model

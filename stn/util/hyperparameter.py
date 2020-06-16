import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################################
# Miscellaneous functions
###############################################################################

def logit(x):
    return torch.log(x) - torch.log(1-x)

def s_logit(x, min=0, max=1):
    """Stretched logit function: Maps x lying in (min, max) to R"""
    return logit((x - min)/(max-min))

def s_sigmoid(x, min=0, max=1):
    """Stretched sigmoid function: Maps x lying in R to [min, max]"""
    return (max-min)*torch.sigmoid(x) + min

def inv_softplus(x):
    """ Inverse softplus function: Maps x lying in (0, infty) to R"""
    return torch.log(torch.exp(x) - 1)

def robustify(x, eps):
    """
    Adjusts x lying in an interval [a, b] so that it lies in [a+eps, b-eps]
    through a linear projection to that interval.
    """
    return (1-2*eps) * x + eps

def project(x, range_min, range_max):
    if range_min == -float('inf') and range_max == float('inf'):
        return x
    elif range_min == -float('inf') and range_max != float('inf'):
        return range_max - F.softplus(x)
    elif range_min != -float('inf') and range_max == float('inf'):
        return range_min + F.softplus(x)
    elif range_min != -float('inf') and range_max != float('inf'):
        return s_sigmoid(x, range_min, range_max)

def gaussian_cdf(x):
    """
    Computes cdf of standard normal distribution.

    Arguments:
        x (Tensor)
    """
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

###############################################################################
# Hyperparameter convenience functions
###############################################################################
class HyperparameterInfo():
    def __init__(self, index, range, discrete=False, minibatch=True):
        """
        Arguments:
            index (int): index of hyperparameter in htensor
            range (float, float): tuple specifying range hyperparameter must lie in
            discrete (bool): whether hyperparameter is discrete
            reinforce (bool): whether hyperparameter should be tuned using REINFORCE
            minibatch (bool): whether hyperparameter can use minibatched perturbations
        """
        self.index = index
        self.range = range
        self.discrete = discrete
        self.minibatch = minibatch

def perturb(htensor, hscale, batch_size, hdict=None):
    """
    Arguments:
        htensor (tensor): size is (H,)
        hscale (tensor): size is (H,)

    Returns:
        perturb_htensor (tensor): perturbed hyperparameters in unconstrained space,
            size is (B, H)
    """
    noise = htensor.new(batch_size, htensor.size(0)).normal_()
    perturb_htensor = htensor + F.softplus(hscale)*noise
    if hdict is not None:
        for hinfo in [hinfo for hinfo in hdict.values() if not hinfo.minibatch]:
            hidx = hinfo.index
            perturb_htensor[:, hidx] = perturb_htensor[0, hidx]
    return perturb_htensor

def hnet_transform(htensor, hdict):
    """
    Arguments:
        htensor (tensor): size is (B, H)
        hdict: dictionary mapping hyperparameter names to relevant info

    Returns:
        hnet_tensor (tensor): tensor of size (B, H) ready to be fed into hypernet
    """
    hnet_tensor_list = []
    for hinfo in hdict.values():
        hvalue = htensor[:,hinfo.index]
        hnet_tensor_list.append(hvalue)
    return torch.stack(hnet_tensor_list, dim=1)

def compute_entropy(hscale):
    """
    Arguments:
        hscale (tensor): size is (H,)

    Returns:
        entropy (tensor): returns scalar value of entropy of perturbation distribution
    """
    scale = F.softplus(hscale)
    return torch.sum(torch.log(scale * math.sqrt(2*math.pi*math.e)))

def hparam_transform(htensor, hdict):
    """
    Arguments:
        htensor (tensor): size is (B, H)
        hdict: dictionary mapping hyperparameter names to relevant info

    Returns:
        hparam_tensor (tensor): tensor ready to be used as actual hyperparameters
    """
    hparam_tensor_list = []
    for hinfo in hdict.values():
        range_min, range_max = hinfo.range
        hvalue = htensor[:,hinfo.index]
        hparam = project(hvalue, range_min, range_max)

        if hinfo.discrete:
            hparam = torch.floor(hparam)
        hparam_tensor_list.append(hparam)

    return torch.stack(hparam_tensor_list, dim=1)

def create_hparams(args, cnn_class, device):
    """
    Arguments:
        args: the arguments supplied by the user to the main script
        cnn_class: the convolutional net class
        device: device we are training on

    Returns:
        htensor: unconstrained reparametrization of the starting hyperparameters
        hscale: unconstrained reparametrization of the perturbation distribution's scale
        hdict: dictionary mapping hyperparameter names to info about the hyperparameter
    """
    hdict = OrderedDict()
    htensor_list = []
    hscale_list = []

    if args.tune_dropout:
        htensor_list.append(logit(torch.tensor(args.start_drop)))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['dropout'] = HyperparameterInfo(index=len(hdict), range=(0.,1.))

    if args.tune_dropoutl:
        for i in range(cnn_class.num_drops):
            htensor_list.append(logit(torch.tensor(args.start_drop)))
            hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
            hdict['dropout' + str(i)] = HyperparameterInfo(index=len(hdict),
                range=(0.,1.))

    if args.tune_hue or args.tune_jitters:
        htensor_list.append(logit(torch.tensor(2*args.start_hue)))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['hue'] = HyperparameterInfo(index=len(hdict), range=(0.,0.5))

    if args.tune_contrast or args.tune_jitters:
        htensor_list.append(logit(torch.tensor(args.start_contrast)))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['contrast'] = HyperparameterInfo(index=len(hdict), range=(0.,1.))

    if args.tune_sat or args.tune_jitters:
        htensor_list.append(logit(torch.tensor(args.start_sat)))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['sat'] = HyperparameterInfo(index=len(hdict), range=(0.,1.))

    if args.tune_bright or args.tune_jitters:
        htensor_list.append(logit(torch.tensor(args.start_bright)))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['bright'] = HyperparameterInfo(index=len(hdict), range=(0.,1.))

    if args.tune_indropout:
        htensor_list.append(logit(torch.tensor(args.start_indrop)))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['indropout'] = HyperparameterInfo(index=len(hdict), range=(0.,1.))

    if args.tune_inscale:
        htensor_list.append(s_logit(torch.tensor(args.start_inscale), min=0, max=0.3))
        hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
        hdict['inscale'] = HyperparameterInfo(index=len(hdict),range=(0.,0.3))

    if args.tune_cutlength:
        # Search over patch length of {0, 1, ..., 24}
        htensor_list.append(s_logit(torch.tensor(args.start_cutlength), min=0., max=25.))
        hscale_list.append(inv_softplus(torch.tensor(args.cutlength_scale)))
        hdict['cutlength'] = HyperparameterInfo(index=len(hdict), discrete=True,
            range=(0.,24.))

    if args.tune_cutholes:
        htensor_list.append(s_logit(torch.tensor(args.start_cutholes), min=0., max=4.))
        hscale_list.append(inv_softplus(torch.tensor(args.cutholes_scale)))
        hdict['cutholes'] = HyperparameterInfo(index=len(hdict), discrete=True,
            range=(0.,4.))

    if args.tune_fcdropout:
        for i in range(cnn_class.num_fcdrops):
            htensor_list.append(logit(torch.tensor(args.start_fcdrop)))
            hscale_list.append(inv_softplus(torch.tensor(args.perturb_scale)))
            hdict['fcdropout' + str(i)] = HyperparameterInfo(index=len(hdict),
                range=(0.,1.))

    htensor = nn.Parameter(torch.stack(htensor_list).to(device))
    hscale = nn.Parameter(torch.stack(hscale_list).to(device))
    return htensor, hscale, hdict

def create_hlabels(hdict, args):
    """Returns a tuple of the names of hyperparameters being tuned and their scales
    (if they are being tuned)."""
    hlabels = list(hdict.keys())
    if args.tune_scales:
        hlabels += [hlabel + '_scale' for hlabel in hlabels]
    hlabels = tuple(hlabels)
    return hlabels

def create_hstats(htensor, hscale, hdict, args):
    """Returns a dictionary mapping names of hyperparameters to their current values.
    """
    hstats = OrderedDict()

    for hname, hinfo in hdict.items():
        range_min, range_max = hinfo.range
        hstats[hname] = project(htensor[hinfo.index], range_min, range_max).item()
        if args.tune_scales:
            hstats[hname + '_scale'] = F.softplus(hscale[hinfo.index]).item()
    return hstats
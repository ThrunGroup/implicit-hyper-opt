import sys
import ipdb
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, use_device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    data = data.to(use_device)
    return data


def get_batch(source, i, args, seq_len=None, flatten_targets=True):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len])
    if flatten_targets:
        target = Variable(source[i+1:i+1+seq_len].view(-1))
    else:
        target = Variable(source[i+1:i+1+seq_len])
    return data, target


def inv_softplus(x):
    """ Inverse softplus function: Maps x lying in (0, infty) to R"""
    if isinstance(x, float):
        return math.log(math.exp(x) - 1)
    else:
        return torch.log(torch.exp(x) - 1)


def logit(x):
    if isinstance(x, float):
        return math.log(x) - math.log(1-x)
    else:
        return torch.log(x) - torch.log(1-x)


def gumbel_binary(theta, temperature=0.5, hard=False):
    """theta is a vector of unnormalized probabilities
    Returns:
        A vector that becomes binary as the temperature --> 0
    """
    u = Variable(torch.rand(theta.size(), device=theta.device))
    z = theta + torch.log(u / (1 - u))
    a = F.sigmoid(z / temperature)

    if hard:
        a_hard = torch.round(a)
        return (a_hard - a).detach() + a
    else:
        return a


# class ConcreteDropout(nn.Module):
#     def __init__(self):
#         super(ConcreteDropout, self).__init__()

#     def forward(self, x, p_logit):
#         if not self.training:
#             return x

#         p = torch.sigmoid(p_logit)

#         eps = 1e-7
#         temp = 0.1

#         # unif_noise = torch.rand_like(x)
#         unif_noise = torch.rand((1, x.size(1), x.size(2)), device=x.device)

#         drop_prob = (torch.log(p + eps) - torch.log(1 - p + eps) + torch.log(unif_noise + eps) - torch.log(1 - unif_noise + eps))
#         drop_prob = torch.sigmoid(drop_prob / temp)
#         mask = 1 - drop_prob
#         retain_prob = 1 - p

#         mask = mask / retain_prob
#         mask = mask.expand_as(x)
#         return x * mask


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout_logit=0.5):
        if not self.training or not dropout_logit:
            return x

        drop_prob = torch.sigmoid(dropout_logit)

        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - drop_prob)
        mask = Variable(m.div_(1 - drop_prob), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


class GumbelSoftmaxLockedDropout(nn.Module):
    def __init__(self):
        super(GumbelSoftmaxLockedDropout, self).__init__()

    def forward(self, x, dropout):
        if not self.training:
            return x

        theta = logit(1 - torch.sigmoid(dropout).repeat(1, x.size(1), x.size(2)))
        # theta = dropout.repeat(1, x.size(1), x.size(2))  # Assumes that dropout is already unnormalized logits
        mask = gumbel_binary(theta, temperature=0.5, hard=True)
        mask = mask / (1 - torch.sigmoid(dropout))
        mask = mask.expand_as(x)
        # m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        # mask = Variable(m.div_(1 - dropout), requires_grad=False)
        # mask = mask.expand_as(x)
        return mask * x


# def mask2d(B, D, keep_prob, cuda=True):
#     m = torch.floor(torch.rand(B, D) + keep_prob) / keep_prob
#     m = Variable(m, requires_grad=False)
#     if cuda:
#         m = m.cuda()
#     return m


class ConcreteDropout(nn.Module):
    def __init__(self):
        super(ConcreteDropout, self).__init__()

    def forward(self, x, p_logit):
        if not self.training:
            return x

        p = torch.sigmoid(p_logit)

        # ipdb.set_trace()

        eps = 1e-7
        temp = 0.1

        # unif_noise = torch.rand_like(x)
        unif_noise = torch.rand((1, x.size(1), x.size(2)), device=x.device)  # (1, 40, 650)

        drop_prob = (torch.log(p + eps) - torch.log(1 - p + eps) + torch.log(unif_noise + eps) - torch.log(1 - unif_noise + eps))
        drop_prob = torch.sigmoid(drop_prob / temp)
        mask = 1 - drop_prob
        retain_prob = 1 - p

        mask = mask / retain_prob
        mask = mask.expand_as(x)  # (70, 40, 650)

        return x * mask

import sys
import ipdb

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Local imports
from weight_drop import WeightDrop
# from rnn_utils import ConcreteDropout
# from locked_dropout import LockedDropout
from rnn_utils import LockedDropout, ConcreteDropout
from embed_regularize import embedded_dropout


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropouto=0., dropouth=0., dropouti=0., dropoute=0., wdrop=0.,
                 tie_weights=True, wdecay=0.0, wdecay_type='global', dropout_type='standard'):
        super(RNNModel, self).__init__()

        if dropout_type in ['concrete', 'per_param']:
            self.lockdrop = ConcreteDropout()
        elif dropout_type == 'standard':
            self.lockdrop = LockedDropout()

        self.encoder = nn.Embedding(ntoken, ninp, sparse=True)
        # self.encoder = nn.Embedding(ntoken, ninp)

        if rnn_type == 'LSTM':
            self.rnns = [DropconnectCell(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), wdrop=wdrop) for l in range(nlayers)]
            # if wdrop:
            #     print("Using weight drop {}".format(wdrop))

        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights
        if tie_weights:
            print("Tie weights")
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        if wdecay_type == 'global':
            self.weight_decay = torch.tensor([math.log(wdecay)], requires_grad=True, device='cuda:0')
            # self.weight_decay = torch.tensor([math.log(wdecay)], requires_grad=False, device='cuda:0')
            # self.weight_decay = torch.tensor([0.0], device='cuda:0')
        elif wdecay_type == 'per_layer':
            weights = np.ones(nlayers, dtype='float32') * math.log(wdecay)
            self.weight_decay = torch.tensor(weights, requires_grad=True, device='cuda:0')
        elif wdecay_type == 'per_param':
            num_p = sum(p.numel() for p in self.parameters())
            weights = np.ones(num_p, dtype='float32') * math.log(wdecay)
            self.weight_decay = torch.tensor(weights, requires_grad=True, device='cuda:0')

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.ntoken = ntoken
        self.nlayers = nlayers
        self.dropouto = dropouto
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.wdrop = wdrop
        self.tie_weights = tie_weights
        self.wdecay_type = wdecay_type

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def emb_drop(self, p_logit):
        if not self.training:
            return self.encoder.weight

        p = torch.sigmoid(p_logit)

        eps = 1e-7
        temp = 0.1
        unif_noise = torch.rand((self.encoder.weight.shape[0]), device='cuda:0')

        drop_prob = (torch.log(p + eps) - torch.log(1 - p + eps) + torch.log(unif_noise + eps) - torch.log(1 - unif_noise + eps))
        drop_prob = torch.sigmoid(drop_prob / temp)
        mask = 1 - drop_prob
        retain_prob = 1 - p
        mask = mask / retain_prob  # (10000)

        emb_weights = self.encoder.weight * mask.unsqueeze(1)  # mask.unsqueeze(1) gives (10000, 1)
        return emb_weights  # (10000, 650)

    def forward(self, input, hidden, return_h=False):

        # emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        dropped_emb_weights = self.emb_drop(self.dropoute)
        emb = F.linear(one_hot(input), weight=dropped_emb_weights.t())  # TODO(PV): Need to manually incorporate dropout here!
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output

            rnn.h2h.mask_weights(wdrop=self.wdrop)
            rnn.i2h.mask_weights(wdrop=None)

            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropouto)
        outputs.append(output)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()),
                    Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()))
                    for l in range(self.nlayers)]

    def L2_loss(self):
        if self.wdecay_type == 'global':
            loss = 0
            for p in self.parameters():
                loss = loss + torch.sum(torch.mul(p, p))
            return loss * torch.exp(self.weight_decay)
        elif self.wdecay_type == 'per_layer':
            loss = 0
            for i, rnn in enumerate(self.rnns):
                # Apply the per-layer weight decay to the weights (not biases) of the i2h and h2h weight matrices
                loss = loss + torch.sum(torch.mul(torch.exp(self.weight_decay[i]), torch.flatten(torch.mul(self.rnns[i].i2h.elem_weight, self.rnns[i].i2h.elem_weight))))
                loss = loss + torch.sum(torch.mul(torch.exp(self.weight_decay[i]), torch.flatten(torch.mul(self.rnns[i].h2h.elem_weight, self.rnns[i].h2h.elem_weight))))
            return loss
        elif self.wdecay_type == 'per_param':
            loss = 0
            count = 0
            for p in self.parameters():
                loss = loss + torch.sum(torch.mul(torch.exp(self.weight_decay[count:count+p.numel()]), torch.flatten(torch.mul(p, p))))
                count += p.numel()
            return loss


class DropconnectLinear(nn.Module):
    def __init__(self, input_dim, output_dim, wdrop=0):
        super(DropconnectLinear, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.wdrop = wdrop

        self.elem_weight_raw = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.elem_bias = nn.Parameter(torch.Tensor(output_dim))

        self.init_params()

    # def mask_weights(self, wdrop):
    #     if self.training:
    #         m = self.elem_weight_raw.data.new(self.elem_weight_raw.size()).bernoulli_(1 - wdrop)
    #         mask = Variable(m, requires_grad=False) / (1 - wdrop)
    #     else:
    #         m = self.elem_weight_raw.data.new(self.elem_weight_raw.size()).fill_(1)  # All 1's (nothing dropped) at test-time
    #         mask = Variable(m, requires_grad=False)

    #     self.elem_weight = self.elem_weight_raw * mask

    def mask_weights(self, wdrop=None):
        if self.training and wdrop is not None:
            p = torch.sigmoid(wdrop)

            eps = 1e-7
            temp = 0.1
            unif_noise = torch.rand(self.elem_weight_raw.size(), device='cuda:0')

            drop_prob = (torch.log(p + eps) - torch.log(1 - p + eps) + torch.log(unif_noise + eps) - torch.log(1 - unif_noise + eps))
            drop_prob = torch.sigmoid(drop_prob / temp)
            mask = 1 - drop_prob
            retain_prob = 1 - p
            mask = mask / retain_prob

            # m = self.elem_weight_raw.data.new(self.elem_weight_raw.size()).bernoulli_(1 - wdrop)
            # mask = Variable(m, requires_grad=False) / (1 - p)
        else:
            m = self.elem_weight_raw.data.new(self.elem_weight_raw.size()).fill_(1)  # All 1's (nothing dropped) at test-time
            mask = Variable(m, requires_grad=False)

        self.elem_weight = self.elem_weight_raw * mask

    def forward(self, input):
        return F.linear(input, self.elem_weight, self.elem_bias)

    def init_params(self):
        # Initialize elementary parameters.
        n = self.input_dim
        stdv = 1. / math.sqrt(n)
        self.elem_weight_raw.data.uniform_(-stdv, stdv)
        self.elem_bias.data.uniform_(-stdv, stdv)


class DropconnectCell(nn.Module):
    """Cell for dropconnect RNN."""

    def __init__(self, ninp, nhid, wdrop=0):
        super(DropconnectCell, self).__init__()

        self.ninp = ninp
        self.nhid = nhid
        self.wdrop = wdrop

        self.i2h = DropconnectLinear(ninp, 4*nhid, wdrop=0)
        self.h2h = DropconnectLinear(nhid, 4*nhid, wdrop=wdrop)

    def forward(self, input, hidden):

        hidden_list = []

        nhid = self.nhid

        h, cell = hidden

        # Loop over the indexes in the sequence --> process each index in parallel across items in the batch
        for i in range(len(input)):

            h = h.squeeze()
            cell = cell.squeeze()

            x = input[i]

            x_components = self.i2h(x)
            h_components = self.h2h(h)

            preactivations = x_components + h_components

            gates_together = torch.sigmoid(preactivations[:, 0:3*nhid])
            forget_gate = gates_together[:, 0:nhid]
            input_gate = gates_together[:, nhid:2*nhid]
            output_gate = gates_together[:, 2*nhid:3*nhid]
            new_cell = torch.tanh(preactivations[:, 3*nhid:4*nhid])

            cell = forget_gate * cell + input_gate * new_cell
            h = output_gate * torch.tanh(cell)

            hidden_list.append(h)

        hidden_stacked = torch.stack(hidden_list)

        return hidden_stacked, (h.unsqueeze(0), cell.unsqueeze(0))


def one_hot(x, dimensionality=10000):
    # Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
    y = x.view(-1, 1)
    y_onehot = torch.zeros((y.size(0), dimensionality), device=x.device)
    y_onehot.scatter_(1, y, 1)
    return y_onehot.view(x.size(0), x.size(1), -1)

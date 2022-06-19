# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/distributions.py

import torch
import torch.nn as nn

from utils.model import AddBias

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: \
    log_prob_cat(self, actions.squeeze(-1))
FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: \
    log_prob_normal(self, actions).sum(-1, keepdim=False)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


class Categorical(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)

class Heatmap(nn.Module):

    def __init__(self):
        super(Heatmap, self).__init__()

    def forward(self, x):
        return FixedCategorical(probs=x)

class MultiCategorical:
    def __init__(self, logits):
        self.batch_size = len(logits)
        self.multi = [logit.shape[0] for logit in logits]
        assert max(self.multi) == min(self.multi)
        self.multi = self.multi[0]
        self.n_class = [logit.shape[1] for logit in logits]
        self.dist = [FixedCategorical(logits=logit) for logit in logits]

    def sample(self):
        return torch.stack([d.sample() for d in self.dist])

    def mode(self):
        return torch.stack([d.mode().squeeze(-1) for d in self.dist])

    def log_probs(self, actions):
        assert actions.shape == (self.batch_size, self.multi)
        assert torch.tensor([d.probs.shape[1] > actions[b].max().item() for b, d in enumerate(self.dist)]).all()
        return torch.stack([d.log_probs(actions[b]).sum() for b, d in enumerate(self.dist)])

    def entropy(self):
        return torch.stack([d.entropy().sum() for b, d in enumerate(self.dist)])
        

class MultiHeatmap(nn.Module):

    def __init__(self):
        super(MultiHeatmap, self).__init__()

    def forward(self, x):
        return MultiCategorical(x)


class DiagGaussian(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, (action_logstd - 1).exp())

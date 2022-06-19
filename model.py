import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
import torch.optim as optim

from utils.distributions import Categorical, DiagGaussian, Heatmap, MultiHeatmap
from utils.model import get_grid, Flatten, NNBase
from utils.gnn import GNN

class ANS_Policy(nn.Module):

    def __init__(self, input_shape, **kwargs):
        super(ANS_Policy, self).__init__()

        # self.bias = 1 / (input_shape[1] / 8. * input_shape[2] / 8.)
        out_size = int(input_shape[1] / 8. * input_shape[2] / 8.)

        self.is_recurrent = False
        self.rec_state_size = 1
        self.output_size = 256

        hidden_size = 512

        Conv2d = nn.Conv2d

        self.actor = nn.Sequential(
            nn.Conv2d(9, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )

        self.critic = nn.ModuleList(
            [
                nn.Linear(out_size * 32 + 8, hidden_size),
                nn.Linear(hidden_size, self.output_size),
                nn.Linear(self.output_size, 1),
                nn.Embedding(72, 8)
            ]
        )

        self.downscaling = 2
        self.train()

    def forward(self, inputs, rnn_hxs, masks, extras):
        x = self.actor(inputs)
        orientation_emb = self.critic[3](extras[:, -1]).squeeze(1)
        x = torch.cat((x, orientation_emb), 1)

        x = nn.ReLU()(self.critic[0](x))
        x = nn.ReLU()(self.critic[1](x))

        return self.critic[2](x).squeeze(-1), x, rnn_hxs

# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L15
class RL_Policy(nn.Module):

    def __init__(self, obs_shape, action_space, model_type='gconv',
                 base_kwargs=None, lr=None, eps=None):

        super(RL_Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if model_type == 'ans':
            self.network = ANS_Policy(obs_shape, **base_kwargs)
        elif model_type == 'gnn':
            self.network = GNN(obs_shape, base_kwargs.get('num_gnn_layer') * ['self', 'cross'], base_kwargs.get('use_history'), base_kwargs.get('ablation'))
        else:
            raise NotImplementedError

        if model_type == 'gnn':
            assert action_space.__class__.__name__ == "Box"
            self.dist = MultiHeatmap()
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.network.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.actor_optimizer = optim.Adam(set(filter(lambda p: p.requires_grad,
            self.network.actor.parameters())).union(filter(lambda p: p.requires_grad,
            self.dist.parameters())), lr=lr[0], eps=eps)
        self.critic_optimizer = optim.Adam(filter(lambda p: p.requires_grad,
            self.network.critic.parameters()), lr=lr[0] * lr[1], eps=eps)

        self.model_type = model_type

    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return self.network.rec_state_size

    @property
    def downscaling(self):
        return self.network.downscaling

    def forward(self, inputs, rnn_hxs, masks, extras):
        if extras is None:
            return self.network(inputs, rnn_hxs, masks)
        else:
            return self.network(inputs, rnn_hxs, masks, extras)

    def act(self, inputs, rnn_hxs, masks, extras=None, deterministic=False):
        
        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs, actor_features

    def get_value(self, inputs, rnn_hxs, masks, extras=None):
        value, actor_features, _ = self(inputs, rnn_hxs, masks, extras)
        return value, actor_features

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, extras=None):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs, actor_features

    def load(self, path, device):
        self.actor_optimizer = optim.Adam(set(filter(lambda p: p.requires_grad,
            self.network.actor.parameters())).union(filter(lambda p: p.requires_grad,
            self.dist.parameters())), lr=1e-3)
        self.critic_optimizer = optim.Adam(filter(lambda p: p.requires_grad,
            self.network.critic.parameters()), lr=1e-3)
        # state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        state_dict = torch.load(path, map_location=device)
        self.network.load_state_dict(state_dict['network'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        del state_dict

    def load_critic(self, path, device):
        state_dict = torch.load(path, map_location=device)['network']
        self.network.critic.load_state_dict({k.replace('critic.', ''):v for k,v in state_dict.items() if 'critic' in k})
        # self.network.actor.load_state_dict({k.replace('actor.', ''):v for k,v in state_dict.items() if 'actor' in k})
        del state_dict

    def save(self, path):
        state = {
            'network': self.network.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }
        torch.save(state, path)

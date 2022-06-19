# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from copy import deepcopy
import torch
from torch import nn
from utils.model import Flatten
import numpy as np


def MLP(channels, do_bn=False):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)



class Encoder(nn.Module):
    """ Joint encoding of semantic tag and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([4] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)
        self.dist_encoder = MLP([1, feature_dim, feature_dim])

    def forward(self, inputs, dist, pos_history, goal_history, extras):
        inputs = inputs[:, 1, :, :]
        sz = inputs.size(1)
        frontier_idxs = []
        frontier_batches = []
        agent_batches = []
        dist_batches = []
        phistory_idxs = []
        phistory_batches = []
        ghistory_idxs = []
        ghistory_batches = []
        for b in range(inputs.size(0)):
            frontier = torch.nonzero(inputs[b, :, :])
            frontier_idxs.append(frontier)
            # dist_feat: n_agent x frontier
            dist_feat = dist[b, :, :, :][(inputs[b, :, :] > 0).unsqueeze(0).repeat(dist.size(1), 1, 1)].view(1, 1, dist.size(1) * frontier.size(0))
            dist_batches.append(dist_feat)
            
            pts = inputs.new_zeros((1, frontier.size(0), 4))
            pts[0, :, :2] = (frontier.float() - sz // 2) / (sz * 0.7)
            pts[0, :, 2] = 1
            frontier_batches.append(pts.transpose(1, 2))

            if pos_history is not None:
                phistory_pos = torch.nonzero(pos_history[b, :, :])
                phistory_idxs.append(phistory_pos)
                
                pts = pos_history.new_zeros((1, phistory_pos.size(0), 4))
                pts[0, :, :2] = (phistory_pos.float() - sz // 2) / (sz * 0.7)
                pts[0, :, 3] = 1
                phistory_batches.append(pts.transpose(1, 2))

            if goal_history is not None:
                ghistory_pos = torch.nonzero(goal_history[b, :, :])
                ghistory_idxs.append(ghistory_pos)
                
                pts = goal_history.new_zeros((1, ghistory_pos.size(0), 4))
                pts[0, :, :2] = (ghistory_pos.float() - sz // 2) / (sz * 0.7)
                pts[0, :, 2] = 1
                ghistory_batches.append(pts.transpose(1, 2))

            pts = inputs.new_zeros((1, extras.size(1), 4))
            pts[0, :, :2] = (extras[b].float() - sz // 2) / (sz * 0.7)
            pts[0, :, 3] = 1
            agent_batches.append(pts.transpose(1, 2))
        
        return (
            frontier_idxs,
            phistory_idxs if pos_history is not None else ([None] * len(frontier_idxs)),
            ghistory_idxs if goal_history is not None else ([None] * len(frontier_idxs)),
            [self.dist_encoder(batch) for batch in dist_batches],
            [self.encoder(batch) for batch in frontier_batches],
            [self.encoder(batch) for batch in agent_batches],
            [(self.encoder(batch) if batch.size(-1) > 0 else None) for batch in phistory_batches] if pos_history is not None else ([None] * len(frontier_idxs)),
            [(self.encoder(batch) if batch.size(-1) > 0 else None) for batch in ghistory_batches] if goal_history is not None else ([None] * len(frontier_idxs))
        )


class MLPAttention(nn.Module):
    def __init__(self, desc_dim):
        super().__init__()
        self.mlp = MLP([desc_dim * 3, desc_dim, 1])

    def forward(self, query, key, value, dist, mask):
        '''query: 1 x 128 x n_agent
        key: 1 x 128 x n_frontier
        dist: 1 x 128 x (n_agent x n_frontier)

        cat: 1 x 384 x (n_agent x n_frontier)

        value: 1 x 128 x n_frontier

        scores: 1 x n_agent x n_frontier

        output: n_agent x 128'''

        nq, nk = query.size(-1), key.size(-1)
        scores = self.mlp(torch.cat((
            query.view(1, -1, nq, 1).repeat(1, 1, 1, nk).view(1, -1, nq * nk),
            key.view(1, -1, 1, nk).repeat(1, 1, nq, 1).view(1, -1, nq * nk),
            dist), dim=1)).view(1, nq, nk)
        if mask is not None:
            if type(mask) is float:
                scores_detach = scores.detach()
                scale = torch.clamp(mask / (scores_detach.max(2).values - scores_detach.median(2).values), 1., 1e3)
                scores = scores * scale.unsqueeze(-1).repeat(1, 1, nk)
            else:
                scores = scores + (scores.min().detach() - 20) * (~mask).float().view(1, nq, nk)
        prob = scores.softmax(dim=-1)
        return torch.einsum('bnm,bdm->bdn', prob, value), scores
        

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def attention(self, query, key, value, mask):
        dim = query.shape[1]
        scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
        if mask is not None:
            scores = scores + (scores.min().detach() - 20) * (~mask).float().unsqueeze(0).unsqueeze(0).repeat(1, self.num_heads, 1, 1)
        prob = torch.nn.functional.softmax(scores, dim=-1)
        return torch.einsum('bhnm,bdhm->bdhn', prob, value), scores

    def forward(self, query, key, value, dist, mask):
        query, key, value = [l(x).view(1, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, scores = self.attention(query, key, value, mask)
        return self.merge(x.contiguous().view(1, self.dim*self.num_heads, -1)), scores.mean(1)


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, type: str):
        super().__init__()
        self.attn = MLPAttention(feature_dim) if type == 'cross' else MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, dist, mask):
        message, weights = self.attn(x, source, source, dist, mask)
        return self.mlp(torch.cat([x, message], dim=1)), weights


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, use_history: bool, ablation: int):
        super().__init__()
        self.attn = nn.ModuleList([AttentionalPropagation(feature_dim, 4, type) for type in layer_names])
        if use_history:
            self.phattn = nn.ModuleList([AttentionalPropagation(feature_dim, 4, 'self') for type in layer_names])
            self.ghattn = nn.ModuleList([AttentionalPropagation(feature_dim, 4, 'self') for type in layer_names])
        else:
            self.phattn = [None for type in layer_names]
            self.ghattn = [None for type in layer_names]
        # self.attn = MLP([feature_dim, 1])
        self.use_history = use_history
        self.score_layer = MLP([2*feature_dim, feature_dim, 1])
        self.names = layer_names
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        self.ablation = ablation

    def forward(self, desc0, desc1, desc2, desc3, lmb, fidx, phidx, ghidx, dist, unreachable):
        # desc0: frontier
        # desc1: agent
        # fidx: n_frontier x 2
        # lmb: n_agent x 4

        if self.ablation != 2:

            dist0 = dist.view(-1, desc1.size(-1), desc0.size(-1)).transpose(1, 2).reshape(1, -1, desc1.size(-1) * desc0.size(-1))
            dist1 = dist


            for idx, attn, phattn, ghattn, name in zip(range(len(self.names)), self.attn, self.phattn, self.ghattn, self.names):

                if name == 'cross':
                    src0, src1 = desc1, desc0
                else:
                    src0, src1 = desc0, desc1

                delta0, score0 = attn(desc0, src0, dist0, None)
                delta1, score1 = attn(desc1, src1, dist1, None)

                if self.use_history:
                    if name == 'cross':
                        if desc2 is not None:
                            delta21, _ = phattn(desc2, desc1, None, None)
                            delta12, _ = phattn(desc1, desc2, None, None)
                            desc2 = desc2 + delta21
                        else:
                            delta12 = 0
                        if desc3 is not None:
                            delta03, _ = ghattn(desc0, desc3, None, None)
                            delta30, _ = ghattn(desc3, desc0, None, None)
                            desc3 = desc3 + delta30
                        else:
                            delta03 = 0
                        desc0, desc1 = (desc0 + delta0 + delta03), (desc1 + delta1 + delta12)
                    else:  # if name == 'self':
                        if desc2 is not None:
                            delta2, _ = phattn(desc2, desc2, None, None)
                            desc2 = desc2 + delta2
                        if desc3 is not None:
                            delta3, _ = ghattn(desc3, desc3, None, None)
                            desc3 = desc3 + delta3
                        desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
                else:
                    desc0, desc1 = (desc0 + delta0), (desc1 + delta1)

            
        # weights1: n_agent x n_frontier
        fidx = torch.repeat_interleave(fidx.view(1, fidx.size(0), 2), repeats=lmb.size(0), dim=0)
        lmb = torch.repeat_interleave(lmb.view(lmb.size(0), 1, 4), repeats=fidx.size(1), dim=1)
        # fidx: n_agent x n_frontier x 2
        # lmb: n_agent x n_frontier x 4
        # unreachable: n_agent x n_frontier
        invalid = ((fidx < lmb[:, :, [0,2]]) | (fidx >= lmb[:, :, [1,3]])).any(2)
        # assert (~invalid).any(1).all()
        if self.ablation == 1:
            scores = self.score_layer(torch.cat((
                torch.repeat_interleave(desc1, repeats=unreachable.size(1), dim=-1),
                desc0.repeat(1, 1, unreachable.size(0))
            ), dim=1)).view(1, *unreachable.shape)
        elif self.ablation == 2:
            scores = 2 / (dist.view(1, *unreachable.shape) + 1e-3)
        else:
            scores = score1
        scores = log_optimal_transport(scores.log_softmax(dim=-2), self.bin_score, iters=5)[:, :-1, :-1].view(unreachable.shape)
        score_min = scores.min() - scores.max()
        scores = scores + (score_min - 40) * invalid.float() + (score_min - 20) * unreachable.float()

        return scores * 15



class Actor(nn.Module):
    def __init__(self, desc_dim, gnn_layers, use_history, ablation):
        super().__init__()
        self.kenc = Encoder(desc_dim, [32, 64, 128, 256])
        self.gnn = AttentionalGNN(desc_dim, gnn_layers, use_history, ablation)
        self.ablation = ablation

    def forward(self, inputs, dist, pos_history, goal_history, extras):
        # MLP encoder.
        extras = extras.view(inputs.size(0), -1, 6)
        unreachable = [
            dist[b, :, :, :][(inputs[b, 1, :, :] > 0).unsqueeze(0).repeat(dist.size(1), 1, 1)].view(dist.size(1), -1) > 2
            for b in range(inputs.size(0))
        ]

        if self.ablation == 2:
            idxs, phidxs, ghidxs, _, desc0s, desc1s, desc2s, desc3s = self.kenc(inputs, dist, pos_history, goal_history, extras[:, :, :2])
            dist = [
                dist[b, :, :, :][(inputs[b, 1, :, :] > 0).unsqueeze(0).repeat(dist.size(1), 1, 1)].view(dist.size(1), -1)
                for b in range(inputs.size(0))
            ]
        else:
            idxs, phidxs, ghidxs, dist, desc0s, desc1s, desc2s, desc3s = self.kenc(inputs, dist, pos_history, goal_history, extras[:, :, :2])

        # Multi-layer Transformer network.
        return [self.gnn(desc0s[b], desc1s[b], desc2s[b], desc3s[b], extras[b, :, 2:], idxs[b], phidxs[b], ghidxs[b], dist[b], unreachable[b]) for b in range(inputs.size(0))]



class GNN(nn.Module):
    def __init__(self, input_shape, gnn_layers, use_history, ablation):
        super().__init__()
        self.output_size = 0
        self.is_recurrent = False
        self.rec_state_size = 1
        self.downscaling = 1
        # desc_dim = 128
        desc_dim = 32
        self.actor = Actor(desc_dim, gnn_layers, use_history, ablation)
        # self.critic = Critic(desc_dim)
        out_size = int(input_shape[1] / 8. * input_shape[2] / 8.)
        self.critic = nn.Sequential(
            nn.Conv2d(6, 64, 6, stride=2, padding=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 6, stride=2, padding=2),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 5, stride=1, padding=2),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 6, stride=2, padding=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, 5, stride=1, padding=2),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(out_size * 16, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.train()

    def forward(self, inputs, rnn_hxs, masks, extras):
        value = self.critic(inputs[:, :6, :, :]).squeeze(-1) - 1.5
        actor_features = self.actor(inputs[:, :6, :, :], inputs[:, 8:, :, :], inputs[:, 6, :, :], inputs[:, 7, :, :], extras)
        return value, actor_features, rnn_hxs

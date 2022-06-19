# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/storage.py

from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import logging


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

def get_rotation_mat(theta):
    theta = torch.tensor(theta * 3.14159265359 / 180.)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0.],
                         [torch.sin(theta), torch.cos(theta), 0.]]).cuda()


def rotate_tensor(x, theta):
    rot_mat = torch.repeat_interleave(get_rotation_mat(theta).unsqueeze(0), repeats=x.size(0), dim=0)
    grid = F.affine_grid(rot_mat, x.shape)
    x = F.grid_sample(x, grid)
    return x

def rotate_scalar(x, theta, map_size):
    origin = (map_size - 1) / 2
    theta = torch.tensor(theta * 3.14159265359 / 180.)
    x, y = (x // map_size - origin).float(), (x % map_size - origin).float()
    x, y = torch.cos(theta) * x - torch.sin(theta) * y + origin, torch.sin(theta) * x + torch.cos(theta) * y + origin
    x, y = torch.clamp(x.long(), 0, map_size - 1), torch.clamp(y.long(), 0, map_size - 1)
    return x * map_size + y

    


class RolloutStorage(object):

    def __init__(self, num_steps, num_processes, eval_freq, num_repeats, num_robots, obs_shape, action_space, rec_state_size):

        if action_space.__class__.__name__ == 'Discrete':
            self.n_actions = 1
            self.map_size = int(action_space.n ** 0.5)
            action_type = torch.long
        else:
            self.map_size = 0
            self.n_actions = action_space.shape[0]
            action_type = torch.float32 if action_space.dtype == 'float32' else torch.long

        self.obs = torch.zeros(num_steps + 1, num_processes * num_robots * num_repeats, *obs_shape)
        self.rec_states = torch.zeros(num_steps + 1, num_processes * num_robots * num_repeats, rec_state_size)
        self.rewards = torch.zeros(num_steps, num_processes * num_robots * num_repeats)
        self.value_preds = torch.zeros(num_steps + 1, num_processes * num_robots * num_repeats)
        self.returns = torch.zeros(num_steps + 1, num_processes * num_robots * num_repeats)
        self.action_log_probs = torch.zeros(num_steps, num_processes * num_robots * num_repeats)
        self.actions = torch.zeros((num_steps, num_processes * num_robots * num_repeats, self.n_actions), dtype=action_type)
        self.masks = torch.ones(num_steps + 1, num_processes * num_robots * num_repeats)
        self.open = torch.ones(num_processes * num_robots * num_repeats).bool()

        self.num_mini_step = num_repeats
        self.mini_step_size = num_processes * num_robots
        self.num_steps = num_steps
        self.step = 0
        self.mini_step = 0
        self.has_extras = False
        self.extras_size = None
        self.first_use_to_eval = (eval_freq == num_repeats and num_repeats > 1)

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rec_states = self.rec_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.open = self.open.to(device)
        if self.has_extras:
            self.extras = self.extras.to(device)
        return self

    def insert(self, obs, rec_states, actions, action_log_probs, value_preds,
               rewards, masks):
        l, h = self.mini_step * self.mini_step_size, (self.mini_step + 1) * self.mini_step_size
        if self.step == 0:
            ll, lh = l-self.mini_step_size, h-self.mini_step_size
            if lh == 0:
                lh = self.mini_step_size * self.num_mini_step
            self.obs[0][l:h].copy_(self.obs[-1][ll:lh])
            self.rec_states[0][l:h].copy_(self.rec_states[-1][ll:lh])
        self.obs[self.step + 1][l:h].copy_(obs)
        self.rec_states[self.step + 1][l:h].copy_(rec_states)
        self.actions[self.step][l:h].copy_(actions.view(-1, self.n_actions))
        self.action_log_probs[self.step][l:h].copy_(action_log_probs)
        self.value_preds[self.step][l:h].copy_(value_preds)
        self.rewards[self.step][l:h].copy_(rewards)
        self.masks[self.step + 1][l:h].copy_(masks)
        self.open[l:h] = self.open[l:h] & masks.bool()

        self.step = (self.step + 1) % self.num_steps
        if self.step == 0:
            self.mini_step = (self.mini_step + 1) % self.num_mini_step

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.rec_states[0].copy_(self.rec_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.open[:] = True
        if self.has_extras:
            self.extras[0].copy_(self.extras[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma \
                        * self.value_preds[step + 1] * self.masks[step + 1] \
                        - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma \
                                     * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch, max_batch_size, rotation_augmentation, ds=1, verbose=True):
        num_steps = self.num_steps
        num_processes = self.mini_step_size * self.num_mini_step
        batch_size = num_processes * num_steps
        batch_begin = self.mini_step_size * num_steps if self.first_use_to_eval else 0
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps,
                      num_mini_batch))
        idx = [i for i in range(batch_begin, batch_size) if self.masks[i // num_processes, i % num_processes]]
        # idx = range(batch_size)
        if verbose:
            logging.info(f"actual-batch-size: {len(idx)}/{batch_size - batch_begin}")
            logging.info(f"open-ratio: {self.open.sum()}/{self.open.size(0)}")
        mini_batch_size = len(idx) // num_mini_batch
        if max_batch_size > 0:
            mini_batch_size = min(max_batch_size, mini_batch_size)
        if mini_batch_size > 0:
            sampler = BatchSampler(SubsetRandomSampler(idx),
                                mini_batch_size, drop_last=False)

            for idx, indices in enumerate(sampler):
                if idx >= num_mini_batch:
                    break
                raw_data = {
                    'obs': self.obs[:-1].view(-1, *self.obs.size()[2:])[indices],
                    'rec_states': self.rec_states[:-1].view(-1,
                                                            self.rec_states.size(-1))[indices],
                    'actions': self.actions.view(-1, self.n_actions)[indices],
                    'value_preds': self.value_preds[:-1].view(-1)[indices],
                    'returns': self.returns[:-1].view(-1)[indices],
                    'masks': self.masks[:-1].view(-1)[indices],
                    'old_action_log_probs': self.action_log_probs.view(-1)[indices],
                    'adv_targ': advantages.view(-1)[indices],
                    'extras': self.extras[:-1].view(-1, self.extras_size)[indices] \
                        if self.has_extras else None,
                    'augmentation': False
                }
                yield raw_data
                if rotation_augmentation > 1:
                    raw_data['augmentation'] = True
                    raw_obs = raw_data['obs']
                    raw_actions = raw_data['actions']
                    for i in range(1, rotation_augmentation):
                        raw_data['obs'] = rotate_tensor(raw_obs, i * 360. / rotation_augmentation)
                        raw_data['actions'] = rotate_scalar(raw_actions, i * 360. / rotation_augmentation, self.map_size // ds)
                        x = raw_data['actions'] // (self.map_size // ds)
                        y = raw_data['actions'] % (self.map_size // ds)
                        raw_data['obs'][:, 1, x * ds, y * ds] = 1.
                        yield raw_data

    def recurrent_generator(self, advantages, num_mini_batch):
        raise NotImplementedError
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        T, N = self.num_steps, num_envs_per_batch

        for start_ind in range(0, num_processes, num_envs_per_batch):

            obs = []
            rec_states = []
            actions = []
            value_preds = []
            returns = []
            masks = []
            old_action_log_probs = []
            adv_targ = []
            if self.has_extras:
                extras = []

            for offset in range(num_envs_per_batch):

                ind = perm[start_ind + offset]
                obs.append(self.obs[:-1, ind])
                rec_states.append(self.rec_states[0:1, ind])
                actions.append(self.actions[:, ind])
                value_preds.append(self.value_preds[:-1, ind])
                returns.append(self.returns[:-1, ind])
                masks.append(self.masks[:-1, ind])
                old_action_log_probs.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
                if self.has_extras:
                    extras.append(self.extras[:-1, ind])

            # These are all tensors of size (T, N, ...)
            obs = torch.stack(obs, 1)
            actions = torch.stack(actions, 1)
            value_preds = torch.stack(value_preds, 1)
            returns = torch.stack(returns, 1)
            masks = torch.stack(masks, 1)
            old_action_log_probs = torch.stack(old_action_log_probs, 1)
            adv_targ = torch.stack(adv_targ, 1)
            if self.has_extras:
                extras = torch.stack(extras, 1)

            yield {
                'obs': _flatten_helper(T, N, obs),
                'actions': _flatten_helper(T, N, actions),
                'value_preds': _flatten_helper(T, N, value_preds),
                'returns': _flatten_helper(T, N, returns),
                'masks': _flatten_helper(T, N, masks),
                'old_action_log_probs': _flatten_helper(T, N, old_action_log_probs),
                'adv_targ': _flatten_helper(T, N, adv_targ),
                'extras': _flatten_helper(T, N, extras) if self.has_extras else None,
                'rec_states': torch.stack(rec_states, 1).view(N, -1),
            }


class GlobalRolloutStorage(RolloutStorage):

    def __init__(self, num_steps, num_processes, eval_freq, num_repeats, num_robots, obs_shape, action_space,
                 rec_state_size, extras_size):
        super(GlobalRolloutStorage, self).__init__(num_steps, num_processes, eval_freq, num_repeats, num_robots, obs_shape, action_space, rec_state_size)
        self.extras = torch.zeros((num_steps + 1, num_processes * num_robots * num_repeats, extras_size), dtype=torch.long)
        self.has_extras = True
        self.extras_size = extras_size

    def insert(self, obs, rec_states, actions, action_log_probs, value_preds,
               rewards, masks, extras):
        l, h = self.mini_step * self.mini_step_size, (self.mini_step + 1) * self.mini_step_size
        if self.step == 0:
            ll, lh = l-self.mini_step_size, h-self.mini_step_size
            if lh == 0:
                lh = self.mini_step_size * self.num_mini_step
            self.extras[0][l:h].copy_(self.extras[-1][ll:lh])
        self.extras[self.step + 1][l:h].copy_(extras)
        super(GlobalRolloutStorage, self).insert(obs, rec_states, actions,
                                                 action_log_probs, value_preds, rewards, masks)

# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO():

    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        max_batch_size,
        rotation_augmentation,
        value_loss_coef,
        action_loss_coef,
        entropy_coef,
        max_grad_norm=None,
        use_clipped_value_loss=True):

        self.actor_critic = actor_critic
        self.actor_optimizer = actor_critic.actor_optimizer
        self.critic_optimizer = actor_critic.critic_optimizer

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.max_batch_size = max_batch_size
        self.rotation_augmentation = rotation_augmentation

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.action_loss_coef = action_loss_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        rollouts_begin = (rollouts.mini_step_size * rollouts.num_steps) if rollouts.first_use_to_eval else 0
        valid_advantages = advantages[rollouts.masks[:-1].bool()][rollouts_begin:]
        if min(valid_advantages.shape) == 0:
            print('empty samples ... skip !')
            return 0, 0, 0
        print('min/max/mean/med adv: {:.3f}/{:.3f}/{:.3f}/{:.3f}'.format(valid_advantages.min(), valid_advantages.max(), valid_advantages.mean(), valid_advantages.median()))
        advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-5)
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):

            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                        advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                        advantages, self.num_mini_batch, self.max_batch_size, self.rotation_augmentation, ds=self.actor_critic.network.downscaling)

            for sample in data_generator:
                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, action_feature = \
                    self.actor_critic.evaluate_actions(
                        sample['obs'], sample['rec_states'],
                        sample['masks'], sample['actions'],
                        extras=sample['extras']
                    )

                augmentation = sample['augmentation']
                value_preds = values if augmentation else sample['value_preds']
                old_action_log_probs = action_log_probs if augmentation else sample['old_action_log_probs']
                returns = sample['returns']
                adv_targ = sample['adv_targ']

                ratio = torch.exp(action_log_probs - old_action_log_probs)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()
                action_clip = (torch.abs(ratio - 1.0) <= self.clip_param).float().mean().item()
                if torch.isnan(action_loss):
                    print('aloss nan')
                    continue

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds + \
                                        (values - value_preds).clamp(
                                            -self.clip_param, self.clip_param)
                    value_losses = (values - returns).pow(2)
                    value_losses_clipped = (value_pred_clipped
                                            - returns).pow(2)
                    value_loss = .5 * torch.max(value_losses,
                                                value_losses_clipped).mean()
                    value_clip = (torch.abs(values - value_preds) <= self.clip_param).float().mean().item()
                else:
                    value_loss = 0.5 * (returns - values).pow(2).mean()
                    value_clip = 1.

                print('a/v clip: {:.3f}/{:.3f}'.format(action_clip, value_clip))

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                (value_loss * self.value_loss_coef + action_loss * self.action_loss_coef - dist_entropy * self.entropy_coef).backward()
                
                nn.utils.clip_grad_norm_(self.actor_critic.network.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.actor_critic.network.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                if not augmentation:
                    value_loss_epoch += value_loss.item()
                    action_loss_epoch += action_loss.item()
                    dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

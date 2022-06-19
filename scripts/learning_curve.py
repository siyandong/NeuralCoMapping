import matplotlib.pyplot as plt
import argparse
import os
import re
import numpy as np


def main(path):
    timestep = 0
    critic_losses = []
    actor_losses = []
    steps = []
    mean_reward = []
    eps_step = 0
    ppo_sample_eps = 0
    exp_name = 'exp'
    eps = []
    with open(path, 'r') as f:
        for line in f.readlines():
            if len(line) < 5:
                continue
            line = line.rstrip()
            if 'Namespace(' in line:
                res = re.match(r'.*?exp_name=\'(.*?)\'.*?max_episode_length=([0-9]*).*?num_processes=([0-9]*).*?ppo_sample_eps=([0-9]*).*', line)
                exp_name = res.group(1)
                eps_step = int(res.group(2)) * int(res.group(3))
                ppo_sample_eps = int(res.group(4))
                eps = []
                critic_losses = []
                actor_losses = []
                value_losses = []
                action_losses = []
                dist_losses = []
                steps = []
                mean_reward = []
                mean_length = []
                mean_val_length = []
            elif line.find('num timesteps') >= 0:
                timestep = int(re.match(r'.*?num timesteps ([0-9]*),.*', line).group(1))
                if timestep % eps_step == 0 and timestep > 0:
                    eps.append(timestep // eps_step)
                    mean_reward.append(np.nan)
                    mean_length.append(np.nan)
                    mean_val_length.append(np.nan)
                steps.append(timestep)
                critic_losses.append(np.nan)
                actor_losses.append(np.nan)
                value_losses.append(np.nan)
                action_losses.append(np.nan)
                dist_losses.append(np.nan)
            if line.find('Global Loss critic/actor:') >= 0:
                res = re.match(r'.*?Global Loss critic/actor: ([0-9-.]*)/([0-9-.]*).*', line)
                critic_losses[-1] = float(res.group(1))
                actor_losses[-1] = float(res.group(2))
            if line.find('Global Loss value/action/dist/consistency:') >= 0:
                res = re.match(r'.*?Global Loss value/action/dist/consistency: ([0-9-.]*)/([0-9-.]*)/([0-9-.]*)/.*', line)
                value_losses[-1] = float(res.group(1))
                action_losses[-1] = float(res.group(2))
                dist_losses[-1] = float(res.group(3))
            if line.find('Global eps mean/med/min/max eps rew:') >= 0:
                if mean_reward and mean_reward[-1] is np.nan:
                    mean_reward[-1] = float(re.match(r'.*?Global eps mean/med/min/max eps rew: ([0-9-.]*).*', line).group(1))
            if line.find('Global eps mean/med eps len:') >= 0:
                if mean_length and mean_length[-1] is np.nan:
                    mean_length[-1] = int(re.match(r'.*?Global eps mean/med eps len: ([0-9-.]*).*', line).group(1))
            if line.find('Validation eps mean/med eps len:') >= 0:
                if mean_val_length and mean_val_length[-1] is np.nan:
                    mean_val_length[-1] = int(re.match(r'.*?Validation eps mean/med eps len: ([0-9-.]*).*', line).group(1))

    steps = np.asarray(steps)
    g_steps = np.asarray(steps)
    actor_losses = np.asarray(actor_losses)
    critic_losses = np.asarray(critic_losses)
    eps = np.asarray(eps) / ppo_sample_eps
    mean_reward = np.asarray(mean_reward)
    mean_length = np.asarray(mean_length)
    mean_val_length = np.asarray(mean_val_length)
    value_losses = np.asarray(value_losses)
    action_losses = np.asarray(action_losses)
    dist_losses = np.asarray(dist_losses)

    '''fig, ax = plt.subplots(1, 3, figsize=(9, 4))
    ax[0].plot(steps, critic_losses, color='g')
    ax[1].plot(steps, actor_losses, color='r')
    ax[2].plot(eps, mean_reward, color='b')'''
    fig, ax = plt.subplots(1, 6, figsize=(32, 8))
    ax[0].plot(steps, value_losses, color='g')
    ax[0].set_title('value loss')
    ax[0].set_xlabel('timestep')
    ax[1].plot(g_steps, action_losses, color='r')
    ax[1].set_title('action loss')
    ax[1].set_xlabel('timestep')
    ax[2].plot(g_steps, dist_losses, color='b')
    ax[2].set_title('dist loss')
    ax[2].set_xlabel('timestep')
    ax[3].plot(eps, mean_reward, color='orange')
    ax[3].set_title('mean reward')
    ax[3].set_xlabel('episode')
    ax[4].plot(eps, mean_length, color='purple')
    ax[4].set_title('mean length')
    ax[4].set_xlabel('episode')
    ax[5].plot(eps, mean_val_length, color='brown')
    ax[5].set_title('mean val length')
    ax[5].set_xlabel('episode')
    plt.savefig(exp_name + '.png')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--is_dir', action='store_true', default=False)
    args = parser.parse_args()
    if args.is_dir:
        for file in os.listdir(args.path):
            if file[-4:] == '.log':
                main(os.path.join(args.path, file))
    else:
        main(args.path)
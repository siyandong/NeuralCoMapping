import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import gym
import logging
from arguments import get_args
from env.gibson_api import construct_envs
from env.gibson_api.utils.shared_memory import SharedNumpyPool
from utils.storage import GlobalRolloutStorage
from utils.map_manager import MapManager
from model import RL_Policy

import algo

import sys
import matplotlib
import random

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists("{}/images/".format(dump_dir)):
        os.makedirs("{}/images/".format(dump_dir))

    fh = logging.FileHandler(log_dir + 'basic.log')
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    print("Dumping at {}".format(log_dir))
    logging.info(args)
    summary_writer = SummaryWriter(log_dir)
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    np.seterr(divide='ignore', invalid='ignore')

    # Logging and loss variables
    num_scenes = args.num_processes
    num_robots = args.num_robots
    num_batches = num_scenes if args.centralized else num_scenes * num_robots
    num_episodes = int(args.num_episodes)
    device = args.device = torch.device("cuda:{}".format(0) if args.cuda else "cpu")
    torch.cuda.set_device("cuda:{}".format(0))

    global_masks = torch.ones(num_scenes).float().to(device)
    one_masks = torch.ones(num_scenes * num_robots).float().to(device)

    best_g_reward = -np.inf

    if args.eval:
        num_global_steps = args.max_episode_length // args.num_local_steps
        explored_area_log = np.zeros((num_scenes, num_episodes, num_global_steps))
        explored_ratio_log = np.zeros((num_scenes, num_episodes, num_global_steps))
    close_episode_len = np.zeros((num_scenes, num_episodes))
    bump_cnt = np.zeros((num_scenes, num_episodes), dtype=np.int32)
    last_bump = np.zeros(num_scenes, dtype=np.int32)
    cont_bump_cnt = np.zeros((num_scenes, num_episodes), dtype=np.int32)

    g_episode_rewards = deque(maxlen=100)
    g_value_losses = deque(maxlen=100)
    g_action_losses = deque(maxlen=100)
    g_dist_entropies = deque(maxlen=100)
    per_step_g_rewards = deque(maxlen=100)
    g_process_rewards = np.zeros((num_scenes))

    g_episode_length = deque(maxlen=100)
    g_val_episode_length = deque(maxlen=100)

    # Starting environments
    torch.set_num_threads(1)

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.unit_size_cm
    global_map_w, global_map_h = map_size, map_size
    local_map_w, local_map_h = int(global_map_w / args.global_downscaling), int(global_map_h / args.global_downscaling)

    snp = SharedNumpyPool(args.snp_location)
    for _ in snp.allocate_lazy():
        sensor_pose = snp.allocate('sensor_pose', (num_scenes * num_robots, 3))
        pose_err = snp.allocate('pose_err', (num_scenes * num_robots, 3))
        origin_pose = snp.allocate('origin_pose', (num_scenes * num_robots, 3))
        last_stg = snp.allocate('last_stg', (num_scenes * num_robots, 2))
        obstacle = snp.allocate('obstacle', (num_scenes, map_size, map_size))
        frontier = snp.allocate('frontier', (num_scenes, map_size, map_size))
        explored = snp.allocate('explored', (num_scenes, map_size, map_size))
        explorable = snp.allocate('explorable', (num_scenes, map_size, map_size))
        obs = snp.allocate('obs', (num_scenes * num_robots, 4, args.frame_height, args.frame_width), np.uint8)
        exp_reward = snp.allocate('exp_reward', (num_scenes,))
        exp_ratio = snp.allocate('exp_ratio', (num_scenes,))
        g_reward = snp.allocate('g_reward', (num_scenes,))
        l_reward = snp.allocate('l_reward', (num_scenes * num_robots,))
        bump = snp.allocate('bump', (num_scenes * num_robots,), np.bool)
    logging.info("SNP allocated {:.1f} MB".format(snp.max_size / 1024 ** 2))

    envs = construct_envs(args, snp.dump())
    for _ in range(args.restore_eps):
        envs.reset()
    envs.reset()

    # Initialize map variables

    manager = MapManager(args, global_map_w, global_map_h, local_map_w, local_map_h, device)

    torch.set_grad_enabled(False)

    manager.init_map_and_pose(origin_pose)

    # Global policy space
    if args.centralized:
        g_observation_space = gym.spaces.Box(0, 1, (8 + num_robots, global_map_w // 4, global_map_h // 4), dtype='uint8')
        g_action_space = gym.spaces.Box(0, (global_map_w // 4) * (global_map_h // 4) - 1, (num_robots,), dtype='int32')
        g_history = torch.zeros((num_scenes, global_map_w // 4, global_map_h // 4))
    else:
        # for ans
        g_observation_space = gym.spaces.Box(0, 1, (9, local_map_w // 2, local_map_h // 2), dtype='uint8')
        g_action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        g_history = None

    # Global policy
    g_policy = RL_Policy(g_observation_space.shape, g_action_space,
                         model_type='gnn' if args.centralized else 'ans',
                         base_kwargs={'num_gnn_layer': args.num_gnn_layer,
                                      'use_history': args.use_history,
                                      'ablation': args.ablation},
                         lr=(args.global_lr, args.critic_lr_coef), eps=args.eps).to(device)
    # assert args.centralized or g_policy.downscaling == 2
    g_agent = algo.PPO(g_policy, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                       args.max_batch_size, args.rotation_augmentation, args.value_loss_coef, args.action_loss_coef,
                       args.entropy_coef, max_grad_norm=args.max_grad_norm,
                       use_clipped_value_loss = args.use_clipped_value_loss)

    # Storage
    g_rollouts = GlobalRolloutStorage(args.num_global_steps, num_scenes,
                                      args.eval_eps_freq, args.ppo_sample_eps,
                                      1 if args.centralized else num_robots,
                                      g_observation_space.shape,
                                      g_action_space, g_policy.rec_state_size,
                                      num_robots * 6 if args.centralized else 7).to(device)             

    if args.load_global != "0":
        print("Loading global {}".format(args.load_global))
        g_policy.load(args.load_global, device)
    elif args.load_global_critic != "0":
        g_policy.load_critic(args.load_global_critic, device)

    if not args.train_global:
        g_policy.eval()

    manager.update_local(sensor_pose)
    manager.update_global(obstacle, frontier, explored, explorable)
    global_input, global_position = manager.get_global_input(g_history)
    planner_pose_inputs = manager.get_planner_input()

    l = g_rollouts.mini_step * g_rollouts.mini_step_size
    h = (g_rollouts.mini_step + 1) * g_rollouts.mini_step_size
    g_rollouts.obs[0][l:h].copy_(global_input.view(num_batches, *g_observation_space.shape))
    if args.centralized:
        g_rollouts.extras[0][l:h].copy_(global_position.view(num_batches, -1) // 4)
    else:
        global_position.view(num_batches, -1)[:, :-1] //= 2
        g_rollouts.extras[0][l:h].copy_(global_position.view(num_batches, -1))
    ll, lh = l-g_rollouts.mini_step_size, h-g_rollouts.mini_step_size
    if lh == 0:
        lh = g_rollouts.mini_step_size * g_rollouts.num_mini_step
    g_rollouts.obs[-1][ll:lh].copy_(g_rollouts.obs[0][l:h])
    g_rollouts.rec_states[-1][ll:lh].copy_(g_rollouts.rec_states[0][l:h])
    g_rollouts.extras[-1][ll:lh].copy_(g_rollouts.extras[0][l:h])

    # Run Global Policy (global_goals = Long-Term Goal)
    g_value, g_action, g_action_log_prob, g_rec_states, g_action_map = \
        g_policy.act(
            g_rollouts.obs[0][l:h],
            g_rollouts.rec_states[0][l:h],
            g_rollouts.masks[0][l:h],
            extras=g_rollouts.extras[0][l:h],
            deterministic=False
        )

    
    to_draw_heatmap = 1
    if args.centralized:
        cpu_actions = g_action.view(num_scenes, num_robots).cpu().numpy()
        ds = 4 * g_policy.downscaling
        global_goals = []
        heatmap = global_input[:, 1, :, :].detach().clone() if to_draw_heatmap else ([None] * num_scenes)
        # heatmap = np.zeros(global_input[:, 1, :, :].shape) if to_draw_heatmap else ([None] * num_scenes)
        global_goals = []
        global_position_npy = global_position.view(num_scenes, num_robots, -1)[:, :, [2, 4]].cpu().numpy()
        for i in range(num_scenes):
            frontier_idx = torch.nonzero(global_input[i, 1, :, :]).cpu().numpy()
            for a in range(num_robots):
                g_history[(i, *frontier_idx[cpu_actions[i, a]])] = 1
            global_goals.append([[*(frontier_idx[cpu_actions[i, a]] * ds + ds // 2 - global_position_npy[i, a]), ds // 2] for a in range(num_robots)])
            if to_draw_heatmap:
                heatmap[i, heatmap[i] > 0] = g_action_map[i].softmax(dim=1)[0]
        if to_draw_heatmap:
            heatmap = torch.flip(heatmap, [1]).cpu().numpy()
    else:
        '''ds = 2 * g_policy.downscaling
        global_goals = [[[(cpu_actions[e, a] // (local_map_w // ds)) * ds + ds//2, (cpu_actions[e, a] % (local_map_w // ds)) * ds + ds//2, ds//2] for a in range(num_robots)] for e in range(num_scenes)]
        heatmap = torch.flip(g_action_map.view(num_scenes, num_robots, local_map_w // ds, local_map_h // ds)[:, 0, :, :], [1]).detach().cpu().numpy()'''
        # for ans
        cpu_actions = (nn.Sigmoid()(g_action * 2).view(num_scenes, num_robots, 2).cpu().numpy() - 0.5) / 2 + 0.5
        ds = 2 * g_policy.downscaling
        global_goals = [[[(int(cpu_actions[e, a, 0] * local_map_w) // ds) * ds + ds//2, (int(cpu_actions[e, a, 1] * local_map_h) // ds) * ds + ds//2, ds//2] for a in range(num_robots)] for e in range(num_scenes)]
        heatmap = np.zeros((num_scenes, 1, 1))


    # Compute planner inputs
    planner_inputs = [[{} for a in range(num_robots)] for e in range(num_scenes)]
    for e in range(num_scenes):
        for a, p_input in enumerate(planner_inputs[e]):
            p_input['goal'] = global_goals[e][a]
            p_input['pose_pred'] = planner_pose_inputs[e, a]

    # Output stores local goals as well as the the ground-truth action
    
    output = envs.get_short_term_goal(planner_inputs, heatmap)

    l_action = output.long().view(num_scenes, num_robots)

    start = time.time()

    total_num_steps = -1
    g_reward_tensor = 0
    l_reward_tensor = 0

    torch.set_grad_enabled(False)

    global user_action

    for idx_episode in range(args.restore_eps, num_episodes):
        eval_flag = (args.eval_eps_freq and idx_episode % args.eval_eps_freq == 0)
        for step in range(args.max_episode_length):
            total_num_steps += 1

            g_step = (step // args.num_local_steps) % args.num_global_steps
            g_step_eval = step // args.num_local_steps + 1
            l_step = step % args.num_local_steps

            # ------------------------------------------------------------------
            
            # Env step
            done = envs.step(l_action)

            local_masks = torch.FloatTensor([0 if x else 1
                                         for x in done]).to(device)
            global_masks *= local_masks
            
            for e in range(num_scenes):
                if (done[e] or step == args.max_episode_length - 1) and close_episode_len[e, idx_episode] == 0:
                    close_episode_len[e, idx_episode] = step
            # ------------------------------------------------------------------


            # ------------------------------------------------------------------
            # Reinitialize variables when episode ends
            if step == args.max_episode_length - 1:  # Last episode step
                if eval_flag:
                    g_val_episode_length.append(close_episode_len[:, idx_episode].mean())
                else:
                    g_episode_length.append(close_episode_len[:, idx_episode].mean())
                l_action *= 0
                last_bump[:] = -1
                manager.init_map_and_pose(origin_pose)
                # g_policy.reset()


            bump_per_scene = bump.reshape(num_scenes, num_robots).astype(np.int32).sum(1)
            bump_cnt[:, idx_episode] += bump_per_scene
            for i in range(num_scenes):
                if last_bump[i] >= 0:
                    cont_bump_cnt[i, idx_episode] = np.maximum(cont_bump_cnt[i, idx_episode], step - last_bump[i])
                    if bump_per_scene[i] == 0:
                        last_bump[i] = -1
                elif bump_per_scene[i] > 0:
                    last_bump[i] = step


            manager.update_local(sensor_pose)

            if step == args.max_episode_length - 1:
                if g_history is not None:
                    g_history.fill_(0)
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Global Policy
            if l_step == args.num_local_steps - 1:
                # For every global step, update the full and local maps
                
                manager.update_global(obstacle, frontier, explored, explorable)                
                global_input, global_position = manager.get_global_input(g_history)
                
                # Get exploration g_reward and metrics

                g_reward_tensor = torch.from_numpy(g_reward).float().to(device)
                if not args.centralized:
                    g_reward_tensor = torch.repeat_interleave(g_reward_tensor.unsqueeze(1), repeats=num_robots, dim=1).view(-1)

                g_process_rewards += exp_reward
                per_step_g_rewards.append(np.mean(exp_reward))

                if step == args.max_episode_length - 1:
                    tr = [tr for tr in g_process_rewards if tr > 0]
                    g_episode_rewards.append(np.mean(tr))
                    g_process_rewards *= 0.
                else:
                    pass
                    # g_process_rewards *= global_masks.cpu().numpy()

                if args.eval:
                    for e in range(num_scenes):
                        explored_area_log[e, idx_episode, g_step_eval - 1] = explored_area_log[e, idx_episode, g_step_eval - 2] + exp_reward[e]
                        explored_ratio_log[e, idx_episode, g_step_eval - 1] = explored_ratio_log[e, idx_episode, g_step_eval - 2] + exp_ratio[e]

                l = g_rollouts.mini_step * g_rollouts.mini_step_size
                h = (g_rollouts.mini_step + 1) * g_rollouts.mini_step_size
                # Add samples to global policy storage

                if not args.centralized:
                    global_position.view(num_batches, -1)[:, -1] *= 4
                g_rollouts.insert(
                    global_input.view(num_batches, *g_observation_space.shape),
                    g_rec_states, g_action, g_action_log_prob, g_value, g_reward_tensor,
                    global_masks if args.centralized else torch.repeat_interleave(global_masks.unsqueeze(1), repeats=num_robots, dim=1).view(-1),
                    global_position.view(num_batches, -1) // 4
                )

                # Sample long-term goal from global policy
                g_value, g_action, g_action_log_prob, g_rec_states, g_action_map = \
                    g_policy.act(
                        g_rollouts.obs[g_step + 1][l:h],
                        g_rollouts.rec_states[g_step + 1][l:h],
                        g_rollouts.masks[g_step + 1][l:h],
                        extras=g_rollouts.extras[g_step + 1][l:h],
                        deterministic=False
                    )


                to_draw_heatmap = args.print_images or idx_episode % 10 == 0 or args.eval
                if args.centralized:
                    cpu_actions = g_action.view(num_scenes, num_robots).cpu().numpy()
                    global_goals = []
                    heatmap = global_input[:, 1, :, :].detach().clone() if to_draw_heatmap else ([None] * num_scenes)
                    # heatmap = np.zeros(global_input[:, 1, :, :].shape) if to_draw_heatmap else ([None] * num_scenes)
                    global_goals = []
                    global_position_npy = global_position.view(num_scenes, num_robots, -1)[:, :, [2, 4]].cpu().numpy()
                    for i in range(num_scenes):
                        frontier_idx = torch.nonzero(global_input[i, 1, :, :]).cpu().numpy()
                        for a in range(num_robots):
                            g_history[(i, *frontier_idx[cpu_actions[i, a]])] = 1
                        global_goals.append([[*(frontier_idx[cpu_actions[i, a]] * ds + ds // 2 - global_position_npy[i, a]), ds // 2] for a in range(num_robots)])
                        if to_draw_heatmap:
                            heatmap[i, heatmap[i] > 0] = g_action_map[i].softmax(dim=1)[0]
                    if to_draw_heatmap:
                        heatmap = torch.flip(heatmap, [1]).cpu().numpy()
                else:
                    # for ans
                    cpu_actions = (nn.Sigmoid()(g_action * 2).view(num_scenes, num_robots, 2).cpu().numpy() - 0.5) / 2 + 0.5
                    ds = 2 * g_policy.downscaling
                    global_goals = [[[(int(cpu_actions[e, a, 0] * local_map_w) // ds) * ds + ds//2, (int(cpu_actions[e, a, 1] * local_map_h) // ds) * ds + ds//2, ds//2] for a in range(num_robots)] for e in range(num_scenes)]
                    heatmap = np.zeros((num_scenes, 1, 1)) if to_draw_heatmap else ([None] * num_scenes)
                
                g_reward_tensor = 0
                global_masks = torch.ones(num_scenes).float().to(device)
            elif not args.print_images:
                heatmap = ([None] * num_scenes)
            # ------------------------------------------------------------------

            
            # ------------------------------------------------------------------
            # Get short term goal
            planner_pose_inputs = manager.get_planner_input()
            planner_inputs = [[{} for a in range(num_robots)] for e in range(num_scenes)]
            for e in range(num_scenes):
                for a, p_input in enumerate(planner_inputs[e]):
                    p_input['goal'] = global_goals[e][a]
                    p_input['pose_pred'] = planner_pose_inputs[e, a]

            
            output = envs.get_short_term_goal(planner_inputs, heatmap)
            l_action = output.long().view(num_scenes, num_robots)
            # ------------------------------------------------------------------

            

            # ------------------------------------------------------------------
            ### TRAINING
            torch.set_grad_enabled(True)


            # Train Global Policy

            if (g_step % args.num_global_steps == args.num_global_steps - 1 and l_step == args.num_local_steps - 1) and g_rollouts.mini_step == 0:
                if args.train_global and not eval_flag:
                    g_next_value = g_policy.get_value(
                        g_rollouts.obs[-1],
                        g_rollouts.rec_states[-1],
                        g_rollouts.masks[-1],
                        extras=g_rollouts.extras[-1]
                    )[0].detach()

                    g_rollouts.compute_returns(g_next_value, args.use_gae,
                                               args.gamma, args.tau)
                    g_value_loss, g_action_loss, g_dist_entropy = \
                        g_agent.update(g_rollouts)
                    if g_value_loss > 0:
                        g_value_losses.append(g_value_loss)
                        g_action_losses.append(g_action_loss)
                        g_dist_entropies.append(g_dist_entropy)
                g_rollouts.after_update()

            # Finish Training
            torch.set_grad_enabled(False)
            # ------------------------------------------------------------------


            # ------------------------------------------------------------------
            # Logging
            if total_num_steps % args.log_interval == 0:
                end = time.time()
                time_elapsed = time.gmtime(end - start)

                log = " ".join([
                    "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                    "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                    "num timesteps {},".format(total_num_steps *
                                               num_scenes),
                    "FPS {},".format(int(total_num_steps * num_scenes \
                                         / (end - start)))
                ])

                log += "\n\tRewards:"

                if len(g_episode_rewards) > 0:
                    log += " ".join([
                        " Global step mean/med rew:",
                        "{:.4f}/{:.4f},".format(
                            np.mean(per_step_g_rewards),
                            np.median(per_step_g_rewards)),
                        " Global eps mean/med/min/max eps rew:",
                        "{:.3f}/{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_episode_rewards),
                            np.median(g_episode_rewards),
                            np.min(g_episode_rewards),
                            np.max(g_episode_rewards))
                    ])

                if len(g_episode_length) > 0:
                    log += " ".join([
                        " Global eps mean/med eps len:",
                        "{:.0f}/{:.0f},".format(
                            np.mean(g_episode_length),
                            np.median(g_episode_length))
                    ])
                if len(g_val_episode_length) > 0:
                    log += " ".join([
                        " Validation eps mean/med eps len:",
                        "{:.0f}/{:.0f},".format(
                            np.mean(g_val_episode_length),
                            np.median(g_val_episode_length))
                    ])

                log += "\n\tLosses:"

                if args.train_global and len(g_value_losses) > 0:
                    log += " ".join([
                        " Global Loss value/action/dist:",
                        "{:.5f}/{:.5f}/{:.5f},".format(
                            np.mean(g_value_losses),
                            np.mean(g_action_losses),
                            np.mean(g_dist_entropies))
                    ])

                logging.info(log)
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Save best models
            if (total_num_steps * num_scenes) % args.save_interval < \
                    num_scenes:

                # Save Global Policy Model
                if args.train_global and len(g_episode_length) >= 10 and \
                        (np.mean(g_episode_length) >= best_g_reward) \
                        and not args.eval:
                    g_policy.save(os.path.join(log_dir, "model_best.global"))
                    best_g_reward = np.mean(g_episode_length)

            # Save periodic models
            if (total_num_steps * num_scenes) % args.save_periodic < \
                    num_scenes:
                step = total_num_steps * num_scenes
                if args.train_global:
                    g_policy.save(os.path.join(dump_dir, "periodic_{}.global".format(step)))

            # ------------------------------------------------------------------
        if not eval_flag and len(g_episode_rewards) > 0:
            summary_writer.add_scalar('train/area', g_episode_rewards[-1], global_step=idx_episode)
        if not eval_flag and len(g_episode_length) > 0:
            for e in range(num_scenes):
                summary_writer.add_scalar(f'train/length{e}', close_episode_len[e, idx_episode], global_step=idx_episode)
        if eval_flag and len(g_val_episode_length) > 0:
            for e in range(num_scenes):
                summary_writer.add_scalar(f'test/length{e}', close_episode_len[e, idx_episode], global_step=idx_episode)
        if args.train_global and len(g_value_losses) > 0:
            summary_writer.add_scalar('loss/actor', g_action_losses[-1], global_step=idx_episode)
            summary_writer.add_scalar('loss/critic', g_value_losses[-1], global_step=idx_episode)
            summary_writer.add_scalar('loss/entropy', g_dist_entropies[-1], global_step=idx_episode)

    # Print and save model performance numbers during evaluation
    if args.eval:
        logfile = open("{}/explored_area.txt".format(dump_dir), "w+")
        for e in range(num_scenes):
            for i in range(explored_area_log[e].shape[0]):
                logfile.write(str(explored_area_log[e, i]) + "\n")
                logfile.flush()

        logfile.close()

        logfile = open("{}/explored_ratio.txt".format(dump_dir), "w+")
        for e in range(num_scenes):
            for i in range(explored_ratio_log[e].shape[0]):
                logfile.write(str(explored_ratio_log[e, i]) + "\n")
                logfile.flush()

        logfile.close()

        logfile = open("{}/close_episode_len.txt".format(dump_dir), "w+")
        for e in range(num_scenes):
            for i in range(close_episode_len[e].shape[0]):
                logfile.write(str(close_episode_len[e, i]) + "\n")
                logfile.flush()

        logfile.close()

        logfile = open("{}/bump_cnt.txt".format(dump_dir), "w+")
        for e in range(num_scenes):
            for i in range(bump_cnt[e].shape[0]):
                logfile.write(str(bump_cnt[e, i]) + "\n")
                logfile.flush()

        logfile.close()

        logfile = open("{}/cont_bump_cnt.txt".format(dump_dir), "w+")
        for e in range(num_scenes):
            for i in range(cont_bump_cnt[e].shape[0]):
                logfile.write(str(cont_bump_cnt[e, i]) + "\n")
                logfile.flush()

        logfile.close()

        log = "\nFinal Exp Area: \n"
        for i in range(explored_area_log.shape[2]):
            log += "{:.5f}, ".format(
                np.mean(explored_area_log[:, :, i]))

        log += "\nFinal Exp Ratio: \n"
        for i in range(explored_ratio_log.shape[2]):
            log += "{:.5f}, ".format(
                np.mean(explored_ratio_log[:, :, i]))

        log += "\nFinal Close Lengths: \n"
        for e in range(num_scenes):
            log += "{:.2f}, ".format(np.mean(close_episode_len[e, :]))

        log += "\nFinal Bump Counts: \n"
        for e in range(num_scenes):
            log += "{}, ".format(np.mean(bump_cnt[e, :]))

        log += "\nFinal Continuous Bump Counts: \n"
        for e in range(num_scenes):
            log += "{}, ".format(np.mean(cont_bump_cnt[e, :]))
        logging.info(log)

    summary_writer.close()


if __name__ == "__main__":

    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        
    main()

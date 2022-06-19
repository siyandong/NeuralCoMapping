import math
import os
import pickle
import sys
import yaml

import gym
import matplotlib
import numpy as np
import quaternion
import skimage.morphology
import torch
from math import ceil
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from collections import deque
import random

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

from .ma_env_base import MAGibsonEnv

from env.utils.map_builder import MapBuilder
from env.utils.fmm_planner import FMMPlanner
import env.utils.rotation_utils as ru

from .utils import pose as pu
from .utils import visualizations as vu
from .utils.supervision import GibsonMaps
from .utils.shared_memory import SharedNumpyPool

from model import get_grid
from gibson2.utils.mesh_util import quat2rotmat

import cv2


def _preprocess_depth(depth):
    depth = depth[:, :, :, 0] * 0.1

    for a in range(depth.shape[0]):
        if depth[a].min() <= 0.:
            broken = np.pad(depth[a] == 0., ((0, 0), (1, 1)), 'constant').astype(np.int8)
            broken_tag = broken[:, 1:] - broken[:, :-1]
            broken_begin = np.where(broken_tag == 1)
            broken_end = np.where(broken_tag == -1)
            assert broken_begin[0].shape[0] == broken_end[0].shape[0]
            for bx, by, ex, ey in zip(*broken_begin, *broken_end):
                assert bx == ex
                d1 = (depth[a, bx-1, by-1], 1) if by > 0 else (0, 0)
                d2 = (depth[a, ex, ey], 1) if ey < depth.shape[2] else (0, 0)
                depth[a, bx, by:ey] = (d1[0] + d2[0]) / max(1, d1[1] + d2[1])

    depth[depth <= 0] = np.NaN
    depth[depth > 0.99] = np.NaN
    return depth * 1000.



# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging


class GibsonLogger(logging.Logger):
    def __init__(
        self,
        name,
        level,
        filename=None,
        filemode="a",
        stream=None,
        format=None,
        dateformat=None,
        style="%",
    ):
        super().__init__(name, level)
        if filename is not None:
            handler = logging.FileHandler(filename, filemode)
        else:
            handler = logging.StreamHandler(stream)
        self._formatter = logging.Formatter(format, dateformat, style)
        handler.setFormatter(self._formatter)
        super().addHandler(handler)

    def add_filehandler(self, log_filename):
        filehandler = logging.FileHandler(log_filename)
        filehandler.setFormatter(self._formatter)
        self.addHandler(filehandler)


logger = GibsonLogger(
    name="gibson", level=logging.ERROR, format="%(asctime)-15s %(message)s"
)

def imgize(mp):
    mp_min = mp.min()
    mp_max = mp.max()
    mp = (mp - mp_min) / (mp_max - mp_min + 1e-10)
    return (mp * 255).astype(np.uint8)


class Exploration_Env(MAGibsonEnv):

    def __init__(self, args, config_file, scene_ids, device_idx, rank, snp_data):
        np.random.seed(args.seed + rank * 1000)
        random.seed(args.seed + rank * 1000)
        np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
        np.seterr(divide='ignore', invalid='ignore')
        if args.vis_type in [1, 3]:
            vis_iter = [[], [1, 1], [2, 1], [3, 1], [2, 2], [3, 2], [3, 2], [3, 3], [3, 3], [3, 3]][args.num_robots]
            self.figure = plt.figure(figsize=(6*16/9, 6), facecolor="whitesmoke", num="Thread {}".format(rank))
            gs = gridspec.GridSpec(1, vis_iter[1] + 1, width_ratios=[1]*vis_iter[1]+[vis_iter[0]])
            self.ax = []
            for i in range(args.num_robots):
                j = (i % vis_iter[1]) + i // vis_iter[1] * (vis_iter[0] + vis_iter[1])
                self.ax.append(plt.subplot(vis_iter[0], vis_iter[0] + vis_iter[1], j + 1))
            self.ax.append(plt.subplot(gs[-1]))
            assert args.num_robots < 10

        self.args = args
        self.num_actions = 4
        self.dt = 10
        self.rank = rank
        self.snp = SharedNumpyPool(**snp_data)

        with open(config_file) as f:
            config_file = yaml.load(f, Loader=yaml.FullLoader)
        config_file['reset_orientation'] = args.reset_orientation
        config_file['output'] = ['pc', 'bump']
        if args.print_images:
            config_file['output'].append('rgb')
        config_file['depth_noise_rate'] = args.depth_noise_rate
        config_file['robot_scale'] = args.robot_scale
        config_file['reset_floor'] = args.reset_floor
        config_file['initial_pos_z_offset'] = args.z_offset
        config_file['num_robots'] = args.num_robots
        config_file['reset_min_dist'] = args.obstacle_boundary * 2 / args.unit_size_cm
        config_file['reset_max_dist'] = args.obstacle_boundary * 4 / args.unit_size_cm
        config_file['image_width'] = args.env_frame_width
        config_file['image_height'] = args.env_frame_height
        config_file['vertical_fov'] = np.rad2deg(math.atan(math.tan(np.deg2rad(args.hfov * 0.5)) * args.env_frame_height / args.env_frame_width) * 2.)
        config_file['max_step'] = args.max_episode_length
        config_file['max_collisions_allowed'] = args.max_episode_length
        config_file['texture_randomization_freq'] = args.texture_randomization_freq if args.texture_randomization_freq > 0.0 else None
        config_file['object_randomization_freq'] = args.object_randomization_freq if args.object_randomization_freq > 0.0 else None

        super().__init__(config_file, scene_ids=scene_ids, device_idx=device_idx)

        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.observation_space = gym.spaces.Box(0, 255,
                                                (4, args.frame_height,
                                                    args.frame_width),
                                                dtype='uint8')

        self.mapper = self.build_mapper()
        self.res = transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize((args.frame_height, args.frame_width),
                                      interpolation = Image.NEAREST)])
        self.scene_name = None
        self.maps_dict = {}
        self.collision_tolerance = 0

    def reset(self):
        args = self.args
        self.timestep = 0
        self.straight = [0] * self.num_robots
        self.rotate = [0] * self.num_robots
        self.trajectory_states = []
        self.bump = [False] * self.num_robots
        self.cont_bump = [0] * self.num_robots
        self.collision_dir = [0] * self.num_robots # 0n1f2b3l4r
        self.escape = [0] * self.num_robots
        self.close = False
        self.cont_stuck = 0
        self.collision_tolerance = args.max_collisions_allowed
        self.baseline_goal = [None] * self.num_robots
        self.last_stg = [(-1., -1.)] * self.num_robots
        self.last_stg_dist = [0.] * self.num_robots
        self.last_stg_ori = [0.] * self.num_robots
        self.l_reward = [0.] * self.num_robots
        self.heuristic_type = 0

        # Get Ground Truth Map
        self.explorable_map = None
        while self.explorable_map is None:
            obs, floor_num = super().reset()
            full_map_size = args.map_size_cm//args.unit_size_cm
            self.explorable_map = self._get_gt_map(full_map_size, floor_num)
        self.prev_explored_area = 0.

        if args.print_images:
            self.obs = (obs['rgb'] * 255.).astype(np.uint8)
        else:
            self.obs = np.zeros((self.num_robots, args.env_frame_height, args.env_frame_width, 3), dtype=np.uint8)
        depth = _preprocess_depth(-obs['pc'][:, :, :, 2:3])

        # Initialize map and pose
        # In the iGibson environment, the unit size is 1 m, and the unit size in our world map is 1 cm. So we have to divide the location by 100 when transfering to the iGibson environment.
        self.map_size_cm = args.map_size_cm
        
        ### The initial position is in the middle of the map.
        self.last_sim_location = [self.get_sim_location(i) for i in range(self.num_robots)]
        # origin = self.last_sim_location[0][0:2]
        origin = (self.map_obj.max + self.map_obj.origin) / -200
        if args.eval:
            self.curr_loc = [[self.map_size_cm/100.0/2.0, self.map_size_cm/100.0/2.0, np.rad2deg(self.last_sim_location[0][2])] for _ in range(self.num_robots)]
        else:
            self.curr_loc = [[self.map_size_cm/100.0/2.0, self.map_size_cm/100.0/2.0, 0] for i in range(self.num_robots)]
        origin_loc = [origin[0], origin[1], self.last_sim_location[0][2]]
        for i in range(self.num_robots):
            ### The obtained simulation location still follows the world coodinate system in the iGibson environment
            dx, dy, do = pu.get_rel_pose_change(self.last_sim_location[i], origin_loc)
            self.curr_loc[i] = pu.get_new_pose(self.curr_loc[i], (dx, dy, do))
        self.curr_loc_gt = np.copy(self.curr_loc)
        self.last_loc_gt = np.copy(self.curr_loc_gt)
        self.last_loc = np.copy(self.curr_loc)

        # Set info
        self.info = dict(
            [
                (k,
                self.snp.get(k, (self.rank*self.num_robots, (self.rank+1)*self.num_robots))
                )
            for k in ['sensor_pose', 'pose_err', 'origin_pose', 'last_stg', 'l_reward', 'bump']]
        )
        self.info['sensor_pose'][:] = 0.
        self.info['pose_err'][:] = 0.
        self.info['origin_pose'][:] = np.asarray(self.curr_loc)
        self.info['last_stg'][:] = np.asarray(self.last_stg)

        # Convert pose to cm and degrees for mapper
        mapper_gt_pose = [(self.curr_loc_gt[i][0]*100.0,
                           self.curr_loc_gt[i][1]*100.0,
                           np.deg2rad(self.curr_loc_gt[i][2]),
                           self.get_pitch(i))
                           for i in range(self.num_robots)]

        self.mapper.reset(mapper_gt_pose)
        self.mapper.update_map(depth, mapper_gt_pose)
        if args.print_images and args.vis_type == 2:
            self.obs = depth
        elif args.depth_obs:
            self.obs = self.mapper.depth
        # Update ground_truth map and explored area
        self.obstacle_map, self.explored_map = self.mapper.get()
        self.prev_frontier = np.zeros(self.obstacle_map.shape)
        self.curr_obstacle_map, self.curr_frontier = self.frontier()

        # Initialize variables
        self.scene_name = self.config['scene_id']
        self.visited_gt = np.zeros((self.num_robots, *self.obstacle_map.shape))
        self.local_map = np.zeros(self.obstacle_map.shape)
        self.pos_map = np.zeros((self.num_robots, *self.obstacle_map.shape))
        self.pos_dilation_mask = np.ones((self.num_robots, *self.obstacle_map.shape))
        self.r_threshold = 5
        if args.baseline != 'none':
            raise NotImplementedError
        self.long_term_planner = {
            'none': None
        }[args.baseline]


        for k in ['obstacle', 'frontier', 'explored', 'explorable']:
            self.info[k] = self.snp.get(k, (self.rank, self.rank+1), True)
        for k in ['exp_reward', 'exp_ratio', 'g_reward']:
            self.info[k] = self.snp.get(k, (self.rank, self.rank+1))
        self.info['obstacle'][:] = self.curr_obstacle_map
        self.info['frontier'][:] = self.curr_frontier
        self.info['explored'][:] = self.explored_map
        self.info['explorable'][:] = self.explorable_map

    def step(self, actions):

        args = self.args
        self.timestep += 1
        final_reward = 0

        if self.timestep >= args.max_episode_length:
            
            if self.timestep%args.num_local_steps==0:
                area, ratio = self.get_global_reward()
                g_reward = 0
            else:
                raise RuntimeError("max_episode_length % num_local_steps != 0")
                area, ratio = 0.0, 0.0
                g_reward = 0
            if not self.close:
                l_reward = np.asarray([self.l_reward[i] for i in range(self.num_robots)])
                self.reset()
                self.info['exp_reward'][:] = area
                self.info['exp_ratio'][:] = ratio
                self.info['bump'][:] = False
                self.info['g_reward'][:] = (area - args.reward_bias) * args.reward_scale + g_reward
                self.info['l_reward'][:] = l_reward
            else:
                self.reset()
                self.info['exp_reward'][:] = 0.
                self.info['exp_ratio'][:] = 0.
                self.info['bump'][:] = False
                self.info['g_reward'][:] = final_reward
                self.info['l_reward'][:] = 0.
            return self.close

        if self.close:
            return True

        self.last_loc = np.copy(self.curr_loc)
        self.last_loc_gt = np.copy(self.curr_loc_gt)
        
        actions_real = []
        self.info['last_stg'][:] = np.asarray(self.last_stg)
        for i in range(self.num_robots):
            action = actions[i]
            # Action remapping
            if action == 3: # Backward
                action_real = [-.6, -.6]
                self.straight[i] = (self.straight[i] - 1 if self.straight[i] < 0 else -1)
                self.rotate[i] = 0
            elif action == 2: # Forward
                action_real = [.6, .6]
                self.straight[i] = (self.straight[i] + 1 if self.straight[i] > 0 else 1)
                self.rotate[i] = 0
            elif action == 1: # Right
                action_real = [0.25, -0.25]
                self.straight[i] = 0
                self.rotate[i] = (self.rotate[i] - 1 if self.rotate[i] < 0 else -1)
            elif action == 0: # Left
                action_real = [-0.25, 0.25]
                self.straight[i] = 0
                self.rotate[i] = (self.rotate[i] + 1 if self.rotate[i] > 0 else 1)
            else: # Nothing
                action_real = [0., 0.]
                self.straight[i] = 0
                self.rotate[i] = 0
            if self.collision_dir[i] > 0:
                self.straight[i] = 0
                self.rotate[i] = 0
            if args.noisy_actions:
                noise_percent = 0.025
                noise = [(np.random.random() * 2 - 1.) * noise_percent * action_real[0],
                         (np.random.random() * 2 - 1.) * noise_percent * action_real[1]]
                action_real = [action_real[0] + noise[0], action_real[1] + noise[1]]
            actions_real.append(action_real)
        obs, rew, done, info = super().step(actions_real)

        if args.print_images:
            self.obs = (obs['rgb'] * 255.).astype(np.uint8)
        depth = _preprocess_depth(-obs['pc'][:, :, :, 2:3])

        for a in range(self.num_robots):
            # Get base sensor and ground-truth pose
            dx_gt, dy_gt, do_gt = self.get_gt_pose_change(a)
            self.curr_loc_gt[a] = pu.get_new_pose(self.curr_loc_gt[a], (dx_gt, dy_gt, do_gt))

            if (not args.noisy_odometry):
                self.curr_loc[a] = self.curr_loc_gt[a]
                dx_base, dy_base, do_base = dx_gt, dy_gt, do_gt
            else:
                dx_base, dy_base, do_base = self.get_base_pose_change(actions[a], (dx_gt, dy_gt, do_gt))

                self.curr_loc[a] = pu.get_new_pose(self.curr_loc[a], (dx_base, dy_base, do_base))
            
            self.info['sensor_pose'][a] = np.asarray([dx_base, dy_base, do_base])
            self.bump[a] = (obs['bump'][a] > 0.5)
            self.cont_bump[a] = 0.5 * self.cont_bump[a] + int(self.bump[a])

        self.info['l_reward'][:] = np.asarray(self.l_reward)
        self.info['bump'][:] = np.asarray(self.bump)

        # Convert pose to cm and degrees for mapper
        mapper_gt_pose = [(self.curr_loc_gt[a][0]*100.0,
                           self.curr_loc_gt[a][1]*100.0,
                           np.deg2rad(self.curr_loc_gt[a][2]),
                           self.get_pitch(a))
                           for a in range(self.num_robots)]

        self.mapper.update_map(depth, mapper_gt_pose)
        if args.print_images and args.vis_type == 2:
            self.obs = depth
        elif args.depth_obs:
            self.obs = self.mapper.depth

        for a in range(self.num_robots):
            # Update collision map
            r_threshold = self.r_threshold
            s_threshold = 0
            if self.escape[a] > 1:
                self.escape[a] -= 1
            elif self.escape[a] == 1:
                self.escape[a] = 0
                self.collision_dir[a] = 0

            if self.cont_bump[a] > 1.6:
                if (self.rotate[a] > r_threshold or self.rotate[a] < -r_threshold):
                    self.r_threshold += 2
                    self.collision_dir[a] = 3 if self.rotate[a] > 0 else 4
                    self.escape[a] = 8
                    depth_f = -obs['pc'][a, :, :, 2:3].mean()
                    self.heuristic_type = 0 if depth_f > 0.5 else 4
                    # self.heuristic_type = (self.heuristic_type + 4) % 8
                elif (self.straight[a] > s_threshold or self.straight[a] < -s_threshold):
                    self.collision_dir[a] = 1 if self.straight[a] > 0 else 2
                    self.escape[a] = 8
                    depth_l_minus_r = -obs['pc'][a, :, :, 2:3].sum(0)
                    depth_l_minus_r = (depth_l_minus_r[:depth_l_minus_r.shape[0] // 2] - depth_l_minus_r[depth_l_minus_r.shape[0] // 2:]).sum(0)
                    self.heuristic_type = 0 if depth_l_minus_r > 0 else 4
                    # self.heuristic_type = (self.heuristic_type + 4) % 8
            else:
                pass

        # Update ground_truth map and explored area
        self.obstacle_map, self.explored_map = self.mapper.get()
        if self.timestep % args.num_local_steps == 0:
            self.prev_frontier = self.curr_frontier
            self.curr_obstacle_map, self.curr_frontier = self.frontier()

        if (not self.close):
            if args.baseline != 'none':
                if self.long_term_planner.check_finish(self.curr_frontier, actions == 4):
                    self.baseline_goal = [None] * self.num_robots
                    print(self.timestep)
            else:
                if self.timestep % args.num_local_steps == 0:
                    self.baseline_goal = [None] * self.num_robots


        if (not self.close) and self.timestep%args.num_local_steps==0:
            
            g_reward = 0
            area, ratio = self.get_global_reward()
            self.info['exp_reward'][:] = area
            self.info['exp_ratio'][:] = ratio
            self.info['g_reward'][:] = (area - args.reward_bias) * args.reward_scale + g_reward
            self.collision_tolerance = args.max_collisions_allowed
        else:
            self.info['exp_reward'][:] = 0.0
            self.info['exp_ratio'][:] = 0.0
            self.info['g_reward'][:] = final_reward

        self.collision_tolerance -= (1 if any(self.bump) else 0)
        if self.collision_tolerance == 0 and (not self.close):
            self.close = True
            self.info['exp_reward'][:] = 0.0
            self.info['exp_ratio'][:] = 0.0
            self.info['g_reward'][:] = 0.0
            self.info['l_reward'][:] = 0.0
            self.info['bump'][:] = False
            self.info['sensor_pose'][:] = 0.

        if self.timestep%args.num_local_steps==0:
            self.info['obstacle'][:] = self.curr_obstacle_map
            self.info['frontier'][:] = self.curr_frontier
            self.info['explored'][:] = self.explored_map

        return self.close

    def frontier(self):
        obstacle_region = skimage.morphology.binary_dilation(self.obstacle_map, self.selem)

        # remove the small holes in the explored region
        explored_region = cv2.erode(cv2.dilate(self.explored_map, np.ones((3,3))), np.ones((3,3)))
        
        explored_boundary = explored_region - cv2.erode(explored_region, np.ones((3,3)))
        frontier = explored_boundary - obstacle_region
        return obstacle_region, np.clip(frontier, 0.0, 1.0)


    def get_global_reward(self):
        
        curr_explored = self.explored_map*self.explorable_map
        curr_explored_area = curr_explored.sum()

        reward_scale = self.explorable_map.sum()
        m_reward = (curr_explored_area - self.prev_explored_area)*1.
        m_ratio = m_reward/reward_scale
        m_reward = m_reward * (self.args.unit_size_cm ** 2) / 10000. # converting to m^2
        self.prev_explored_area = curr_explored_area

        return m_reward, m_ratio

    def get_spaces(self):
        return self.observation_space, self.action_space

    def build_selem(self, radius, invert):
        buf = int(ceil(radius))
        kernel_size = buf * 2 + 1
        radius2 = radius ** 2
        selem = np.zeros((kernel_size, kernel_size))
        for i in range(kernel_size):
            for j in range(kernel_size):
                if radius2 >= (i - buf) ** 2 + (j - buf) ** 2:
                    selem[i, j] = 1
        return (buf, 1. - selem) if invert else (buf, selem)

    def build_mapper(self):
        params = {}
        params['frame_width'] = self.args.env_frame_width
        params['frame_height'] = self.args.env_frame_height
        params['fov'] =  self.args.hfov
        params['depth_obs'] = self.args.depth_obs
        params['unit_size_cm'] = self.args.unit_size_cm
        params['mask_size'] = 2 * (2 * self.args.obstacle_boundary // self.args.unit_size_cm) + 1
        params['num_robots'] = self.args.num_robots
        params['map_size_cm'] = self.args.map_size_cm
        params['agent_height'] = (self.simulator.robots[0].eyes.get_position()[2] - self.simulator.robots[0].robot_body.get_position()[2]) * 100.
        params['du_scale'] = self.args.du_scale
        params['vision_range'] = self.args.vision_range
        params['obs_threshold'] = self.args.obs_threshold
        self.selem = skimage.morphology.disk(self.args.obstacle_boundary / self.args.unit_size_cm)
        self.mini_selem = self.build_selem((self.args.obstacle_boundary) / self.args.unit_size_cm, invert=True)
        mapper = MapBuilder(params)
        return mapper

    # get the camera pose
    def get_sim_location(self, idx):
        agent_pose = self.robots[idx].robot_body.get_pose()
        x = -agent_pose[0]
        y = -agent_pose[1]
        o = self.robots[idx].robot_body.get_rpy()[2] + np.pi
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pitch(self, idx):
        return self.robots[idx].eyes.get_rpy()[1]

    def get_camera_info(self, idx=-1):
        intrinsics = []
        extrinsics = []
        for i, robot in enumerate(self.robots):
            if idx >= 0 and i != idx:
                continue

            renderer = self.simulator.renderer
            intrinsics.append(renderer.get_intrinsics())

            extrinsics_x, extrinsics_y, extrinsics_z = self.curr_loc_gt[i][0], self.curr_loc_gt[i][1], self.mapper.agent_height/100
            extrinsics_roll, extrinsics_pitch, extrinsics_yaw = self.robots[i].robot_body.get_rpy()[0], -self.robots[i].robot_body.get_rpy()[1], np.deg2rad(self.curr_loc_gt[i][2])  - np.pi / 2
            
            rot_pitch = ru.get_r_matrix([1., 0., 0.], angle=extrinsics_pitch)
            rot_roll = ru.get_r_matrix([0., 1., 0.], angle=extrinsics_roll)
            rot_yaw = ru.get_r_matrix([0., 0., 1.], angle=extrinsics_yaw)
            extrinsics_rot = np.matmul(np.matmul(rot_yaw, rot_roll), rot_pitch)

            extrinsics_trans = np.array([extrinsics_x, extrinsics_y, extrinsics_z])
            
            extrinsics_matrix = np.eye(4)
            extrinsics_matrix[:3,:3] = extrinsics_rot
            extrinsics_matrix[:3,3] = extrinsics_trans
            # print(extrinsics_matrix)
            extrinsics.append(extrinsics_matrix)

        return (intrinsics, extrinsics)


    ### The world coordinate system is defined according to the initial agent pose. In order to achieve that, the world map in the iGibson environment is transformed by inverting the initial agent pose to form the new world coordinate system.
    def _get_gt_map(self, full_map_size, floor_num):
        self.scene_name = self.config['scene_id']
        logger.error('Computing map for %s', self.scene_name)

        # Get map in gibson simulator coordinates
        self.map_obj = GibsonMaps(self, floor_num, N=int(1e4), resolution=self.args.unit_size_cm)
        if self.map_obj.size[0] < 1 or self.map_obj.size[1] < 1:
            logger.error("Invalid map: {}/{}".format(self.scene_name, self.current_episode))
            return None

        #TODO: multi-agent
        agent_pose = self.robots[0].robot_body.get_pose()
        agent_y = agent_pose[2]*100.
        sim_map = self.map_obj.get_map(agent_y, -50., 50.0)
        sim_map = cv2.erode(cv2.dilate(sim_map.astype(np.float32), np.ones((5,5))), np.ones((5,5)))
        sim_map[sim_map > 0] = 1.

        min_x, min_y = self.map_obj.origin/100.0
        x, y, o = self.get_sim_location(0)
        x, y = -x - min_x, -y - min_y
        range_x, range_y = self.map_obj.max/100. - self.map_obj.origin/100.

        map_size = sim_map.shape
        scale = 2.
        grid_size = int(scale*max(map_size))
        grid_map = np.zeros((grid_size, grid_size))

        grid_map[(grid_size - map_size[0])//2:
                 (grid_size - map_size[0])//2 + map_size[0],
                 (grid_size - map_size[1])//2:
                 (grid_size - map_size[1])//2 + map_size[1]] = sim_map

        if map_size[0] > map_size[1]:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale) * map_size[1] * 1. / map_size[0],
                    (y - range_y/2.) * 2. / (range_y * scale),
                    180.0 + (0 if self.args.eval else np.rad2deg(o))
                ]])
        else:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale),
                    (y - range_y/2.) * 2. / (range_y * scale) * map_size[0] * 1. / map_size[1],
                    180.0 + (0 if self.args.eval else np.rad2deg(o))
                ]])

        st[0, 0:2] = 0.

        rot_mat, trans_mat = get_grid(st, (1, 1, grid_size, grid_size), torch.device("cpu"))

        grid_map = torch.from_numpy(grid_map).float()
        grid_map = grid_map.unsqueeze(0).unsqueeze(0)
        translated = F.grid_sample(grid_map, trans_mat)
        rotated = F.grid_sample(translated, rot_mat)

        episode_map = torch.zeros((full_map_size, full_map_size)).float()
        if full_map_size > grid_size:
            episode_map[(full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size,
                        (full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size] = rotated[0,0]
        else:
            episode_map = rotated[0,0,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size]

        episode_map = episode_map.numpy()
        episode_map[episode_map > 0] = 1.

        return episode_map

    def _get_scene_full_size(self, floor_num):
        self.scene_name = self.config['scene_id']
        self.map_obj = GibsonMaps(self, floor_num, N=int(1e4), resolution=self.args.unit_size_cm)
        if self.map_obj.size[0] < 1 or self.map_obj.size[1] < 1:
            logger.error("Invalid map: {}/{}".format(self.scene_name, self.current_episode))
            return False
        agent_pose = self.robots[0].robot_body.get_pose()
        agent_y = agent_pose[2]*100.
        sim_map = self.map_obj.get_map(agent_y, -50., 50.0)
        sim_map[sim_map > 0] = 1.
        map_size = sim_map.shape
        return int(2 * max(map_size))

    def get_gt_pose_change(self, idx):
        curr_sim_pose = self.get_sim_location(idx)
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location[idx])
        self.last_sim_location[idx] = curr_sim_pose
        return dx, dy, do


    def get_base_pose_change(self, action, gt_pose_change):
        dx_gt, dy_gt, do_gt = gt_pose_change
        x_r, y_r, o_r = [np.random.random() * 2 - 1. for _ in range(3)]
        if action in [2, 3]: ## Forward & Backward
            x_err, y_err, o_err = x_r * 0.005, y_r * 0.005, o_r * 0.005
        elif action in [0, 1]: ## Left & Right
            x_err, y_err, o_err = x_r * 0.003, y_r * 0.003, o_r * 0.005
        else: ## Stop
            x_err, y_err, o_err = 0., 0., 0.

        x_err = x_err * self.args.noise_level
        y_err = y_err * self.args.noise_level
        o_err = o_err * self.args.noise_level
        return dx_gt + x_err, dy_gt + y_err, do_gt + np.deg2rad(o_err)


    def get_short_term_goal(self, inputs, heatmap):
        args = self.args
        outputs = []
        goals = []
        poses = []
        poses_gt = []
        stgs = []
        local_window = []
        close = True

        if self.close:
            return np.zeros((self.num_robots))

        for i in range(self.num_robots):
            # Get pose prediction and global policy planning window
            start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs[i]['pose_pred']
            gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
            
            r, c = start_y, start_x
            start = [int(r * 100.0/args.unit_size_cm - gx1),
                    int(c * 100.0/args.unit_size_cm - gy1)]
            start = pu.threshold_poses(start, (gx2-gx1, gy2-gy1))
            start = [start[0], start[1], start_o]
            
            x10 = start[0]+gx1-self.mini_selem[0]
            x20 = start[0]+gx1+self.mini_selem[0]
            y10 = start[1]+gy1-self.mini_selem[0]
            y20 = start[1]+gy1+self.mini_selem[0]
            x1, y1 = pu.threshold_poses([x10, y10], self.pos_dilation_mask[i].shape)
            x2, y2 = pu.threshold_poses([x20, y20], self.pos_dilation_mask[i].shape)
            self.pos_map[i, start[0]+gx1, start[1]+gy1] = 1.
            self.pos_dilation_mask[i, x1:x2, y1:y2] *= self.mini_selem[1][x1-x10:x2-x10, y1-y10:y2-y10]

            self.obstacle_map *= self.pos_dilation_mask[i]

        actions = [0] * self.num_robots
        for i in range(self.num_robots):
            # Get pose prediction and global policy planning window
            start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs[i]['pose_pred']
            gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
            planning_window = [gx1, gx2, gy1, gy2]
            
            # grid = map_pred
            grid = self.obstacle_map[gx1:gx2, gy1:gy2]
            explored = self.explored_map[gx1:gx2, gy1:gy2]

            # Get last loc
            last_start_x, last_start_y = self.last_loc[i][0], self.last_loc[i][1]
            r, c = last_start_y, last_start_x
            last_start = [int(r * 100.0/args.unit_size_cm - gx1),
                        int(c * 100.0/args.unit_size_cm - gy1)]
            last_start = pu.threshold_poses(last_start, grid.shape)

            # Get curr loc
            self.curr_loc[i] = [start_x, start_y, start_o]
            r, c = start_y, start_x
            start = [int(r * 100.0/args.unit_size_cm - gx1),
                    int(c * 100.0/args.unit_size_cm - gy1)]
            start = pu.threshold_poses(start, grid.shape)
            start = [start[0], start[1], start_o]

            # Get last loc ground truth pose
            last_start_x, last_start_y = self.last_loc_gt[i][0], self.last_loc_gt[i][1]
            r, c = last_start_y, last_start_x
            last_start = [int(r * 100.0/args.unit_size_cm),
                        int(c * 100.0/args.unit_size_cm)]
            last_start = pu.threshold_poses(last_start, self.visited_gt.shape[1:])

            # Get ground truth pose
            start_x_gt, start_y_gt, start_o_gt = self.curr_loc_gt[i]
            r, c = start_y_gt, start_x_gt
            start_gt = [int(r * 100.0/args.unit_size_cm),
                        int(c * 100.0/args.unit_size_cm)]
            start_gt = pu.threshold_poses(start_gt, self.visited_gt.shape[1:])

            steps = 25
            for j in range(steps):
                x = int(last_start[0] + (start_gt[0] - last_start[0]) * (j+1) / steps)
                y = int(last_start[1] + (start_gt[1] - last_start[1]) * (j+1) / steps)
                self.visited_gt[i, x, y] = 1

            # Get goal
            if self.baseline_goal[i] is not None:
                goal = self.baseline_goal[i]
                goal = [goal[0] - gx1, goal[1] - gy1, goal[2]]
            else:
                goal = inputs[i]['goal']
                goal = pu.threshold_poses(goal, grid.shape)

            grid = self.obstacle_map[gx1:gx2, gy1:gy2]

            # Get short-term goal
            stg = self._get_stg(grid, explored, start, np.copy(goal), planning_window, i)

            if self.baseline_goal[i] is not None:
                goal = self.baseline_goal[i]
                goal = [goal[0] - gx1, goal[1] - gy1, goal[2]]

            if close and self.timestep%args.num_local_steps == 0:
                close = close and self._is_close(self.obstacle_map, (start[0] + gx1, start[1] + gy1), self.pos_map[i])

            (stg_x, stg_y) = stg
            relative_dist = pu.get_l2_distance(stg_x, start[0], stg_y, start[1])
            relative_dist = relative_dist*args.unit_size_cm/100.
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o)%360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal)%360.0
            if relative_angle > 180:
                relative_angle -= 360

            self.last_stg[i] = stg

            ra = int((relative_angle%360.)/5.)
            if self.collision_dir[i] > 0:
                actions[i] = [
                    [3, 3, 0, 0, 0, 0, 2, 2],
                    [2, 2, 0, 0, 0, 0, 3, 3],
                    [2, 2, 1, 1, 1, 1, 3, 3],
                    [2, 2, 0, 0, 0, 0, 0, 0],
                    [3, 3, 1, 1, 1, 1, 2, 2],
                    [2, 2, 1, 1, 1, 1, 3, 3],
                    [3, 3, 1, 1, 1, 1, 2, 2],
                    [3, 3, 0, 0, 0, 0, 2, 2]
                ][self.heuristic_type + self.collision_dir[i] - 1][8 - self.escape[i]]
            elif relative_dist < 0.01:
                actions[i] = 4
            elif ra < 2 or ra > 69:
                actions[i] = 2
            elif ra < 36:
                actions[i] = 1
            else:
                actions[i] = 0
            output = actions[i]
                
            outputs.append(output)
            
            if args.print_images or heatmap is not None:
                if args.vis_type == 1:
                    if i == 0:
                        local_window = [gx1, gx2, gy1, gy2]
                    self.local_map[gx1:gx2, gy1:gy2] = grid
                    poses.append(pu.threshold_poses([start_x - local_window[2]*args.unit_size_cm/100.0, start_y - local_window[0]*args.unit_size_cm/100.0], grid.shape) + [start_o])
                    poses_gt.append(pu.threshold_poses([start_x_gt - local_window[2]*args.unit_size_cm/100.0, start_y_gt - local_window[0]*args.unit_size_cm/100.0], grid.shape) + [start_o_gt])
                    goals.append(pu.threshold_poses([goal[0]+gx1-local_window[0], goal[1]+gy1-local_window[2]], grid.shape))
                    stgs.append(pu.threshold_poses([int(stg[0])+gx1-local_window[0], int(stg[1])+gy1-local_window[2]], grid.shape))
                else:
                    poses.append((start_x_gt, start_y_gt, start_o_gt))
                    poses_gt.append((start_x_gt, start_y_gt, start_o_gt))
                    goals.append((goal[0]+gx1, goal[1]+gy1))
                    stgs.append((int(stg[0])+gx1, int(stg[1])+gy1))

        if (len([1 for i in actions if i != 4]) == 0 or (len([1 for a in range(self.num_robots) if (self.curr_loc_gt[a][0] - self.last_loc_gt[a][0]) ** 2 + (self.curr_loc_gt[a][1] - self.last_loc_gt[a][1]) ** 2 > 0.01 ** 2]) == 0)) and self.timestep%args.num_local_steps == 0:
            self.cont_stuck += 1
        elif self.timestep%args.num_local_steps == 0:
            self.cont_stuck = 0
        
        if close and self.timestep%args.num_local_steps == 0:
            self.close = True
            self.info['exp_reward'][:] = 0.0
            self.info['exp_ratio'][:] = 0.0
            self.info['g_reward'][:] = 0.0
            self.info['l_reward'][:] = 0.0
            self.info['bump'][:] = False
            self.info['sensor_pose'][:] = 0.

        if args.print_images or heatmap is not None:
            dump_dir = "{}/dump/{}/".format(args.dump_location,
                                                args.exp_name)
            ep_dir = '{}/episodes/{}/{}/'.format(
                            dump_dir, self.rank+1, self.current_episode)
            if not os.path.exists(ep_dir):
                os.makedirs(ep_dir)

            vis_info = {
                'close': self.close,
                'actions': actions,
                'bump': self.bump
            }

            if args.vis_type == 1:
                gx1, gx2, gy1, gy2 = local_window
                vis_grid = vu.get_colored_map(
                    skimage.morphology.binary_dilation(self.local_map[gx1:gx2, gy1:gy2], self.selem),
                    self.visited_gt[:, gx1:gx2, gy1:gy2],
                    goals,
                    stgs,
                    self.explored_map[gx1:gx2, gy1:gy2],
                    self.explorable_map[gx1:gx2, gy1:gy2],
                    self.obstacle_map[gx1:gx2, gy1:gy2],
                    self.curr_frontier[gx1:gx2, gy1:gy2]
                )
                vis_grid = np.flipud(vis_grid)
                vu.visualize(self.figure, self.ax,
                            self.obs, vis_grid[:,:,::-1], heatmap,
                            poses, poses_gt, goals, vis_info,
                            dump_dir, self.rank, self.current_episode,
                            self.timestep, False,
                            args.print_images, args.vis_type, args.unit_size_cm)

            elif args.vis_type == 2:
                obstacle_map, frontier = self.frontier()
                vu.dump(
                    obstacle_map,
                    self.explored_map,
                    self.explorable_map,
                    frontier,
                    self.obs,
                    [[*poses_gt[i][:2], self.get_sim_location(i)[2]] for i in range(self.num_robots)],
                    self.visited_gt,
                    goals,
                    stgs,
                    vis_info,
                    dump_dir,
                    args.unit_size_cm,
                    self.rank,
                    self.current_episode,
                    self.timestep
                )
            elif args.vis_type == 3:
                vis_grid = vu.get_colored_map(
                    skimage.morphology.binary_dilation(self.obstacle_map, self.selem),
                    self.visited_gt,
                    goals,
                    stgs,
                    self.explored_map,
                    self.explorable_map,
                    self.obstacle_map,
                    self.curr_frontier
                )
                vis_grid = np.flipud(vis_grid)
                vu.visualize(self.figure, self.ax,
                            self.obs, vis_grid[:,:,::-1], heatmap,
                            poses, poses_gt, goals, vis_info,
                            dump_dir, self.rank, self.current_episode,
                            self.timestep, False,
                            args.print_images, args.vis_type, args.unit_size_cm)

        return np.stack(outputs)
    
    def _is_close(self, grid, start, pos_map):
        rows = grid.sum(1)
        rows[rows>0] = 1
        ex1 = np.argmax(rows)
        ex2 = len(rows) - np.argmax(np.flip(rows))

        cols = grid.sum(0)
        cols[cols>0] = 1
        ey1 = np.argmax(cols)
        ey2 = len(cols) - np.argmax(np.flip(cols))

        ex1 = max(0, min(int(start[0]) - 2, ex1))
        ex2 = min(grid.shape[0], max(int(start[0]) + 2, ex2))
        ey1 = max(0, min(int(start[1]) - 2, ey1))
        ey2 = min(grid.shape[1], max(int(start[1]) + 2, ey2))


        start = [start[0] - ex1, start[1] - ey1]
        buf = 6
        traversible = (skimage.morphology.binary_dilation(np.pad(grid[ex1:ex2, ey1:ey2], ((buf, buf), (buf, buf)), 'constant'), self.selem) != True).astype(np.int32)
        traversible[buf:-buf, buf:-buf][pos_map[ex1:ex2, ey1:ey2] == 1] = 1
        traversible[buf:-buf, buf:-buf][int(start[0])-1:int(start[0])+2, int(start[1])-1:int(start[1])+2] = 1

        planner = FMMPlanner(traversible, use_distance_field=self.args.use_distance_field)
        ret = not planner.reachable([int(start[0]) + buf, int(start[1]) + buf], np.pad(self.curr_frontier, ((buf, buf), (buf, buf)), 'constant')[ex1:ex2+buf*2, ey1:ey2+buf*2])

        return ret



    def _get_stg(self, grid, explored, start, goal, planning_window, idx):

        pos_map = self.pos_map[idx]
        [gx1, gx2, gy1, gy2] = planning_window

        rows = explored.sum(1)
        rows[rows>0] = 1
        ex1 = np.argmax(rows)
        ex2 = len(rows) - np.argmax(np.flip(rows))

        cols = explored.sum(0)
        cols[cols>0] = 1
        ey1 = np.argmax(cols)
        ey2 = len(cols) - np.argmax(np.flip(cols))

        ex1 = max(0, min(int(start[0]) - 2, ex1))
        ex2 = min(grid.shape[0], max(int(start[0]) + 2, ex2))
        ey1 = max(0, min(int(start[1]) - 2, ey1))
        ey2 = min(grid.shape[1], max(int(start[1]) + 2, ey2))
        
        if self.args.baseline != 'none' and self.baseline_goal[idx] is None:
            agent_pos = None
            obstacle = None
            if idx == 0:
                obstacle = skimage.morphology.binary_dilation(self.obstacle_map, self.selem)
                agent_pos = np.asarray([pu.threshold_poses([int(r * 100.0/self.args.unit_size_cm), int(c * 100.0/self.args.unit_size_cm)], obstacle.shape) for c, r, _ in self.curr_loc])
            goal = self.long_term_planner.get_long_term_goal(obstacle, self.curr_frontier, agent_pos, idx)
            if goal is None:
                goal = [0, 0, 0]
            else:
                if goal[0] >= gx2 or goal[1] >= gy2 or goal[0] < gx1 or goal[1] < gy1:
                    if obstacle is None:
                        obstacle = skimage.morphology.binary_dilation(self.obstacle_map, self.selem)
                    goal = self.long_term_planner.replan(obstacle, self.curr_frontier, goal, planning_window)
                goal = [goal[0] - gx1, goal[1] - gy1, (1 if self.args.baseline in ['seg', 'grd', 'mtsp', 'coscan'] else 0)]
        real_goal = np.copy(goal)
        x1 = min(start[0], goal[0] - goal[2])
        x2 = max(start[0], goal[0] + goal[2])
        y1 = min(start[1], goal[1] - goal[2])
        y2 = max(start[1], goal[1] + goal[2])
        dist = pu.get_l2_distance(goal[0], start[0], goal[1], start[1])
        buf = max(40., dist)
        x1 = max(1, int(x1 - buf))
        x2 = min(grid.shape[0]-1, int(x2 + buf))
        y1 = max(1, int(y1 - buf))
        y2 = min(grid.shape[1]-1, int(y2 + buf))
        
        x1 = max(x1, ex1)
        x2 = min(x2, ex2)
        y1 = max(y1, ey1)
        y2 = min(y2, ey2)

        traversible = skimage.morphology.binary_dilation(
                        self.mapper.mask_local(grid[x1:x2, y1:y2], (gy1+y1, gy1+y2, gx1+x1, gx1+x2), idx),
                        self.selem) != True
        traversible[pos_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
        traversible[int(start[0]-x1):int(start[0]-x1+1), int(start[1]-y1):int(start[1]-y1+1)] = 1

        goal[0] = min(max(x1 + goal[2], goal[0]), x2-goal[2])
        goal[1] = min(max(y1 + goal[2], goal[1]), y2-goal[2])

        def add_boundary(mat):
            h, w = mat.shape
            new_mat = np.ones((h+2,w+2))
            new_mat[1:h+1,1:w+1] = mat
            return new_mat

        traversible = add_boundary(traversible)

        planner = FMMPlanner(traversible, use_distance_field=self.args.use_distance_field)

        if self.args.baseline != 'none' and self.baseline_goal[idx] is None:
            if 0 and self.args.num_robots == 1:
                goals = goals[x1:x2, y1:y2]
                if goals.max() < 1.:
                    goals = np.pad(goals, ((1, 1), (1, 1)), 'constant', constant_values=(1, 1))
                else:
                    goals = np.pad(goals, ((1, 1), (1, 1)), 'constant')
                goal = planner.nearest(goals, (start[0] - x1 + 1, start[1] - y1 + 1))
                goal = (goal[0] + x1 - 1, goal[1] + y1 - 1, 0)
                    
            self.baseline_goal[idx] = [goal[0] + gx1, goal[1] + gy1, goal[2]]
        reachable = planner.set_goal([goal[1]-y1+1, goal[0]-x1+1, goal[2]])

        stg_x, stg_y = start[0] - x1 + 1, start[1] - y1 + 1
        for i in range(self.args.short_goal_dist):
            stg_x, stg_y, replan = planner.get_short_term_goal([stg_x, stg_y], start[2])

        if replan:
            stg_x, stg_y = start[0], start[1]
        else:
            stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y)
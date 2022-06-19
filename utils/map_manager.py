import numpy as np
import torch
import torch.nn as nn
from utils.input_transform import global_input_transform, distance_field


def get_local_map_boundaries(agent_location, local_map_size, global_map_size):
    agent_location_r, agent_location_c = agent_location
    local_map_w, local_map_h = local_map_size
    global_map_w, global_map_h = global_map_size

    if local_map_size != global_map_size:
        gx1, gy1 = agent_location_r - local_map_w // 2, agent_location_c - local_map_h // 2
        gx2, gy2 = gx1 + local_map_w, gy1 + local_map_h
        if gx1 < 0:
            gx1, gx2 = 0, local_map_w
        if gx2 > global_map_w:
            gx1, gx2 = global_map_w - local_map_w, global_map_w

        if gy1 < 0:
            gy1, gy2 = 0, local_map_h
        if gy2 > global_map_h:
            gy1, gy2 = global_map_h - local_map_h, global_map_h
    else:
        gx1, gx2, gy1, gy2 = 0, global_map_w, 0, global_map_h

    return [gx1, gx2, gy1, gy2]

def get_new_pose_batch(pose, rel_pose_change):
    pose[:, 1] += rel_pose_change[:, 0] * np.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * np.cos(pose[:, 2] / 57.29577951308232)
    pose[:, 0] += rel_pose_change[:, 0] * np.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * np.sin(pose[:, 2] / 57.29577951308232)
    pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

    pose[:, 2] = np.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
    pose[:, 2] = np.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

    return pose


class MapManager:
    def __init__(self, args, global_map_w, global_map_h, local_map_w, local_map_h, device):
        self.num_scenes = args.num_processes
        self.num_robots = args.num_robots
        self.unit_size_cm = args.unit_size_cm
        self.global_map_w = global_map_w
        self.global_map_h = global_map_h
        self.local_map_w = local_map_w
        self.local_map_h = local_map_h
        self.g_input_content = args.g_input_content
        self.global_downscaling = args.global_downscaling
        self.centralized = args.centralized
        self.device = device

        # Initializing full and local map
        # obs/frontier/all pos/all trajectory/explored/explorable/history pos
        self.global_map = torch.zeros(self.num_scenes, 8, global_map_w, global_map_h).float().to(device)
        
        # g_input: l_obs/l_frontier/l_other_pos/l_all_trajectory/l_pos
        #          g_obs/g_frontier/g_all_pos/g_all_trajectory

        # Initial full and local pose
        self.global_pose = np.zeros((self.num_scenes * self.num_robots, 3))
        self.local_pose = np.zeros((self.num_scenes * self.num_robots, 3))

        # Origin of local map
        self.local_map_origins = np.zeros((self.num_scenes * self.num_robots, 3))

        # Local Map Boundaries (min x & y, max x & y in the global map)
        self.local_map_boundary = np.zeros((self.num_scenes * self.num_robots, 4)).astype(np.int32)

        ### Planner pose inputs has 7 dimensions
        ### 1-3 store continuous global agent location
        ### 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros((self.num_scenes, self.num_robots, 7))

        if args.centralized:
            self.global_input = torch.zeros(1)
        else:
            self.global_input = torch.zeros(self.num_scenes, self.num_robots, 9, self.local_map_w // 2, self.local_map_h // 2).to(device)
        self.global_orientation = torch.zeros(self.num_scenes, self.num_robots, 1).long()
        self.global_pos = torch.zeros(self.num_scenes * self.num_robots, 6).long()



    def init_map_and_pose(self, origin_pose):
        self.global_map.fill_(0.)
        self.global_input.fill_(0.)
        self.global_pose[:] = origin_pose
        self.planner_pose_inputs[:, :, :3] = origin_pose.reshape(self.num_scenes, self.num_robots, 3)
        agent_location = (origin_pose * (100.0 / self.unit_size_cm)).astype(np.long)
        for e in range(self.num_scenes * self.num_robots):
            agent_location_r, agent_location_c = agent_location[e, 1], agent_location[e, 0]
            self.global_pos[e, :2] = torch.tensor((agent_location_r, agent_location_c))
            self.global_map[e // self.num_robots, 3, agent_location_r, agent_location_c] = 1.

            self.local_map_boundary[e] = get_local_map_boundaries((agent_location_r, agent_location_c), (self.local_map_w, self.local_map_h), (self.global_map_w, self.global_map_h))

            self.planner_pose_inputs[e // self.num_robots, e % self.num_robots, 3:] = self.local_map_boundary[e]
            self.local_map_origins[e] = [self.local_map_boundary[e, 2] * self.unit_size_cm / 100.0, self.local_map_boundary[e, 0] * self.unit_size_cm / 100.0, 0.]
        self.local_pose = self.global_pose - self.local_map_origins


    def update_local(self, sensor_pose):
        self.local_pose = get_new_pose_batch(self.local_pose, sensor_pose)
        global_pose = self.local_pose + self.local_map_origins
        self.planner_pose_inputs[:, :, :3] = global_pose.reshape(self.num_scenes, self.num_robots, 3)
        agent_location = (global_pose * (100.0 / self.unit_size_cm)).astype(np.int32)
        for e in range(self.num_scenes * self.num_robots):
            r, c = self.local_pose[e, 1], self.local_pose[e, 0]
            agent_location_r, agent_location_c = agent_location[e, 1], agent_location[e, 0]
            agent_location_r = max(0, min(self.global_map_w, agent_location_r))
            agent_location_c = max(0, min(self.global_map_h, agent_location_c))
            self.global_map[e // self.num_robots, 3, agent_location_r, agent_location_c] = 1.
            self.global_orientation[e // self.num_robots, e % self.num_robots] = int((self.local_pose[e, 2] + 180.0) / 5.)

    
    def update_global(self, obstacle, frontier, explored, explorable):
        self.global_map[:, 0, :, :] = torch.from_numpy(obstacle).float()
        self.global_map[:, 1, :, :] = torch.from_numpy(frontier).float()
        self.global_map[:, 4, :, :] = torch.from_numpy(explored).float()
        self.global_map[:, 5, :, :] = torch.from_numpy(explorable).float()
        self.global_map[:, 2, :, :].fill_(0.)
        lmb = self.local_map_boundary
        self.global_pose = self.local_pose + self.local_map_origins
        agent_location = (self.global_pose * (100.0 / self.unit_size_cm)).astype(np.long)
        for e in range(self.num_scenes * self.num_robots):
            agent_location_r, agent_location_c = agent_location[e, 1], agent_location[e, 0]
            lmb[e] = get_local_map_boundaries((agent_location_r, agent_location_c), (self.local_map_w, self.local_map_h), (self.global_map_w, self.global_map_h))
            agent_location_r = max(0, min(self.global_map_w, agent_location_r))
            agent_location_c = max(0, min(self.global_map_h, agent_location_c))
            self.global_map[e // self.num_robots, [2, 6], agent_location_r, agent_location_c] = 1.
            self.global_pos[e, :2] = torch.tensor((agent_location_r, agent_location_c))
            self.planner_pose_inputs[e // self.num_robots, e % self.num_robots, 3:] = lmb[e]
            self.local_map_origins[e] = [lmb[e, 2] * self.unit_size_cm / 100.0, lmb[e, 0] * self.unit_size_cm / 100.0, 0.]
        self.local_pose = self.global_pose - self.local_map_origins



    def get_global_input(self, g_history):
        if self.centralized:
            self.global_pos[:, 2:] = torch.from_numpy(self.local_map_boundary)
            global_input = nn.MaxPool2d(4)(self.global_map)
            global_input[:, 6, :, :] -= global_input[:, 2, :, :]
            global_input[:, 7, :, :] = g_history
            global_input[:, 1, :, :][global_input[:, 2, :, :] > 0] = 1
            dist_input = torch.zeros((self.num_scenes, self.num_robots, self.global_map.size(2), self.global_map.size(3)))
            obstacle = self.global_map[:, 0, :, :].bool()

            rows = obstacle.any(2).cpu().numpy()
            cols = obstacle.any(1).cpu().numpy()
            obstacle = obstacle.cpu()

            for e in range(self.num_scenes):
                for a in range(self.num_robots):
                    agent_pos = self.global_pos[e * self.num_robots + a, :2]
                    dist_input[e, a, agent_pos[0], agent_pos[1]] = 1
                    row = np.copy(rows[e])
                    col = np.copy(cols[e])
                    row[agent_pos[0]] = True
                    col[agent_pos[1]] = True
                    distance_field(dist_input[e, a, :, :], obstacle[e, :, :], optimized=(row, col))
            dist_input = dist_input.to(self.device)
            dist_input[self.global_map[:, 1:2, :, :].repeat(1, self.num_robots, 1, 1) == 0] = 4
            for e in range(self.num_scenes):
                for a in range(self.num_robots):
                    agent_pos = self.global_pos[e * self.num_robots + a, :2]
                    dist_input[e, a, agent_pos[0], agent_pos[1]] = 4
            dist_input = -nn.MaxPool2d(4)(-dist_input)
            dist_input[dist_input > 4] = 4
            global_input = torch.cat((global_input, dist_input), dim=1)
            return global_input, self.global_pos

        self.global_input[:, :, 4].fill_(0.)
        for e in range(self.num_scenes * self.num_robots):
            x1, x2, y1, y2 = self.local_map_boundary[e]
            self.global_input[e // self.num_robots, e % self.num_robots, :4, :, :] = nn.MaxPool2d(2)(self.global_map[e // self.num_robots, :4, x1:x2, y1:y2])
            r, c = self.global_pos[e, 0] - x1, self.global_pos[e, 1] - y1
            assert self.global_input[e // self.num_robots, e % self.num_robots, 2, r // 2, c // 2] == 1
            self.global_input[e // self.num_robots, e % self.num_robots, 2, r // 2, c // 2] -= 1.
            self.global_input[e // self.num_robots, e % self.num_robots, 4, r // 2, c // 2] = 1.
        

        self.global_input[:, :, 5:, :, :] = torch.repeat_interleave(nn.MaxPool2d(2 * self.global_downscaling)(self.global_map[:, :4, :, :]).view(self.num_scenes, 1, 4, self.local_map_w // 2, self.local_map_h // 2), repeats=self.num_robots, dim=1)


        return self.global_input, torch.cat((self.global_pos, self.global_orientation.view(-1, 1)), dim=1)

    def get_planner_input(self):
        return self.planner_pose_inputs

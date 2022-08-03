import numpy as np
import skfmm
from scipy.optimize import linear_sum_assignment
from random import shuffle

# ------------------------------------------------------------------
# Global Policy - implementation of "Multi-robot collaborative dense scene reconstruction"

lazy_distance_map = None

def distance_map(shape, cx, cy):
    global lazy_distance_map
    h, w = shape
    if lazy_distance_map is None or lazy_distance_map.shape != (2*h+1, 2*w+1):
        lazy_distance_map = np.zeros((2*h+1, 2*w+1))
        for i in range(2*h+1):
            for j in range(2*w+1):
                lazy_distance_map[i, j] = (i - h) ** 2 + (j - w) ** 2
    return lazy_distance_map[h-cx:2*h-cx, w-cy:2*w-cy]

def kmeans(np_obstacle_map, np_frontier_map, k):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :return: numpy array of shape (k, 2)
    """
    # geodesic distance

    is_frontier = np_frontier_map == 1
    frontier_idx = np.where(is_frontier)
    rows = frontier_idx[0].shape[0]
    if rows == 0:
        return None, None, None, None
    if rows < k:
        k = rows

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    # the Forgy method will fail if the whole array contains the same rows
    cluster_idx = np.arange(rows)
    shuffle(cluster_idx)
    cluster_idx = cluster_idx[:k]

    for count in range(5):

        for k_i, ci in enumerate(cluster_idx):
            cx, cy = frontier_idx[0][ci], frontier_idx[1][ci]

            np_obstacle_map_frontierK = np.ma.masked_values(np_obstacle_map, 1)
            np_obstacle_map_frontierK[cx, cy] = 1
            np_obstacle_map_distance = skfmm.distance(1 - np_obstacle_map_frontierK)
            distances[:, k_i] = np_obstacle_map_distance[frontier_idx[0], frontier_idx[1]]

            # distances[:, k_i] = (frontier_idx[0] - cx)**2 + (frontier_idx[1] - cy)**2
            
            
        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for k_i in range(k):
            nearest_clusters_k_i_idx = np.where(nearest_clusters == k_i)
            if not nearest_clusters_k_i_idx:
                continue
            
            nearest_clusters_k_i_idx = nearest_clusters_k_i_idx[0]

            if frontier_idx[0][nearest_clusters_k_i_idx].size == 0 or frontier_idx[1][nearest_clusters_k_i_idx].size == 0:
                continue

            np_obstacle_map_frontierK = np.ma.masked_values(np_obstacle_map, 1)
            np_obstacle_map_frontierK[int(frontier_idx[0][nearest_clusters_k_i_idx].mean()), int(frontier_idx[1][nearest_clusters_k_i_idx].mean())] = 1
            np_obstacle_map_distance = skfmm.distance(1 - np_obstacle_map_frontierK)

            # cx = int(frontier_idx[0][nearest_clusters_k_i_idx].mean())
            # cy = int(frontier_idx[1][nearest_clusters_k_i_idx].mean())

            # np_obstacle_map_distance = distance_map(np_obstacle_map.shape, cx, cy)
            
            
            temp_distance_k_i = np_obstacle_map_distance[is_frontier][nearest_clusters_k_i_idx]
            cluster_idx[k_i] = nearest_clusters_k_i_idx[np.argmin(temp_distance_k_i)]

        last_clusters = nearest_clusters

    return (frontier_idx[0][cluster_idx], frontier_idx[1][cluster_idx]), nearest_clusters, is_frontier, frontier_idx



class CoScanPlanner:
    def __init__(self):
        self.global_goals = None
        
    def get_coscan_goals(self, np_obstacle_map, np_frontier_map, agent_pos):
        num_agent = agent_pos.shape[0]
        np_frontier_map = np.copy(np_frontier_map)
        np_obstacle_map_distance = []
        dd_mask = np.ones(np_obstacle_map.shape, dtype=np.bool)
        for k_i in range(num_agent):
            np_obstacle_map_frontierK = np.ma.masked_values(np_obstacle_map, 1)
            np_obstacle_map_frontierK[agent_pos[k_i, 0], agent_pos[k_i, 1]] = 1
            dd = skfmm.distance(1 - np_obstacle_map_frontierK)
            np_obstacle_map_distance.append(dd)
            if type(dd) is np.ndarray:
                dd_mask[:] = False
            else:
                dd_mask &= dd.mask
        np_frontier_map[dd_mask] = 0
        cluster_center, nearest_clusters, is_frontier, frontier_idx = kmeans(np_obstacle_map, np_frontier_map, num_agent)
        # nearest_clusters: (n_frontier,)
        if cluster_center is None:
            self.global_goals = [None] * num_agent
        else:
            nc = cluster_center[0].shape[0]
            cost = np.zeros((num_agent, nc))
            # n_agent x n_cluster
            for k_i in range(num_agent):
                cost[k_i, :] = np_obstacle_map_distance[k_i][cluster_center[0], cluster_center[1]]

            cost = np.hstack([cost] * ((num_agent - 1) // nc + 1))

            self.global_goals = np.zeros((num_agent, 2), dtype=np.int32)
            row_ind, col_ind = linear_sum_assignment(cost)
            for i in range(num_agent):
                cluster_idx = col_ind[i] % nc
                agent_idx = row_ind[i]
                frontier_dist = np_obstacle_map_distance[agent_idx][is_frontier]
                frontier_dist[nearest_clusters != cluster_idx] = np.inf
                select = np.argmin(frontier_dist)
                self.global_goals[agent_idx] = [frontier_idx[0][select], frontier_idx[1][select]]

    def get_long_term_goal(self, np_obstacle_map, np_frontier_map, agent_pos, idx):
        if idx == 0:
            self.get_coscan_goals(np_obstacle_map, np_frontier_map, agent_pos)
        return self.global_goals[idx]

    def replan(self, np_obstacle_map, np_frontier_map, goal, planning_window):
        gx1, gx2, gy1, gy2 = planning_window
        np_obstacle_map = np.ma.masked_values(np_obstacle_map, 1)
        np_obstacle_map[goal[0], goal[1]] = 1
        dd = skfmm.distance(1 - np_obstacle_map)[gx1:gx2, gy1:gy2]
        dd[np_frontier_map[gx1:gx2, gy1:gy2] == 0] = np.inf
        goal = np.unravel_index(np.argmin(dd), dd.shape)
        return goal[0], goal[1]

    def check_finish(self, np_frontier_map, stop):
        if len([1 for goal in self.global_goals if goal is not None]) == 1:
            self.global_goals = None
            return True
        for goal, stopped in zip(self.global_goals, stop):
            if goal is None:
                continue
            if stopped:
                stopped = False
                self.global_goals = None
                return True
            elif np_frontier_map[goal[0], goal[1]] == 0:
                self.global_goals = None
                return True
        return False
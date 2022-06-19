import numpy as np

if __name__ == '__main__':
    import depth_utils as du
else:
    import env.utils.depth_utils as du


def cross(window1, window2):
    x1, x2, y1, y2 = window1
    mx1, mx2, my1, my2 = window2
    sx1 = max(0, min(x2 - x1, mx1 - x1))
    sx2 = max(0, min(x2 - x1, mx2 - x1))
    sy1 = max(0, min(y2 - y1, my1 - y1))
    sy2 = max(0, min(y2 - y1, my2 - y1))
    dx1 = max(0, min(mx2 - mx1, x1 - mx1))
    dx2 = max(0, min(mx2 - mx1, x2 - mx1))
    dy1 = max(0, min(my2 - my1, y1 - my1))
    dy2 = max(0, min(my2 - my1, y2 - my1))
    return sx1, sx2, sy1, sy2, dx1, dx2, dy1, dy2


class MapBuilder(object):
    def __init__(self, params):
        frame_width = params['frame_width']
        frame_height = params['frame_height']
        self.fov = params['fov']
        self.camera_matrix = du.get_camera_matrix(
            frame_width,
            frame_height,
            self.fov)
        self.vision_range = params['vision_range']

        self.depth_obs = params['depth_obs']
        self.map_size_cm = params['map_size_cm']
        self.mask_size = params['mask_size']
        self.unit_size_cm = params['unit_size_cm']
        self.du_scale = params['du_scale']
        self.obs_threshold = params['obs_threshold']
        self.num_robots = params['num_robots']

        self.agent_height = params['agent_height']
        self.z_bins = [-20, 5, self.agent_height + 8]
        self.map = np.zeros((self.map_size_cm // self.unit_size_cm,
                             self.map_size_cm // self.unit_size_cm,
                             len(self.z_bins) + 1), dtype=np.float32)
        self.results = np.zeros((2,
                                 self.map_size_cm // self.unit_size_cm,
                                 self.map_size_cm // self.unit_size_cm), dtype=np.float32)
        self.local_obstacle = np.zeros((self.num_robots, self.mask_size, self.mask_size), np.bool)
        self.local_windows = [None] * self.num_robots
        return

    def update_map(self, depth, current_pose):
        with np.errstate(invalid="ignore"):
            # the unit of the depth is in centimeters 
            depth[depth > self.vision_range * self.unit_size_cm] = np.NaN
        
        na = self.num_robots
        geocentric_flat = []
        geocentric_fflat = []
        windows = []
        mask = np.zeros((self.map_size_cm // self.unit_size_cm, self.map_size_cm // self.unit_size_cm), dtype=np.bool)
        self.depth = [None] * na

        for a in range(na):
            x = int(current_pose[a][0] / self.unit_size_cm)
            y = int(current_pose[a][1] / self.unit_size_cm)
            x1, x2 = x - self.vision_range, x + self.vision_range + 1
            y1, y2 = y - self.vision_range, y + self.vision_range + 1
            x1 = max(0, min(self.map.shape[0], x1))
            x2 = max(0, min(self.map.shape[0], x2))
            y1 = max(0, min(self.map.shape[1], y1))
            y2 = max(0, min(self.map.shape[1], y2))
            windows.append((x1, x2, y1, y2, x, y))

            mx1, mx2 = x - self.mask_size // 2, x + self.mask_size // 2 + 1
            my1, my2 = y - self.mask_size // 2, y + self.mask_size // 2 + 1
            sx1, sx2, sy1, sy2, dx1, dx2, dy1, dy2 = cross((x1, x2, y1, y2), (mx1, mx2, my1, my2))
            mask[y1:y2, x1:x2][sy1:sy2, sx1:sx2] = True

            if self.local_windows[a] is not None:
                sx1, sx2, sy1, sy2, dx1, dx2, dy1, dy2 = cross(self.local_windows[a], (mx1, mx2, my1, my2))
                old_obstacle = np.copy(self.local_obstacle[a])
                self.local_obstacle[a, dy1:dy2, dx1:dx2] = old_obstacle[sy1:sy2, sx1:sx2]

            self.local_windows[a] = (mx1, mx2, my1, my2)

            point_cloud = du.get_point_cloud_from_z(depth[a], self.camera_matrix, scale=self.du_scale)

            agent_view = du.transform_camera_view(point_cloud, self.agent_height, current_pose[a][3])

            geocentric_pc = du.transform_pose(agent_view, current_pose[a])

            if self.depth_obs:
                gpcz = geocentric_pc[:, :, 2]
                mask_a = gpcz < -20
                mask_b = gpcz < 5
                mask_c = gpcz < self.agent_height + 8
                self.depth[a] = np.ones(gpcz.shape, dtype=np.uint8) * 255
                self.depth[a][mask_c] = 170
                self.depth[a][mask_b] = 85
                self.depth[a][mask_a] = 0
                self.depth[a] = self.depth[a].reshape(*self.depth[a].shape, 1)
                

            geocentric_flat.append(du.bin_points(
                geocentric_pc,
                (x1, x2, y1, y2),
                self.z_bins,
                self.unit_size_cm))

        if self.depth_obs:
            self.depth = np.repeat(np.stack(self.depth), 3, axis=3)


        for a, flat, window in zip(range(na), geocentric_flat, windows):
            x1, x2, y1, y2, x, y = window

            submap = self.map[y1:y2, x1:x2, :]
            other_robots_area = np.logical_and((flat[:, :, 2] * (self.results[1, y1:y2, x1:x2] - self.results[0, y1:y2, x1:x2])) > 0, mask[y1:y2, x1:x2])
            flat[:, :, 2][other_robots_area] = 0
            fflat = ((flat[:, :, 2] == 0) & (flat[:, :, 0] == 0) & (flat[:, :, 1] > 0)) * 1
            if fflat is not None:
                submap[:, :, :][fflat > 0] = 0.0
            for oa in range(na):
                if oa == a:
                    continue
                sx1, sx2, sy1, sy2, dx1, dx2, dy1, dy2 = cross((x1, x2, y1, y2), self.local_windows[oa])
                self.local_obstacle[oa, dy1:dy2, dx1:dx2] = other_robots_area[sy1:sy2, sx1:sx2]

        for flat, window in zip(geocentric_flat, windows):
            x1, x2, y1, y2, x, y = window
            submap = self.map[y1:y2, x1:x2, :]
            submap += flat
            
            estimated_obstacle = self.results[0, y1:y2, x1:x2]
            estimated_explored = self.results[1, y1:y2, x1:x2]
            
            # This means the obstacle is defined as the area that is both below -20 cm and between [5, self.agent_height + 8] cm.
            estimated_obstacle[:, :] = (submap[:, :, 2] + submap[:, :, 0]) / self.obs_threshold
            estimated_obstacle[estimated_obstacle >= 0.5] = 1.0
            estimated_obstacle[estimated_obstacle < 0.5] = 0.0
            
            estimated_explored[:, :] = submap[:, :, 0:3].sum(2)
            estimated_explored[estimated_explored > 0] = 1.0

    def get_st_pose(self, current_loc):
        loc = [- (current_loc[0] / self.unit_size_cm
                  - self.map_size_cm // (self.unit_size_cm * 2)) / \
               (self.map_size_cm // (self.unit_size_cm * 2)),
               - (current_loc[1] / self.unit_size_cm
                  - self.map_size_cm // (self.unit_size_cm * 2)) / \
               (self.map_size_cm // (self.unit_size_cm * 2)),
               90 - np.rad2deg(current_loc[2])]
        return loc

    def reset(self, init_pose):
        self.map[...] = 0.
        self.results[...] = 0.
        self.local_obstacle[...] = False
        self.local_windows = [None] * self.num_robots

        na = self.num_robots
        buf = self.mask_size // 2
        for a in range(na):
            x = int(init_pose[a][0] / self.unit_size_cm)
            y = int(init_pose[a][1] / self.unit_size_cm)
            self.map[y-buf:y+buf, x-buf:x+buf, 1] = 1
            self.results[1, y-buf:y+buf, x-buf:x+buf] = 1

    def get(self):
        return self.results[0], self.results[1]

    def mask_local(self, map_to_set, map_window, idx, value=1):
        '''map_to_set = origin_map_to_set[map_window]
        '''
        map_to_set = np.copy(map_to_set)
        for i in range(self.num_robots):
            if i == idx or self.local_windows[i] is None:
                continue
            sx1, sx2, sy1, sy2, dx1, dx2, dy1, dy2 = cross(self.local_windows[i], map_window)
            map_to_set[dy1:dy2, dx1:dx2][self.local_obstacle[i, sy1:sy2, sx1:sx2]] = value
        return map_to_set

    def imgize(self, g):
        print('-----')
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                if g[i, j]:
                    print('*', end='')
                else:
                    print('.', end='')
            print('')
        print('-----')

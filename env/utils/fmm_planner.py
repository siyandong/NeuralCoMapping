import cv2
import numpy as np
import skfmm
from numpy import ma
import math



def get_mask(sx, sy, scale, step_size, ori):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size))
    rmask = np.zeros((size, size))
    sx = (size // 2 + sx)
    sy = (size // 2 + sy)
    ori = ori % 360.0
    if ori > 180:
        ori -= 360
    scale = 10.
    for i in range(size):
        for j in range(size):
            dx = (i + 0.5) - sx
            dy = (j + 0.5) - sy
            if dx ** 2 + dy ** 2 <= step_size ** 2:
                mask[i, j] = 1
                dori = (ori - math.degrees(math.atan2(dx, dy))) % 360.0
                dori = 1.0 - abs(dori / 180.0 - 1.0)
                rmask[i, j] = dori * scale
    rmask[size // 2, size // 2] = scale
    return mask, rmask


def get_dist(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size)) + 1e-10
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
                    step_size ** 2:
                mask[i, j] = max(5, (((i + 0.5) - (size // 2 + sx)) ** 2 +
                                     ((j + 0.5) - (size // 2 + sy)) ** 2) ** 0.5)
    return mask


class FMMPlanner():
    def __init__(self, traversible, scale=1, step_size=3, use_distance_field=True):
        self.scale = scale
        self.step_size = step_size
        if scale != 1.:
            self.traversible = cv2.resize(traversible,
                                          (traversible.shape[1] // scale, traversible.shape[0] // scale),
                                          interpolation=cv2.INTER_NEAREST)
            self.traversible = np.rint(self.traversible)
        else:
            self.traversible = traversible

        self.du = int(self.step_size / (self.scale * 1.))
        self.use_distance_field = use_distance_field

    def set_goal(self, goal):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        x1 = int((goal[0] - goal[2]) / (self.scale * 1.))
        x2 = int((goal[0] + goal[2]) / (self.scale * 1.))
        y1 = int((goal[1] - goal[2]) / (self.scale * 1.))
        y2 = int((goal[1] + goal[2]) / (self.scale * 1.))
        if y1 == y2 or x1 == x2:
            traversible_ma[y1, x1] = 0
        else:
            traversible_ma[y1:y2, x1:x2] *= 0
            traversible_ma[int(goal[1] / self.scale), int(goal[0] / self.scale)] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd
        return dd_mask

    def nearest(self, targets, state):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        traversible_ma[state[0], state[1]] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, 1e10) * targets
        dd[dd == 0] = 1e10
        h, w = dd.shape
        idx = np.argmin(dd.reshape(-1))
        return idx // w, idx % w

    def reachable(self, state, goals):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        traversible_ma[0, 0] = 0
        traversible_ma[0, -1] = 0
        traversible_ma[-1, 0] = 0
        traversible_ma[-1, -1] = 0
        traversible_ma *= 1 - goals.astype(np.int32)
        dd = skfmm.distance(traversible_ma, dx=1)
        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        return dd_mask[state[0], state[1]]


    def get_short_term_goal(self, state, ori):
        scale = self.scale * 1.
        state = [x / scale for x in state]
        dx, dy = state[0] - int(state[0]), state[1] - int(state[1])
        mask, rmask = get_mask(dx, dy, scale, self.step_size, ori)

        state = [int(x) for x in state]

        dist = np.pad(self.fmm_dist, self.du,
                      'constant', constant_values=self.fmm_dist.shape[0] ** 2)
        subset = dist[state[0]:state[0] + 2 * self.du + 1,
                 state[1]:state[1] + 2 * self.du + 1]

        assert subset.shape[0] == 2 * self.du + 1 and \
               subset.shape[1] == 2 * self.du + 1, \
            "Planning error: unexpected subset shape {}".format(subset.shape)

        subset = subset * mask
        subset += (1 - mask) * self.fmm_dist.shape[0] ** 2
        subset -= subset[self.du, self.du]

        trav = np.pad(self.traversible, 2 * self.du,
                      'constant', constant_values=0)

        subset_trav = trav[state[0] + self.du:state[0] + 3 * self.du + 1,
                      state[1] + self.du:state[1] + 3 * self.du + 1]
        traversible_ma = ma.masked_values(subset_trav * 1, 0)
        goal_x, goal_y = self.du, self.du
        traversible_ma[goal_y, goal_x] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        dd = ma.filled(dd, np.max(dd) + 1)
        dd[dd < 1] = 1

        if subset_trav.min() == 0 and self.use_distance_field:
            transform = skfmm.distance(trav[state[0]:state[0] + 4 * self.du + 1,
                      state[1]:state[1] + 4 * self.du + 1], dx=1)[self.du:-self.du, self.du:-self.du] * 0.25
            transform[transform > 1] = 1.

            subset_fmm_dist = (dd + rmask) / (1e-3 + transform)
        else:
            subset_fmm_dist = (dd + rmask)

        subset = subset / subset_fmm_dist
        subset[subset < -1.5] = 1
        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)

        if subset[stg_x, stg_y] > -0.0001:
            replan = True
        else:
            replan = False
        return (stg_x + state[0] - self.du) * scale + 0.5, \
               (stg_y + state[1] - self.du) * scale + 0.5, replan

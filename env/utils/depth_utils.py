# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for processing depth images.
"""
from argparse import Namespace
from numba import njit
import numpy as np

from env.utils.rotation_utils import get_r_matrix


def get_camera_matrix(width, height, fov):
    """Returns a camera matrix from image size and fov.
    It defines the camera intrinsics:
    xc and yc are the optical center in x and y dimension,
    f is the focal length is x and y dimension.
    """
    xc = (width - 1.) / 2.
    zc = (height - 1.) / 2.
    f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))
    camera_matrix = {'xc': xc, 'zc': zc, 'f': f}
    camera_matrix = Namespace(**camera_matrix)
    return camera_matrix


def get_point_cloud_from_z(Y, camera_matrix, scale=1):
    # camera coordinate system is [0, -1, 0]
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    x, z = np.meshgrid(np.arange(Y.shape[-1]),
                       np.arange(Y.shape[-2] - 1, -1, -1))
    for i in range(Y.ndim - 2):
        x = np.expand_dims(x, axis=0)
        z = np.expand_dims(z, axis=0)
    X = (x[::scale, ::scale] - camera_matrix.xc) * Y[::scale, ::scale] / camera_matrix.f
    Z = (z[::scale, ::scale] - camera_matrix.zc) * Y[::scale, ::scale] / camera_matrix.f
    XYZ = np.concatenate((X[..., np.newaxis], Y[::scale, ::scale][..., np.newaxis],
                          Z[..., np.newaxis]), axis=X.ndim)
    return XYZ


# roll, pitch and yaw follows the y, x and z axis in gibson
def transform_camera_view(XYZ, sensor_height, camera_elevation_rad):
    """
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ                     : ...x3
        sensor_height           : height of the sensor
        camera_elevation_rad : camera elevation to rectify (pitch).
    Output:
        XYZ : ...x3
    """
    R = get_r_matrix([1., 0., 0.], angle=-camera_elevation_rad)
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    return XYZ


def transform_pose(XYZ, current_pose):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : camera position (x, y, theta (radians, yaw)) 
    Output:
        XYZ : ...x3
    """
    R = get_r_matrix([0., 0., 1.], angle=current_pose[2] - np.pi / 2.)
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[:, :, 0] = XYZ[:, :, 0] + current_pose[0]
    XYZ[:, :, 1] = XYZ[:, :, 1] + current_pose[1]
    return XYZ


def bin_points(XYZ_cm, window, z_bins, xy_resolution):
    """Bins points into xy-z bins
    XYZ_cm is H x W x 3
    Output is map_size x map_size x (len(z_bins)+1)
    """
    n_z_bins = len(z_bins) + 1
    isnotnan = np.logical_not(np.isnan(XYZ_cm[:, :, 0]))
    X_bin = np.round(XYZ_cm[:, :, 0] / xy_resolution).astype(np.int32) - window[0]
    Y_bin = np.round(XYZ_cm[:, :, 1] / xy_resolution).astype(np.int32) - window[2]
    X_size = window[1] - window[0]
    Y_size = window[3] - window[2]
    Z_bin = np.digitize(XYZ_cm[:, :, 2], bins=z_bins).astype(np.int32)

    isvalid = np.array([X_bin >= 0, X_bin < X_size, Y_bin >= 0, Y_bin < Y_size, Z_bin >= 0, Z_bin < n_z_bins, isnotnan])
    isvalid = np.all(isvalid, axis=0)

    ind = (Y_bin * X_size + X_bin) * n_z_bins + Z_bin
    ind[np.logical_not(isvalid)] = 0
    count = np.bincount(ind.ravel(), isvalid.ravel().astype(np.int32), minlength=X_size * Y_size * n_z_bins)

    return np.reshape(count, (Y_size, X_size, n_z_bins))

def get_free_points(ocp, xy_resolution):
    pts = []
    Y_base = ocp[:, :, 1] / xy_resolution
    Y_base[np.isnan(ocp[:, :, 0])] = 0.
    XYZ_base = np.stack((ocp[:, :, 0] / ocp[:, :, 1], np.ones(ocp.shape[:-1]), ocp[:, :, 2] / ocp[:, :, 1] * xy_resolution), 2)
    for y in range(int(Y_base.max())):
        mask = Y_base > y
        XYZ = XYZ_base[mask] * (y * xy_resolution)
        pts.append(XYZ)
    if pts:
        return np.vstack(pts).reshape(-1, 1, 3)
    return None

@njit
def bresenham_safe(grid, x0, y0, x1, y1, value_to_fill):
    dx = x1 - x0
    dy = y1 - y0
    step_x = 1
    step_y = 1
    swap_flag = 0
    if dx < 0:
        dx = -dx
        step_x = -1
    if dy < 0:
        dy = -dy
        step_y = -1
    if dy > dx:
        dx, dy = dy, dx
        swap_flag = ~0
    f = (dy * 2) - dx
    for i in range(dx + 1):
        if 0 <= x0 < grid.shape[0] and 0 <= y0 < grid.shape[1]:
            grid[x0, y0] = value_to_fill
        if f >= 0:
            x0 += step_x & swap_flag
            y0 += step_y & ~swap_flag
            f -= dx * 2
        f += dy * 2
        x0 += step_x & ~swap_flag
        y0 += step_y & swap_flag

def get_free_area(obstacle, x, y):
    free = np.zeros_like(obstacle)
    obstacle = obstacle > 0
    xs, ys = np.where(obstacle)
    for ox, oy in zip(xs, ys):
        bresenham_safe(free, ox, oy, x, y, 1)
    free -= obstacle
    return free
    
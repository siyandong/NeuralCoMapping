import os
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import sys
from os.path import join
from skimage import measure
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import get_config


def color_coding(x):
    r, g, b, a = plt.cm.jet(int(round(x * 255)))
    r = int(round(r * 255))
    g = int(round(g * 255))
    b = int(round(b * 255))
    return b, g, r



def connected_field(trav):
    trav = trav.astype(np.float32) / 255
    trav_connected = np.zeros_like(trav)
    img_label = measure.label(trav, background=0, connectivity=1).astype(np.float32)
    # #### visualization
    # for i in np.unique(img_label):
    #     img_label[img_label == i] = np.linalg.norm(tuple(color_coding(i / img_label.max())))
    # return img_label
    # ###
    label_bincount = np.bincount(img_label.astype(np.int).flatten())
    label_bincount[np.argmax(label_bincount)] = -1
    maxid = np.argmax(label_bincount)
    trav_connected[img_label == maxid] = 1
    trav_connected = trav_connected.astype(np.uint8) * 255
    return trav_connected


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('target')
    parser.add_argument('--height', '-z', type=float, default=0.5, help='step size in centimeter')
    parser.add_argument('--dynamic', default='', help='dynamic scene, e.g. change3')
    args = parser.parse_args()

    config = get_config(args.target, args.dynamic)
    traversable = cv2.imread(join(config.mesh_dir, 'floor_trav_0_m.png'), -1)
    if traversable.ndim != 2:
        traversable = traversable[..., 0]
    trav_connected = connected_field(traversable)
    cv2.imwrite(join(config.mesh_dir, f'floor_trav_0_connected.png'), trav_connected)


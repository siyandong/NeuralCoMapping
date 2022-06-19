import os
from os.path import join
import numpy as np
import cv2
import sys


scene = 'r1-house'
trav_dir = f'/home/wsy/Data/ydyin/qihang/proactive/{scene}/mesh'
trav = cv2.imread(join(trav_dir, 'floor_trav_0_connected_empty.png'), -1)
trav = trav[..., 0]

area = (trav == 255).sum()
print(area / 1e4)

import os
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import sys
from os.path import join
from skimage import measure
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/ydyin/codes/reloc_rl')
from config import get_config
import utils

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('target')
    parser.add_argument('seq')
    parser.add_argument('--height', '-z', type=float, default=0.5, help='step size in centimeter')
    parser.add_argument('--dynamic', default='', help='dynamic scene, e.g. change3')
    args = parser.parse_args()
    config = get_config(args.target, args.dynamic)

    trav = cv2.imread(join(config.mesh_dir, f'floor_trav_0_connected.png'), 1)
    with open(join(config.mesh_dir, 'floor_trav_0.yaml'), 'r') as y:
        data = yaml.safe_load(y)
    origin = data['origin']
    assert origin[0] == origin[1], "[floorplan yaml] origin[0] != origin[1]"
    trav_ori = origin[0]

    f_seq = open(join(config.seq_dir, f'blcam_{args.seq}.txt'), 'r')

    f_blcam = f_seq.readline()
    while f_blcam:
        blcam = f_blcam.strip().split(' ')
        blcam = [float(p) for p in blcam]
        trav_xy = utils.blcam2travxy(trav_ori, blcam[0], blcam[1])
        line_len = 300
        # end_xy = utils.blcam2travxy(trav_ori, blcam[0] - line_len * np.cos(blcam[-1]), blcam[1] - line_len * np.sin(blcam[-1]))
        end_xy = (trav_xy[0] - line_len * np.cos(blcam[-1]), trav_xy[1] - line_len * np.sin(blcam[-1]))
        end_xy = (round(end_xy[0]), round(end_xy[1]))
        cv2.circle(trav, trav_xy[::-1], 2, (0, 0, 255), 2)
        cv2.line(trav, trav_xy[::-1], end_xy[::-1], (0, 0, 255), 1)
        trav[trav_xy] = (0, 0, 255)
        f_blcam = f_seq.readline()

    cv2.imwrite('/home/wsy/trav.png', trav)
    f_seq.close()
import os
import yaml
import cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import get_config
from os.path import join

def get_img_coord(yaml_path, bl_x, bl_y):
    with open(yaml_path, 'r') as y:
        data = yaml.safe_load(y)

    origin = data['origin']
    assert origin[0] == origin[1], "floor.png origin[0] != origin[1]"
    ori = origin[0]
    img_coord = (int(round((-bl_y - ori) * 100)), int(round((bl_x - ori) * 100)))
    return img_coord


def crop_center(img, center, len_x, len_y=None):
    """ All in pixel unit"""
    rint = lambda x: int(round(x))
    if len_y is None:
        len_y = len_x
    center_x, center_y = center
    crop = img[rint(center_x - len_x/2): rint(center_x + len_x/2), rint(center_y - len_y/2): rint(center_y + len_y/2)]
    return crop


if __name__ == '__main__':
    target = sys.argv[-1]
    config = get_config(target)
    position = (2.673, -3.451)
    floorplan = cv2.imread(join(config.mesh_dir, 'floor_trav_0_connected.png'))
    yaml_path = join(config.mesh_dir, 'floor_trav_0.yaml')
    curr = get_img_coord(yaml_path, *position)
    floorplan[curr] = (0, 0, 255)
    cv2.imshow('aa', floorplan)
    cv2.imshow('bb', crop_center(floorplan, curr, 200))
    cv2.waitKey()

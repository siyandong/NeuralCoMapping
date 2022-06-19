import os
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import sys
from os.path import join
from skimage import measure

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config


def load_obj_np(filename_obj, normalization=False, texture_size=4, load_texture=False,
                texture_wrapping='REPEAT', use_bilinear=True):
    """Load Wavefront .obj file into numpy array
    This function only supports vertices (v x x x) and faces (f x x x).
    """
    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = np.vstack(vertices).astype(np.float32)

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = np.vstack(faces).astype(np.int32) - 1

    # load textures
    textures = None

    assert load_texture is False  # Since I commented out the block below
    # if load_texture:
    #     for line in lines:
    #         if line.startswith('mtllib'):
    #             filename_mtl = os.path.join(os.path.dirname(filename_obj), line.split()[1])
    #             textures = load_textures(filename_obj, filename_mtl, texture_size,
    #                                      texture_wrapping=texture_wrapping,
    #                                      use_bilinear=use_bilinear)
    #     if textures is None:
    #         raise Exception('Failed to load textures.')
    #     textures = textures.cpu().numpy()

    assert normalization is False  # Since I commented out the block below
    # # normalize into a unit cube centered zero
    # if normalization:
    #     vertices -= vertices.min(0)[0][None, :]
    #     vertices /= torch.abs(vertices).max()
    #     vertices *= 2
    #     vertices -= vertices.max(0)[0][None, :] / 2

    if load_texture:
        return vertices, faces, textures
    else:
        return vertices, faces


def get_xy_floors(vertices, faces, dist_threshold=-0.98):
    z_faces = []
    z = np.array([0, 0, 1])
    faces_selected = []
    for face in tqdm(faces):
        normal = np.cross(vertices[face[2]] - vertices[face[1]], vertices[face[1]] - vertices[face[0]])
        dist = np.dot(normal, z) / np.linalg.norm(normal)
        if (dist_threshold is None) or ((dist_threshold is not None) and (dist < dist_threshold)):
            z_faces.append(vertices[face[0]][2])
            faces_selected.append(face)

    return np.array(z_faces), vertices, faces_selected


INTERSECT_EDGE = 0
INTERSECT_VERTEX = 1


class Plane(object):
    def __init__(self, orig, normal):
        self.orig = orig
        self.n = normal / np.linalg.norm(normal)

    def __str__(self):
        return 'plane(o=%s, n=%s)' % (self.orig, self.n)


def point_to_plane_dist(p, plane):
    return np.dot((p - plane.orig), plane.n)


def compute_triangle_plane_intersections(vertices, faces, tid, plane, dists, dist_tol=1e-8):
    """
    Compute the intersection between a triangle and a plane
    Returns a list of intersections in the form
        (INTERSECT_EDGE, <intersection point>, <edge>) for edges intersection
        (INTERSECT_VERTEX, <intersection point>, <vertex index>) for vertices
    This return between 0 and 2 intersections :
    - 0 : the plane does not intersect the plane
    - 1 : one of the triangle's vertices lies on the plane (so it just
          "touches" the plane without really intersecting)
    - 2 : the plane slice the triangle in two parts (either vertex-edge,
          vertex-vertex or edge-edge)
    """

    # TODO: Use an edge intersection cache (we currently compute each edge
    # intersection twice : once for each tri)

    # This is to avoid registering the same vertex intersection twice
    # from two different edges
    vert_intersect = {vid: False for vid in faces[tid]}

    # Iterate through the edges, cutting the ones that intersect
    intersections = []
    for e in ((faces[tid][0], faces[tid][1]),
              (faces[tid][0], faces[tid][2]),
              (faces[tid][1], faces[tid][2])):
        v1 = vertices[e[0]]
        d1 = dists[e[0]]
        v2 = vertices[e[1]]
        d2 = dists[e[1]]

        if np.fabs(d1) < dist_tol:
            # Avoid creating the vertex intersection twice
            if not vert_intersect[e[0]]:
                # point on plane
                intersections.append((INTERSECT_VERTEX, v1, e[0]))
                vert_intersect[e[0]] = True
        if np.fabs(d2) < dist_tol:
            if not vert_intersect[e[1]]:
                # point on plane
                intersections.append((INTERSECT_VERTEX, v2, e[1]))
                vert_intersect[e[1]] = True

        # If vertices are on opposite sides of the plane, we have an edge
        # intersection
        if d1 * d2 < 0:
            # Due to numerical accuracy, we could have both a vertex intersect
            # and an edge intersect on the same vertex, which is impossible
            if not vert_intersect[e[0]] and not vert_intersect[e[1]]:
                # intersection factor (between 0 and 1)
                # here is a nice drawing :
                # https://ravehgonen.files.wordpress.com/2013/02/slide8.png
                # keep in mind d1, d2 are *signed* distances (=> d1 - d2)
                s = d1 / (d1 - d2)
                vdir = v2 - v1
                ipos = v1 + vdir * s
                intersections.append((INTERSECT_EDGE, ipos, e))

    return intersections


def gen_map(vertices, faces, output_folder=None, img_filename_format='floor_{}.png'):
    xmin, ymin, _ = vertices.min(axis=0)
    xmax, ymax, _ = vertices.max(axis=0)

    max_length = np.max([np.abs(xmin), np.abs(ymin), np.abs(xmax), np.abs(ymax)])
    max_length = np.ceil(max_length).astype(np.int)

    floors = [args.height]
    print(floors)

    floor_maps = []

    for i_floor, floor in enumerate(floors):
        dists = []
        z = float(floor)
        cross_section = []
        plane = Plane(np.array([0, 0, z]), np.array([0, 0, 1]))

        for v in vertices:
            dists.append(point_to_plane_dist(v, plane))

        for i in tqdm(range(len(faces))):
            res = compute_triangle_plane_intersections(vertices, faces,
                                                       i, plane, dists)
            if len(res) == 2:
                cross_section.append((res[0][1], res[1][1]))

        floor_map = np.ones((2 * max_length * 100, 2 * max_length * 100))

        for item in cross_section:
            x1, x2 = (item[0][0] + max_length) * 100, (item[1][0] + max_length) * 100
            y1, y2 = (item[0][1] + max_length) * 100, (item[1][1] + max_length) * 100

            cv2.line(floor_map, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 0), thickness=2)

        floor_maps.append(floor_map)
        # cur_img = Image.fromarray((floor_map * 255).astype(np.uint8))
        # cur_img = Image.fromarray(np.flipud(cur_img))
        # img_filename = img_filename_format.format(i_floor)
        # cur_img.save(os.path.join(output_folder, img_filename))

    return floor_maps


def gen_trav_map(vertices, faces, output_folder=None, add_clutter=False, filename_format='floor_trav_{}.png'):
    """Generate traversability maps.

    Args:
        mp3d_dir: Root directory of Matterport3D or Gibson. Under this root directory should be
                subdirectories, each of which represents a model/environment. Within each
                subdirectory should be a file named 'mesh_z_up.obj'.
        add_clutter: Boolean for whether to generate traversability maps with or without clutter.
    """
    floors = [0.1]

    z_faces, vertices, faces_selected = get_xy_floors(vertices, faces)
    z_faces_all, vertices_all, faces_selected_all = get_xy_floors(vertices, faces, dist_threshold=None)

    xmin, ymin, _ = vertices.min(axis=0)
    xmax, ymax, _ = vertices.max(axis=0)

    max_length = np.max([np.abs(xmin), np.abs(ymin), np.abs(xmax), np.abs(ymax)])
    max_length = np.ceil(max_length).astype(np.int)

    wall_maps = gen_map(vertices, faces)

    for i_floor in range(len(floors)):
        floor = floors[i_floor]
        mask = (np.abs(z_faces - floor) < 0.2)
        faces_new = np.array(faces_selected)[mask, :]

        t = (vertices[faces_new][:, :, :2] + max_length) * 100
        t = t.astype(np.int32)

        floor_map = np.zeros((2 * max_length * 100, 2 * max_length * 100))

        cv2.fillPoly(floor_map, t, 1)

        if add_clutter is True:  # Build clutter map
            mask1 = ((z_faces_all - floor) < 2.0) * ((z_faces_all - floor) > 0.05)
            faces_new1 = np.array(faces_selected_all)[mask1, :]

            t1 = (vertices_all[faces_new1][:, :, :2] + max_length) * 100
            t1 = t1.astype(np.int32)

            clutter_map = np.zeros((2 * max_length * 100, 2 * max_length * 100))
            cv2.fillPoly(clutter_map, t1, 1)
            floor_map = np.float32((clutter_map == 0) * (floor_map == 1))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        erosion = cv2.dilate(floor_map, kernel, iterations=2)
        erosion = cv2.erode(erosion, kernel, iterations=2)
        wall_map = wall_maps[i_floor]
        wall_map = cv2.erode(wall_map, kernel, iterations=1)
        erosion[wall_map == 0] = 0

        # cur_img = Image.fromarray((erosion * 255).astype(np.uint8))
        # cur_img = Image.fromarray(np.flipud(cur_img))
        # cur_img.save(os.path.join(output_folder, filename_format.format(i_floor)))
        return (erosion * 255).astype(np.uint8), (wall_maps[0] * 255).astype(np.uint8)


def write_yaml(save_dir, map_img, map_img_filepath, yaml_filename, resolution=0.01):  # NOTE: Copied from generate_map_yaml.py
    origin_px_coord = (map_img.shape[0] / 2, map_img.shape[1] / 2)  # (row, col)
    cur_origin_map_coord = (-float(origin_px_coord[1]) * resolution,
                            float(origin_px_coord[0] - map_img.shape[0]) * resolution,
                            0.0)  # (x, y, yaw)
    yaml_content = fill_template(map_img_filepath, resolution=resolution,
                                 origin=cur_origin_map_coord)

    cur_yaml_filepath = os.path.join(save_dir, yaml_filename)
    print('Writing to:', cur_yaml_filepath)
    with open(cur_yaml_filepath, 'w') as f:
        f.write(yaml_content)


def fill_template(map_filepath, resolution, origin):  # NOTE: Copied from generate_map_yaml.py
    """Return a string that contains the contents for the yaml file, filling out the blanks where
    appropriate.

    Args:
        map_filepath: Absolute path to map file (e.g. PNG).
        resolution: Resolution of each pixel in the map in meters.
        origin: Uhhh.
    """
    template = """image: MAP_FILEPATH
resolution: RESOLUTION
origin: [ORIGIN_X, ORIGIN_Y, YAW]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
    template = template.replace('MAP_FILEPATH', map_filepath)
    template = template.replace('RESOLUTION', str(resolution))
    template = template.replace('ORIGIN_X', str(origin[0]))
    template = template.replace('ORIGIN_Y', str(origin[1]))
    template = template.replace('YAW', str(origin[2]))
    return template


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('target')
    parser.add_argument('seq', help='seq name')
    parser.add_argument('--height', '-z', type=float, default=0.5, help='step size in centimeter')
    parser.add_argument('--dynamic', default='', help='dynamic scene, e.g. change3')
    args = parser.parse_args()
    seq_name = args.seq

    config = get_config(args.target, args.dynamic)
    v, f = load_obj_np(join(config.seq_dir, f'mesh_{seq_name}_lowres.obj'))
    traversable, wall_map = gen_trav_map(v, f)
    traversable = cv2.flip(traversable, 0)
    cv2.imwrite(join(config.seq_dir, f'floor_trav_{seq_name}_0.png'), traversable)
    write_yaml(config.seq_dir, traversable, f'floor_trav_{seq_name}_0.png', f'floor_trav_{seq_name}_0.yaml')
    # cv2.imwrite(join(config.mesh_dir, f'floor_0.png'), wall_map)

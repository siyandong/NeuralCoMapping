"""Generate floorplan, traversability map (floor only), and traversability map (with clutter).
"""

import time
from os.path import join

from generate_floor_maps import gen_map, generate_floorplan
from generate_traversable_map import gen_trav_map


def main():
    mesh_dir = '../iGibson/gibson2/data/g_dataset/Cross'
    mesh_name = 'mesh_z_up.obj'
    gen_map(join(mesh_dir, mesh_name), mesh_dir, img_filename_format='floor_{}.png')
    generate_floorplan(mesh_dir, mesh_name)
    gen_trav_map(mesh_dir, mesh_name, add_clutter=False)
    # gen_trav_map(mesh_dir, mesh_name, add_clutter=True)


if __name__ == '__main__':
    start_t = time.time()
    main()
    duration = time.time() - start_t
    print(f'Done with {duration / 3600:.2f} h')

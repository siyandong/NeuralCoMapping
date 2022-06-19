import sys

import matplotlib
import numpy as np

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import pickle


def visualize(fig, ax, img, grid, heatmap, poses, gt_poses, l_goals, vis_info, dump_dir, rank, ep_no, t, visualize, print_images, vis_style, resolution):

    agent_color = ['#E41316', '#FF8000', '#6A3A9B', '#B25823']

    for i, a in enumerate(ax):
        a.clear()
        a.set_yticks([])
        a.set_xticks([])
        a.set_yticklabels([])
        a.set_xticklabels([])
        if i < img.shape[0]:
            title = ['   (left)  ', '  (right)  ', ' (forward) ', ' (backward)', '  (stand)  '][int(vis_info['actions'][i])] + (' (bump)' if vis_info['bump'][i] else ' (free)')
            a.set_title(title)
            r, g, b = agent_color[i % 4][1:3], agent_color[i % 4][3:5], agent_color[i % 4][5:7]
            r, g, b = int('0x' + r, base=16), int('0x' + g, base=16), int('0x' + b, base=16)
            c = np.array([[[r, g, b]]], dtype=np.uint8)
            bordered_img = np.pad(img[i], ((4, 4), (4, 4), (0, 0)), 'constant')
            bordered_img[:4, :] = c
            bordered_img[-4:, :] = c
            bordered_img[:, :4] = c
            bordered_img[:, -4:] = c
            a.imshow(bordered_img)

    if vis_style == 1:
        title = "Reconstructing Map"
    else:
        title = "Ground-Truth Map"
    if vis_info['close']:
        title += " (close)"
    else:
        title += "  (open)"

    ax[-1].imshow(grid)
    ax[-1].set_title(title, family='sans-serif',
                    fontname='Helvetica',
                    fontsize=20)
    
    if heatmap is not None:
        print_images = True

    for idx, gt_pos, pos in zip(range(len(ax)), gt_poses, poses):
        # Draw GT agent pose
        agent_size = 6
    
        # Draw predicted agent pose
        x, y, o = pos
        x, y = x * 100.0 / resolution, grid.shape[1] - y * 100.0 / resolution

        dx = 0
        dy = 0
        fc = agent_color[idx % 4]
        dx = np.cos(np.deg2rad(o))
        dy = -np.sin(np.deg2rad(o))
        ax[-1].arrow(x - 1 * dx, y - 1 * dy, dx * agent_size, dy * agent_size * 1.25,
                    head_width=agent_size, head_length=agent_size * 1.25,
                    length_includes_head=True, fc=fc, ec=fc, alpha=0.7)


    if print_images:
        fn = '{}/episodes/{}/{}/{}-{}-Vis-{}.png'.format(
            dump_dir, (rank + 1), ep_no, rank, ep_no, t)
        plt.savefig(fn)


def insert_circle(mat, x, y, value):
    mat[x - 2: x + 3, y - 2:y + 3] = value
    mat[x - 3:x + 4, y - 1:y + 2] = value
    mat[x - 1:x + 2, y - 3:y + 4] = value
    return mat


def fill_color(colored, mat, color):
    mat = mat.astype(np.bool)
    for i in range(3):
        colored[:, :, 2 - i][mat] = 1 - color[i]
    return colored

def fill_color_by_range(colored, window, color):
    x1, x2, y1, y2 = window
    for i in range(3):
        colored[x1:x2, y1:y2, 2 - i] = 1 - color[i]
    return colored


def get_colored_map(obstacle, visited_gt, g_goals, l_goals,
                    explored, explorable, obstacle_gt, frontier):
    m, n = obstacle.shape
    colored = np.zeros((m, n, 3))
    pal = sns.color_palette("Paired")

    current_palette = [(0.9, 0.9, 0.9)]
    colored = fill_color(colored, explorable, current_palette[0])

    current_palette = [(235. / 255., 243. / 255., 1.)]
    colored = fill_color(colored, explored, current_palette[0])

    colored = fill_color(colored, obstacle, pal[2])

    for i in range(visited_gt.shape[0]):
        colored = fill_color(colored, visited_gt[i], pal[[5, 7, 9, 11][i % 4]])
    colored = fill_color(colored, frontier, pal[1])


    for idx, g_goal in enumerate(g_goals):
        g_goal_mat = 1
        colored = fill_color_by_range(colored, [g_goal[0]-2, g_goal[0]+3, g_goal[1]-2, g_goal[1]+3], pal[[5, 7, 9, 11][idx % 4]])

    colored = 1 - colored
    colored *= 255
    colored = colored.astype(np.uint8)
    return colored


def dump(
    curr_obstacle_map,
    explored_map,
    explorable_map,
    curr_frontier,
    obs,
    poses,
    trajs,
    goals,
    stgs,
    infos,
    dump_dir,
    unit_size_cm,
    rank,
    current_episode,
    timestep
):
    d = {
        'curr_obstacle_map' : curr_obstacle_map.astype(np.bool),
        'explored_map' : explored_map.astype(np.bool),
        'explorable_map' : explorable_map.astype(np.bool),
        'curr_frontier' : curr_frontier.astype(np.bool),
        'obs' : obs,
        'poses' : np.asarray(poses),
        'trajs' : trajs.astype(np.bool),
        'goals' : np.asarray(goals),
        'stgs': np.asarray(stgs),
        'infos' : infos,
        'unit_size_cm' : unit_size_cm,
        'current_scene_idx' : rank,
        'current_episode_idx' : current_episode,
        'timestep' : timestep
    }
    pth = '{}/episodes/{}/{}/{}-{}-Vis-{}.pkl'.format(dump_dir, (rank + 1), current_episode, rank, current_episode, timestep)
    with open(pth, 'wb') as f:
        pickle.dump(d, f)
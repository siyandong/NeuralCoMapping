import pickle
import cv2
import numpy as np
import math
import argparse
import tqdm
# import astar
import os
np.random.seed(0)


# params...
file_mask = '0-1-Vis-{}.pkl'
folder_in = None
folder_out = None
vis_skip_steps = 0
show_frontiers      = True
show_goals          = True
show_history_goals  = False
show_history_robots = False
flip_map            = True
flip_left_right     = False # True only for the scene of cvpr.

# RGB colors.
color_unknown = (255 /255., 255 /255., 255 /255.)
color_occupied = (30 /255., 30 /255., 30 /255.) # [55 /255., 143 /255., 38 /255.]
color_explored = (200 /255., 200 /255., 200 /255.) # [232 /255., 242 /255., 254 /255.]
color_frontier = (93 /255., 133 /255., 204 /255.) # [52 /255., 124 /255., 175 /255.]
color_robots = [
(240 /255., 44 /255., 19 /255.),    # red.
(255 /255., 131 /255., 0 /255.),    # orange.
(107 /255., 71 /255., 156 /255.),   # purple.
(64 /255., 255 /255., 64 /255.),    # green.
(255 /255., 255 /255., 128 /255.),  # yellow.
(64 /255., 64 /255., 255 /255.),    # blue.
(255 /255., 128 /255., 192 /255.),  # 
(128 /255., 64 /255., 0 /255.),     #
(128 /255., 128 /255., 64 /255.),   # 
# repeat
(240 /255., 44 /255., 19 /255.),    # red.
(255 /255., 131 /255., 0 /255.),    # orange.
(107 /255., 71 /255., 156 /255.),   # purple.
(64 /255., 255 /255., 64 /255.),    # green.
(255 /255., 255 /255., 128 /255.),  # yellow.
(64 /255., 64 /255., 255 /255.),    # blue.
(255 /255., 128 /255., 192 /255.),  # 
(128 /255., 64 /255., 0 /255.),     #
(128 /255., 128 /255., 64 /255.),   # 
(240 /255., 44 /255., 19 /255.),    # red.
(255 /255., 131 /255., 0 /255.),    # orange.
(107 /255., 71 /255., 156 /255.),   # purple.
(64 /255., 255 /255., 64 /255.),    # green.
(255 /255., 255 /255., 128 /255.),  # yellow.
(64 /255., 64 /255., 255 /255.),    # blue.
(255 /255., 128 /255., 192 /255.),  # 
(128 /255., 64 /255., 0 /255.),     #
(128 /255., 128 /255., 64 /255.),   # 
(240 /255., 44 /255., 19 /255.),    # red.
(255 /255., 131 /255., 0 /255.),    # orange.
(107 /255., 71 /255., 156 /255.),   # purple.
(64 /255., 255 /255., 64 /255.),    # green.
(255 /255., 255 /255., 128 /255.),  # yellow.
(64 /255., 64 /255., 255 /255.),    # blue.
(255 /255., 128 /255., 192 /255.),  # 
(128 /255., 64 /255., 0 /255.),     #
(128 /255., 128 /255., 64 /255.),   # 
(240 /255., 44 /255., 19 /255.),    # red.
(255 /255., 131 /255., 0 /255.),    # orange.
(107 /255., 71 /255., 156 /255.),   # purple.
(64 /255., 255 /255., 64 /255.),    # green.
(255 /255., 255 /255., 128 /255.),  # yellow.
(64 /255., 64 /255., 255 /255.),    # blue.
(255 /255., 128 /255., 192 /255.),  # 
(128 /255., 64 /255., 0 /255.),     #
(128 /255., 128 /255., 64 /255.),   # 
(240 /255., 44 /255., 19 /255.),    # red.
(255 /255., 131 /255., 0 /255.),    # orange.
(107 /255., 71 /255., 156 /255.),   # purple.
(64 /255., 255 /255., 64 /255.),    # green.
(255 /255., 255 /255., 128 /255.),  # yellow.
(64 /255., 64 /255., 255 /255.),    # blue.
(255 /255., 128 /255., 192 /255.),  # 
(128 /255., 64 /255., 0 /255.),     #
(128 /255., 128 /255., 64 /255.),   # 
]

grid_reso = 0.1     # fixed.
img_reso = 3000     # fixed.
n_rbt = 3           # fixed.
history_poses = []  # fixed.
history_goals = []  # fixed.
b_astar = False # True.


def load_dict(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def rot_degree(img, degree): # code from https://www.jianshu.com/p/04b7e7c36cfa.
    rows, cols = img.shape[0], img.shape[1]
    center = (cols / 2, rows / 2)
    mask = img.copy()
    mask[:, :] = 255
    M = cv2.getRotationMatrix2D(center, degree, 1)
    top_right = np.array((cols - 1, 0)) - np.array(center)
    bottom_right = np.array((cols - 1, rows - 1)) - np.array(center)
    top_right_after_rot = M[0:2, 0:2].dot(top_right)
    bottom_right_after_rot = M[0:2, 0:2].dot(bottom_right)
    new_width = max(int(abs(bottom_right_after_rot[0] * 2) + 0.5), int(abs(top_right_after_rot[0] * 2) + 0.5))
    new_height = max(int(abs(top_right_after_rot[1] * 2) + 0.5), int(abs(bottom_right_after_rot[1] * 2) + 0.5))
    offset_x = (new_width - cols) / 2
    offset_y = (new_height - rows) / 2
    M[0, 2] += offset_x
    M[1, 2] += offset_y
    dst = cv2.warpAffine(img, M, (new_width, new_height))
    mask = cv2.warpAffine(mask, M, (new_width, new_height))
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return dst

def crop_center_region(img, cen=None, d=500):
    rows, cols = img.shape[0], img.shape[1]
    cen_r, cen_c = int(cols/2), int(rows/2)
    if cen is not None:
        cen_r, cen_c = cen_r-cen[0], cen_c-cen[1]
    return img[ cen_r-d:cen_r+d, cen_c-d:cen_c+d ]

def draw_view_(img, se2, color=(0.,0.,0.), thickness=2):
    d = 40
    arrow = np.array([d, 0]).reshape((2,1))
    rot_mat = np.array([[math.cos(se2[2]), math.sin(se2[2])], [math.sin(se2[2]), -math.cos(se2[2])]])
    arrow = np.dot(rot_mat, arrow)
    # # draw arrow.
    # img = cv2.arrowedLine(img, (se2[0], se2[1]), (se2[0]+arrow[0], se2[1]+arrow[1]), color=color, thickness=thickness, line_type=8, shift=0, tipLength=0.6)
    # draw frustum.
    left_rot = 30/180*3.14
    left_rot_mat = np.array([[math.cos(left_rot), math.sin(left_rot)], [math.sin(left_rot), -math.cos(left_rot)]])
    left = np.dot(left_rot_mat, arrow)
    right_rot = -30/180*3.14
    right_rot_mat = np.array([[math.cos(right_rot), math.sin(right_rot)], [math.sin(right_rot), -math.cos(right_rot)]])
    right = np.dot(right_rot_mat, arrow)
    img = cv2.line(img, (se2[0], se2[1]), (se2[0]+int(left[0]), se2[1]-int(left[1])), color=color, thickness=thickness, lineType=cv2.LINE_AA)
    img = cv2.line(img, (se2[0], se2[1]), (se2[0]+int(right[0]), se2[1]-int(right[1])), color=color, thickness=thickness, lineType=cv2.LINE_AA)
    img = cv2.line(img, (se2[0]+int(left[0]), se2[1]-int(left[1])), (se2[0]+int(right[0]), se2[1]-int(right[1])), color=color, thickness=thickness, lineType=cv2.LINE_AA)
    for i in range(7):
        img = cv2.circle(img, (se2[0], se2[1]), radius=i, color=color, thickness=2, lineType=cv2.LINE_AA)

def get_lighter_color(color):
    c = []
    for i in range(len(color)): c.append( color[i] + (1-color[i])/2 )
    return c

'''
    status_dict = {
        'curr_obstacle_map' : curr_obstacle_map.astype(np.bool),
        'explored_map' : explored_map.astype(np.bool),
        'explorable_map' : explorable_map.astype(np.bool),
        'curr_frontier' : curr_frontier.astype(np.bool),
        'obs' : obs,
        'poses' : np.asarray(poses),
        'trajs' : trajs.astype(np.bool),
        'goals' : np.asarray(goals),
        'infos' : infos,
        'unit_size_cm' : unit_size_cm,
        'current_scene_idx' : rank,
        'current_episode_idx' : current_episode,
        'timestep' : timestep
    }
'''


def vis_status(rot, step=0):
    status_dict = load_dict('{}/{}'.format(folder_in, file_mask).format(step))

    ########
    # parse.
    ########
    poses = status_dict['poses']
    explored_map = status_dict['explored_map'].astype(np.float64)
    obstacle_map = status_dict['curr_obstacle_map'].astype(np.float64)
    explored_map = cv2.erode(cv2.dilate(explored_map, np.ones((3,3))), np.ones((5,5)))
    obstacle_map = cv2.erode(obstacle_map, np.ones((3,3))) # reduce obstacle thickness.
    frontier_map = status_dict['curr_frontier'].astype(np.float64)
    traj_map = status_dict['trajs'].astype(np.float64)
    ltgs = status_dict['goals']
    depths = status_dict['obs']
    depths[np.isnan(depths)] = 0.
    stgs = status_dict['stgs']
    explorable_map = status_dict['explorable_map'].astype(np.float64)

    # ########
    # # depth.
    # ########
    # depths = ( depths - depths.min() ) / (depths.max() - depths.min())
    # for rid in range(len(depths)):
    #     depth = depths[rid]
    #     if flip_map: depth = cv2.flip(depth, 1)
    #     cv2.imwrite('{}/depth_status{}_robot{}.png'.format(folder_out, step, rid), (depth*255).astype(int))
    # print('{}, return'.format(step))
    # return

    ########
    # map2d.
    ########
    map2d = np.zeros((480,480,3), dtype=np.float64)

    # status.
    map2d[:,:] = color_unknown # init to unknown.
    # map2d[explorable_map==1.] = get_lighter_color(get_lighter_color(color_frontier)) # gt open space.
    map2d[explored_map==1.] = color_explored # origin color.
    map2d[obstacle_map==1.] = color_occupied # origin color.
    # map2d[explored_map==1.] = get_lighter_color(color_explored) # lighter color.
    # map2d[obstacle_map==1.] = get_lighter_color(color_occupied) # lighter color.

    # frontiers.
    if show_frontiers:
        map2d[frontier_map==1.] = color_frontier # lighter color.
        # map2d[frontier_map==1.] = get_lighter_color(color_frontier) # lighter color.

    # long term goals.
    if show_goals:
        for gid in range(len(ltgs)):
            x, y = ltgs[gid][0], ltgs[gid][1]
            cv2.rectangle(map2d, (y-1,x-1), (y,x), color=color_robots[gid], thickness=1)
            cv2.rectangle(map2d, (y-1,x-1), (y+1,x+1), color=color_robots[gid], thickness=1)
            # cv2.rectangle(map2d, (y-1,x-1), (y,x), color=color_frontier, thickness=1)       # in blue.
            # cv2.rectangle(map2d, (y-1,x-1), (y+1,x+1), color=color_frontier, thickness=1)   # in blue.

    # last long term goals.
    if show_history_goals:
        for gid in range(len(ltgs)):
            history_goals[gid].append(ltgs[gid])
            if len(history_goals[0]) > 25:
                x, y = history_goals[gid][-25][0], history_goals[gid][-25][1]
                cv2.rectangle(map2d, (y-1,x-1), (y,x), color=color_robots[2], thickness=1)      # in purple.
                cv2.rectangle(map2d, (y-1,x-1), (y+1,x+1), color=color_robots[2], thickness=1)  # in purple.

    # planed paths.
    if b_astar:
        planned_paths = []
        for rid in range(n_rbt):
            maze = np.logical_or(~status_dict['explorable_map'], status_dict['curr_obstacle_map'])
            yb, xb = int(poses[rid][0]/grid_reso), int(poses[rid][1]/grid_reso)
            xe, ye = ltgs[rid][0], ltgs[rid][1]
            path = astar.solve(maze, (xe,ye), (xb,yb))
            planned_paths.append(path)

    # traj.
    for rid in range(n_rbt):
        map2d[traj_map[rid]==1.] = get_lighter_color(color_robots[rid])


    # resize.
    map2d = cv2.resize(map2d, (img_reso,img_reso), interpolation=cv2.INTER_NEAREST)

    # history poses.
    for rid in range(n_rbt):
        for sid in range(len(history_poses[rid])):
            pose = history_poses[rid][sid]
            y, x = int(pose[0]/grid_reso*img_reso/480), int(pose[1]/grid_reso*img_reso/480)
            #cv2.circle(map2d, (y,x), radius=2, color=get_lighter_color(color_robots[rid]), thickness=2, lineType=cv2.LINE_AA)
            for i in range(6):
                cv2.circle(map2d, (y,x), radius=i, color=get_lighter_color(color_robots[rid]), thickness=2, lineType=cv2.LINE_AA)
        history_poses[rid].append(poses[rid])


    if step < vis_skip_steps: 
        print('{}, return'.format(step))
        return

    # last robot poses.
    if show_history_robots:
        for rid in range(n_rbt):
            history_poses[rid].append(poses[rid])
            if len(history_poses[0]) > 25:
                # history traj.
                for pid in range(-25, -1):
                    pose = history_poses[rid][pid]
                    y, x, rad = int(pose[0]/grid_reso*img_reso/480), int(pose[1]/grid_reso*img_reso/480), pose[2]
                    cv2.circle(map2d, (y,x), radius=2, color=get_lighter_color(color_robots[rid]), thickness=2, lineType=cv2.LINE_AA) # in lighter orange.
                # the last poses.
                pose = history_poses[rid][-25]
                y, x, rad = int(pose[0]/grid_reso*img_reso/480), int(pose[1]/grid_reso*img_reso/480), pose[2]
                draw_view_(map2d, [y,x,rad], color=color_robots[rid], thickness=4) # in orange.
    
    # starting points.
    if 0:
        for rid in range(n_rbt):
            pose = history_poses[rid][0]
            y, x, rad = int(pose[0]/grid_reso*img_reso/480), int(pose[1]/grid_reso*img_reso/480), pose[2]
            # for i in range(8+1):
            #     map2d = cv2.circle(map2d, (y,x), radius=i, color=color_robots[rid], thickness=2, lineType=cv2.LINE_AA)
            draw_view_(map2d, [y,x,rad], color=color_robots[rid], thickness=4)

    # planed paths.
    if b_astar:
        for rid in range(n_rbt):
            for sid in range(len(planned_paths[rid])):
                pose = planned_paths[rid][sid]
                x, y = int(pose[0]*img_reso/480), int(pose[1]*img_reso/480)
                for i in range(6):
                    cv2.circle(map2d, (y,x), radius=i, color=get_lighter_color(color_robots[rid]), thickness=2, lineType=cv2.LINE_AA)

    # # short term goals.
    # for rid in range(len(stgs)):
    #     y0, x0 = int(poses[rid][0]/grid_reso*img_reso/480), int(poses[rid][1]/grid_reso*img_reso/480)
    #     x1, y1 = int(stgs[rid][0]*img_reso/480), int(stgs[rid][1]*img_reso/480)
    #     cv2.line(map2d, (y0,x0), (y1,x1), color=get_lighter_color(color_robots[rid]), thickness=4, lineType=cv2.LINE_AA)

    # current poses. 
    if 1:
        for rid in range(len(poses)):
            y, x, rad = int(poses[rid][0]/grid_reso*img_reso/480), int(poses[rid][1]/grid_reso*img_reso/480), poses[rid][2]
            draw_view_(map2d, [y,x,rad], color=color_robots[rid], thickness=4) # in different colors.
            #draw_view_(map2d, [y,x,rad], color=color_robots[0], thickness=4) # in red.

    # post process.
    map2d = rot_degree(map2d, rot)
    map2d = crop_center_region(map2d, cen=None, d=1100) # d=600.
    map2d = map2d[:,:,[2,1,0]]
    if not flip_map: map2d = cv2.flip(map2d, 0)
    if flip_left_right: map2d = cv2.flip(map2d, 1)
    print('status', step)
    # cv2.imshow('status', map2d)
    # cv2.waitKey(1)
    cv2.imwrite('{}/status{}.png'.format(folder_out, step), (map2d*255).astype(int))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--dir', type=str, default='temp')
    parser.add_argument('-ne', '--num-eps', type=int, default=1)
    parser.add_argument('-ns', '--num-stg', type=int, default=1)
    args = parser.parse_args()


def main(args):
    global folder_in, folder_out, file_mask, history_poses, history_goals
    folder_in = '{}/dump/{}/episodes/{}/{}'.format(args.dir, args.exp_name, args.stg, args.eps)
    folder_out = folder_in
    file_mask = '{}-{}-Vis-{}.pkl'.format(args.stg - 1, args.eps, '{}')
    history_poses = []
    history_goals = []
    n_rbt = len(load_dict('{}/{}'.format(folder_in, file_mask).format(0))['poses'])
    for rid in range(n_rbt): history_poses.append([])
    for rid in range(n_rbt): history_goals.append([])

    # #############
    # # test astar.
    # #############
    # status_dict = load_dict('{}/0-1-Vis-60.pkl'.format(folder_in))
    # map_obs = status_dict['curr_obstacle_map']
    # map_unk = ~status_dict['explored_map']
    # maze = np.logical_or(map_obs, map_unk).astype(int)
    # astar.solve(maze, (0,0), (30,30))


    try:
        i = 0
        while True:
            vis_status(rot=0, step=i)
            i += 10
    except FileNotFoundError as e:
        pass
    except Exception as e:
        raise e

if __name__ == '__main__':
    if ',' in args.exp_name:
        le, re = args.exp_name.split(',')
    else:
        le = args.exp_name
        re = None
    for stg in range(1, args.num_stg + 1):
        for eps in range(1, args.num_eps + 1):
            args.stg = stg
            args.eps = eps
            print(stg, eps)
            args.exp_name = le
            main(args)
            video_name = f'./video/{stg}-{eps}.avi'
            if re is not None:
                args.exp_name = re
                main(args)
                video = cv2.VideoWriter(video_name, fourcc=cv2.VideoWriter_fourcc(*"MP42"), apiPreference=0, fps=12.0, frameSize=(960, 480))
            else:
                video = cv2.VideoWriter(video_name, fourcc=cv2.VideoWriter_fourcc(*"MP42"), apiPreference=0, fps=12.0, frameSize=(640, 480))
            lframe = None
            rframe = None
            for k in range(0, int(1e10), 10):
                filename = f'{args.dir}/dump/{le}/episodes/{args.stg}/{args.eps}/status{k}.png'
                if re is not None:
                    filename2 = f'{args.dir}/dump/{re}/episodes/{args.stg}/{args.eps}/status{k}.png'
                else:
                    filename2 = '/null'
                if not os.path.exists(filename) and not os.path.exists(filename2):
                    break
                if os.path.exists(filename):
                    lframe = cv2.imread(filename)
                if os.path.exists(filename2):
                    rframe = cv2.imread(filename2)
                if re is not None:
                    frame = cv2.resize(np.hstack((lframe, rframe)), (960, 480))
                else:
                    frame = cv2.resize(lframe, (640, 480))
                video.write(frame)
            video.release()
    

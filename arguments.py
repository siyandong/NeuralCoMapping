import argparse
import math
import torch
import os


def get_parser():
    parser = argparse.ArgumentParser(description='DeCoExplore')

    ## General Arguments
    parser.add_argument('--scenes_file', type=str, required=True,
                        help='a file including names of scenes')
    parser.add_argument('--config_file', type=str, default="basic.yaml")
    parser.add_argument('--num_robots', type=int, default=3,
                        help='the number of robots (default: 3)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--restore_eps', type=int, default=0,
                        help='how many episodes to skip (default: 0)')
    parser.add_argument('--robot_scale', type=float, default=1,
                        help='scale of robots\' size')
    parser.add_argument('--baseline', type=str, default='none', choices=['none', 'coscan', 'seg', 'mtsp', 'grd'],
                        help='specify the baseline method (default: none for the RL method)')
    parser.add_argument('--use_single_gpu', type=int, default=0,
                        help='specify a single GPU for CUDA when not setting GIBSON_DEVICE_ID or CUDA_VISIBLE_DEVICES (default: 0)')
    parser.add_argument('-np', '--num_processes', type=int, default=0,
                        help='specify number of processes if > 0, otherwise the number of processes is equal to number of scenes (default: 0)')
    parser.add_argument('--eval_eps_freq', type=int, default=0,
                        help='(not recommended to change)')
    parser.add_argument('--num_episodes', type=int, default=10000,
                        help='number of episodes (default: 10000)')
    parser.add_argument('--train_global', type=int, default=0,
                        help='''0: Do not train the Global Policy
                                1: Train the Global Policy (default: 0)''')

    # Logging, loading models, visualization
    parser.add_argument('--log_interval', type=int, default=250,
                        help='log interval (in steps) (default: 250)')
    parser.add_argument('--save_interval', type=int, default=10000,
                        help='interval (in steps) of saving best models (default: 10000)')
    parser.add_argument('-d', '--dump_location', type=str, default="./temp",
                        help='path to dump models and log (default: ./temp)')
    parser.add_argument('--snp_location', type=str, default="./snp",
                        help='path to dump shared memory files (default: ./snp)')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='experiment name (required)')
    parser.add_argument('--save_periodic', type=int, default=500000,
                        help='model save frequency (in steps) (default: 500000)')
    parser.add_argument('--load_global_critic', type=str, default="0",
                        help='''model path to load,
                                0 to not reload (default: 0)''')
    parser.add_argument('--load_global', type=str, default="0",
                        help='''model path to load,
                                0 to not reload (default: 0)''')
    parser.add_argument('--vis_type', type=int, default=0,
                        help='''0: No visualization
                                1: Show local map
                                2: Dump visualization info
                                3: Show global map (default: 0)''')

    # Environment, dataset and episode specifications
    parser.add_argument('-efw', '--env_frame_width', type=int, default=192,
                        help='Frame width (default:192)')
    parser.add_argument('-efh', '--env_frame_height', type=int, default=192,
                        help='Frame height (default:192)')
    parser.add_argument('-fw', '--frame_width', type=int, default=192,
                        help='Frame width (default:192)')
    parser.add_argument('-fh', '--frame_height', type=int, default=192,
                        help='Frame height (default:192)')
    parser.add_argument('-na', '--noisy_actions', type=int, default=0,
                        help='(not recommended to change)')
    parser.add_argument('-no', '--noisy_odometry', type=int, default=0,
                        help='(not recommended to change)')
    parser.add_argument('--hfov', type=float, default=90.0,
                        help="horizontal field of view in degrees (default: 90)")
    parser.add_argument('--texture_randomization_freq', type=float, default=0.0,
                        help='(not recommended to change)')
    parser.add_argument('--object_randomization_freq', type=float, default=0.0,
                        help='(not recommended to change)')
    parser.add_argument('--reset_orientation', type=int, default=1,
                        help='''0: not to randomize orientation of scenes,
                                1: randomize orientation for each episode (default: 1)''')
    parser.add_argument('--reset_floor', type=int, default=0,
                        help='''0: not to randomize the floor index of scenes,
                                1: randomize the floor index for each episode (default: 0)''')
    parser.add_argument('--depth_noise_rate', type=float, default=0,
                        help='(not recommended to change)')

    ## Global Policy RL PPO Hyperparameters
    parser.add_argument('--centralized', type=int, default=1,
                        help='(not recommended to change)')
    parser.add_argument('--num_gnn_layer', type=int, default=3,
                        help='number of layers in GNN self/cross-attention module (default: 3)')
    parser.add_argument('--ablation', type=int, default=0,
                        help='(not recommended to change)')
    parser.add_argument('--use_history', type=int, default=1,
                        help='''0: disable history information in GNN,
                                1: enable history information in GNN (default: 1)''')
    parser.add_argument('--max_batch_size', type=int, default=0,
                        help='(not recommended to change)')
    parser.add_argument('--rotation_augmentation', type=int, default=1,
                        help='(not recommended to change)')
    parser.add_argument('--g_input_content', choices=['pos', 'gaussian_dist_map', 'distance_field'], default='gaussian_dist_map',
                        help='(not recommended to change)')
    parser.add_argument('--critic_lr_coef', type=float, default=5e-2,
                        help='relative learning rate of critic (relative to global_lr) (default: 5%)')
    parser.add_argument('--global_lr', type=float, default=5e-4,
                        help='global learning rate (default: 5e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RL Optimizer epsilon (default: 1e-5)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use_gae', type=int, default=1,
                        help='''0: disable generalized advantage estimation,
                                1: enable generalized advantage estimation (default: 1)''')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--reward_scale', type=float, default=5e-3,
                        help='scale of reward')
    parser.add_argument('--reward_bias', type=float, default=4,
                        help='bias of reward punishment')
    parser.add_argument('--entropy_coef', type=float, default=1e-4,
                        help='entropy term coefficient (default: 1e-4)')
    parser.add_argument('--value_loss_coef', type=float, default=3.0,
                        help='value loss coefficient (default: 3.0)')
    parser.add_argument('--action_loss_coef', type=float, default=1.0,
                        help='action loss coefficient (default: 1.0)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--num_global_steps', type=int, default=120,
                        help='number of global planning in one episode (default: 120)')
    parser.add_argument('--ppo_epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--ppo_sample_eps', type=int, default=1,
                        help='(not recommended to change)')
    parser.add_argument('--use_clipped_value_loss', type=int, default=1,
                        help='''0: disable value clipping,
                                1: enable value clipping (default: 1)''')
    parser.add_argument('--num_mini_batch', type=str, default="auto",
                        help='number of batches for ppo (default: auto)')
    parser.add_argument('--clip_param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')

    # Local Policy
    parser.add_argument('--num_local_steps', type=int, default=25,
                        help='number of steps between each global planning')
    parser.add_argument('--max_collisions_allowed', type=int, default=-1,
                        help='if > 0, specify the max number of collisions in one episode')
    parser.add_argument('--short_goal_dist', type=int, default=1,
                        help='maximum distance between the agent and the short term goal')

    parser.add_argument('--use_distance_field', type=int, default=1,
                        help='''0: disable distance field in FMM planner,
                                1: enable distance field (default: 1)''')
    parser.add_argument('--depth_obs', type=int, default=0,
                        help='''0: visualize RGB observation,
                                1: visualize depth map instead (default: 0)''')
    parser.add_argument('--z_offset', type=float, default=0.4,
                        help='(not recommended to change)')
    parser.add_argument('--global_downscaling', type=int, default=2,
                        help='specify downsampling scale from global map to local map (default: 2)')
    parser.add_argument('--vision_range', type=int, default=32,
                        help='vision range of robots (in grids) (default: 32)')
    parser.add_argument('--obstacle_boundary', type=int, default=20,
                        help='obstacle size (in grids) (default: 20)')
    parser.add_argument('--unit_size_cm', type=int, default=10,
                        help='size of each grids (in cm) (default: 10)')
    parser.add_argument('--du_scale', type=int, default=2,
                        help='specify downsampling scale at depth observation during map fusion (default: 2)')
    parser.add_argument('--map_size_cm', type=int, default=4800,
                        help='size of global map (in cm) (default: 4800)')
    parser.add_argument('--obs_threshold', type=float, default=1,
                        help='(not recommended to change)')
    parser.add_argument('--noise_level', type=float, default=0.1,
                        help='(not recommended to change)')
    return parser


def get_args():
    parser = get_parser()

    # parse arguments
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    args.print_images = args.vis_type > 0
    args.max_episode_length = args.num_local_steps * args.num_global_steps
    args.eval = 1 - args.train_global

    assert args.cuda
    assert args.short_goal_dist >= 1
    assert args.ppo_sample_eps == 1 or args.eval_eps_freq == 0 or args.ppo_sample_eps == args.eval_eps_freq

    total_num_scenes = len(open(args.scenes_file, 'r').readlines())
    if args.num_processes <= 0:
        args.num_processes = total_num_scenes
    assert total_num_scenes % args.num_processes == 0
    args.scene_per_process = total_num_scenes // args.num_processes


    if args.num_mini_batch == "auto":
        args.num_mini_batch = args.num_processes * max(1, args.ppo_sample_eps - (1 if args.eval_eps_freq else 0))
    else:
        args.num_mini_batch = int(args.num_mini_batch)

    return args

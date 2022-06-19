import os
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['mp3d', 'gibson'])
    parser.add_argument('--method', type=str, required=True, choices=['coscan', 'mtsp', 'grd', 'seg', 'rl'])
    parser.add_argument('-n', '--num-episodes', type=int, required=True)
    parser.add_argument('--load', default='./pretrained/hrl3.global', type=str)
    parser.add_argument('--dir', default='./std', type=str)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-e', '--echo', action='store_true')
    args = parser.parse_args()

    if args.dataset == 'mp3d':
        scenes = ['mp3dhq0-' + i for i in 'abcdefghi']
    elif args.dataset == 'gibson':
        scenes = ['hq-' + i for i in 'abcd']
    else:
        exit(1)

    if args.method != 'rl':
        raise NotImplementedError

    for idx, scene in enumerate(scenes):
        exp_name = 'eval_{}_{}'.format(args.method, scene.replace('-', ''))
        scenes_file = 'scenes/{}.scenes'.format(scene)
        method = '--baseline {}'.format(args.method) if args.method != 'rl' else '--use_history 1 --load_global {}'.format(args.load)
        redirect = '/dev/null' if not args.verbose else '/dev/stdout'
        cmd = f'python main.py --num_episodes {args.num_episodes} --exp_name "{exp_name}" --scenes_file {scenes_file} --dump_location {args.dir} {method} > {redirect} 2>&1'
        print(time.asctime(time.localtime(time.time())), f'{idx + 1}/{len(scenes)}', scene)
        if args.echo:
            print(cmd)
        else:
            assert 0 == os.system(cmd)

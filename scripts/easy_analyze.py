import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str)
    parser.add_argument('--dir', type=str, default='temp')
    parser.add_argument('--dataset', type=str, default='hq')
    parser.add_argument('--subset', type=str, default='abcd')
    args = parser.parse_args()

    result = []
    split = [[0, 30], [30, 60], [60, 1000]]
    mean_len = [[], [], []]
    unable_to_achieve_full_coverage = 0
    unable_to_explore_in_time = 0
    total_eps = 0

    for s in [args.dataset + '-' + i for i in args.subset]:
        args.exp_name = f'eval_{args.method}_{s.replace("-", "")}'
        args.scenes = f'scenes/{s}.scenes'
        with open(f'{args.dir}/dump/{args.exp_name}/explored_ratio.txt', 'r') as f:
            content = '[' + ''.join(f.readlines()).replace('\n', '').replace('][', '],[').replace('  ', ',') + ']'
            ratio = eval(content)
            ratio = np.array([x[-1] for x in ratio])
        with open(f'{args.dir}/dump/{args.exp_name}/explored_area.txt', 'r') as f:
            content = '[' + ''.join(f.readlines()).replace('\n', '').replace('][', '],[').replace('  ', ',') + ']'
            area = eval(content)
            area = np.array([x[-1] for x in area])
        with open(f'{args.dir}/dump/{args.exp_name}/close_episode_len.txt', 'r') as f:
            length = np.array([float(x.rstrip()) for x in f.readlines()])
        with open(args.scenes, 'r') as f:
            scenes = [x.rstrip() for x in f.readlines()]
        args.parallel = len(scenes)
        assert len(scenes) % args.parallel == 0
        scene_per_process = len(scenes) // args.parallel
        scenes = [scenes[i*scene_per_process:(i+1)*scene_per_process] for i in range(args.parallel)]
        ratio = ratio.reshape(args.parallel, -1)
        area = area.reshape(args.parallel, -1) / (ratio + 1e-3)
        length = length.reshape(args.parallel, -1)
        ratio_dict = {}
        area_dict = {}
        length_dict = {}
        valid_dict = {}
        for i in range(args.parallel):
            for j in range(ratio.shape[1]):
                scene_name = scenes[i][j % scene_per_process]
                if ratio_dict.get(scene_name) is None:
                    ratio_dict[scene_name] = []
                    area_dict[scene_name] = []
                    length_dict[scene_name] = []
                    valid_dict[scene_name] = []
                ratio_dict[scene_name].append(ratio[i, j])
                area_dict[scene_name].append(area[i, j])
                length_dict[scene_name].append(length[i, j])
                valid_dict[scene_name].append(ratio[i, j] > 0.7 and length[i, j] <= 2500)
        for d in [ratio_dict, area_dict, length_dict, valid_dict]:
            for k, v in d.items():
                d[k] = np.array(v)

        for i, j in ratio_dict.items():
            k = length_dict[i]
            a = area_dict[i]
            valid = np.logical_and(j > 0.7, k <= 2500)
            if not valid.any():
                valid[:] = True
            result.append('\t'.join([
                i + ' ' * (20 - len(i)),
                '{:.0f}'.format(a.mean()),
                '{}\t{}'.format(int(k[valid].mean()), int(k[valid].std())),
                '{:.1f}\t{:.1f}'.format(100 * j[valid].mean(), 100 * j[valid].std()),
                '{:.0f}'.format(valid_dict[i].mean() * 100)
            ]))
            unable_to_achieve_full_coverage += (j <= 0.9).sum()
            unable_to_explore_in_time += (k == 2999).sum()
            total_eps += np.prod(j.shape)
            if 0.9 <= valid_dict[i].mean() < 1:
                result[-1] += f'\t{np.argmin(valid_dict[i] * 1)}'
            for lst, r in zip(mean_len, split):
                if r[0] <= a.mean() < r[1]:
                    for v in k:
                        lst.append(v)

    print('\n'.join(result))
    print('unable_to_achieve_full_coverage: {:.1f}%'.format(100 * unable_to_achieve_full_coverage / total_eps))
    print('unable_to_explore_in_time: {:.1f}%'.format(100 * unable_to_explore_in_time / total_eps))
    for lst, r in zip(mean_len, split):
        if lst == []:
            continue
        print('mean length [{}~{}m2]: {:.1f}'.format(r[0], r[1], sum(lst) / len(lst)))
    

if __name__ == '__main__':
    main()

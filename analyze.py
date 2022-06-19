import argparse
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='./std')
    parser.add_argument('--dataset', type=str, required=True, choices=['mp3d', 'gibson'])
    parser.add_argument('-ne', '--num-eps', type=int, default=5)
    parser.add_argument('-ms', '--max-step', type=int, default=2999)
    parser.add_argument('--bins', type=str, default='30,60')
    args = parser.parse_args()

    result = {}
    split = [[lb, rb] for lb, rb in zip([0] + eval('[' + args.bins + ']'), eval('[' + args.bins + ']') + [args.max_step + 1])]
    dataset = ['mp3dhq0-' + i for i in 'abcdefghi'] if args.dataset == 'mp3d' else ['hq-' + i for i in 'abcd']

    for method in ['coscan', 'rl']:
        for s in dataset:
            path = os.path.join(args.dir, 'dump', f'eval_{method}_{s.replace("-", "")}')
            try:
                if not os.path.exists(path):
                    continue
                if method not in result:
                    result[method] = [], [], []
                with open(os.path.join(path, 'explored_ratio.txt'), 'r') as f:
                    content = '[' + ''.join(f.readlines()).replace('\n', '').replace('][', '],[').replace('  ', ',') + ']'
                    ratio = eval(content)
                    ratio = np.array([x[-1] for x in ratio])
                with open(os.path.join(path, 'explored_area.txt'), 'r') as f:
                    content = '[' + ''.join(f.readlines()).replace('\n', '').replace('][', '],[').replace('  ', ',') + ']'
                    area = eval(content)
                    area = np.array([x[-1] for x in area])
                with open(os.path.join(path, 'close_episode_len.txt'), 'r') as f:
                    length = np.array([float(x.rstrip()) for x in f.readlines()])
                with open(os.path.join('scenes', f'{s}.scenes'), 'r') as f:
                    scenes = [x.rstrip() for x in f.readlines()]
                parallel = len(scenes)
                ratio = ratio.reshape(parallel * args.num_eps, 1)
                area = area.reshape(parallel * args.num_eps, 1) / (ratio + 1e-3)
                length = length.reshape(parallel * args.num_eps, 1)
                result[method][0].append(ratio)
                result[method][1].append(area)
                result[method][2].append(length)
            except Exception as e:
                print(path)
                raise e

    mean_area = None
    valid_length = None
    valid_ratio = None
    for method_name in list(result.keys()):
        ratio, area, length = result[method_name]
        if mean_area is None:
            mean_area = np.vstack(area)
        else:
            mean_area += np.vstack(area)
        ratio, length = np.vstack(ratio), np.vstack(length)
        valid_length = length < 4999 if valid_length is None else (length < 4999) & valid_length
        valid_ratio = ratio > 0.9 if valid_ratio is None else (ratio > 0.9) & valid_ratio
        result[method_name] = ratio, length
    mean_area /= len(result.keys())

    splited_result = {}
    for r in split:
        mask = (mean_area >= r[0]) & (mean_area < r[1]) & valid_length
        desc = f'[{r[0]},{r[1]})'
        sub_result = {}
        for method_name, (ratio, length) in result.items():
            sub_result[method_name] = ratio[mask].mean(), length[mask].mean()
        splited_result[desc] = sub_result
    
    for desc, sub_result in splited_result.items():
        print(desc + ':')
        for method_name, (ratio, length) in sub_result.items():
            print('\t{}:'.format(method_name), '{:.1f}%'.format(ratio * 100), '{:.1f}'.format(length))
        rl_step = sub_result['rl'][1]
        coscan_step = sub_result['coscan'][1]
        imp = (coscan_step - rl_step) / coscan_step
        print('\timprovement: {:.1f}'.format(imp * 100))
    

if __name__ == '__main__':
    main()

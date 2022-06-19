import cv2
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp1', type=str)
    parser.add_argument('exp2', type=str)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--spec-stage', type=int, default=0)
    parser.add_argument('--spec-episode', type=int, default=1)
    parser.add_argument('--dir', type=str, default='temp')
    args = parser.parse_args()
    assert args.spec_episode == 1
    
    for i in (range(1, 1000) if args.spec_stage <= 0 else [args.spec_stage]):
        if not os.path.exists(f'{args.dir}/dump/{args.exp1}/episodes/{i}') or not os.path.exists(f'{args.dir}/dump/{args.exp2}/episodes/{i}'):
            break
        for j in (range(1, 1000) if args.spec_episode <= 0 else [args.spec_episode]):
            if not os.path.exists(f'{args.dir}/dump/{args.exp1}/episodes/{i}/{j}') or not os.path.exists(f'{args.dir}/dump/{args.exp2}/episodes/{i}/{j}'):
                break
            dir = os.path.join(args.output_dir, f'{i-1}-{j}')
            if not os.path.exists(dir):
                os.makedirs(dir)
            print('Saving as', dir)
            img1, img2 = None, None
            for k in range(0, int(1e10), 25):
                filename1 = f'{args.dir}/dump/{args.exp1}/episodes/{i}/{j}/{i-1}-{j}-Vis-{k}.png'
                filename2 = f'{args.dir}/dump/{args.exp2}/episodes/{i}/{j}/{i-1}-{j}-Vis-{k}.png'
                if not os.path.exists(filename1) and not os.path.exists(filename2):
                    break
                img1 = cv2.imread(filename1) if os.path.exists(filename1) else img1
                img2 = cv2.imread(filename2) if os.path.exists(filename2) else img2
                assert cv2.imwrite(os.path.join(dir, f'merge-{k}.png'), np.vstack((img1, img2)))


if __name__ == '__main__':
    main()
    
    

import cv2
import argparse
import os
from PIL import Image
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--spec-stage', type=int, default=0)
    parser.add_argument('--spec-episode', type=int, default=0)
    parser.add_argument('--dir', type=str, default='temp')
    parser.add_argument('-q', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for i in (range(1, 1000) if args.spec_stage <= 0 else [args.spec_stage]):
        if not os.path.exists(f'{args.dir}/dump/{args.path}/episodes/{i}'):
            break
        for j in (range(1, 1000) if args.spec_episode <= 0 else [args.spec_episode]):
            if not os.path.exists(f'{args.dir}/dump/{args.path}/episodes/{i}/{j}'):
                break
            video_name = os.path.join(args.output_dir, f'{i-1}-{j}-Vis.avi')
            if not args.q:
                print('Saving as', video_name)
            video = cv2.VideoWriter(video_name, fourcc=cv2.VideoWriter_fourcc(*"MP42"), apiPreference=0, fps=12.0, frameSize=(1066, 600))
            for k in range(int(1e10)):
                filename = f'{args.dir}/dump/{args.path}/episodes/{i}/{j}/{i-1}-{j}-Vis-{k}.png'
                if not os.path.exists(filename):
                    break
                video.write(cv2.resize(cv2.imread(filename), (1066, 600)))
            video.release()
    
if __name__ == '__main__':
    main()
    
    

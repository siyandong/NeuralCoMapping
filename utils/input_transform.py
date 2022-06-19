from skfmm import distance
from numpy import ma
import numpy as np
import torch
from math import log
from scipy.ndimage import distance_transform_edt


lazy_gaussian = None

def distance_field(input, DT_target, optimized=()):
    DT_target = DT_target.bool()
    
    if optimized:
        rows, cols = optimized
        ex1 = np.argmax(rows) - 2
        ex2 = len(rows) - np.argmax(np.flip(rows)) + 2
        ey1 = np.argmax(cols) - 2
        ey2 = len(cols) - np.argmax(np.flip(cols)) + 2

        ex1 = max(0, ex1)
        ex2 = min(DT_target.size(0), ex2)
        ey1 = max(0, ey1)
        ey2 = min(DT_target.size(1), ey2)
        
        DT_target = DT_target[ex1:ex2, ey1:ey2]
        goal_mask = input[ex1:ex2, ey1:ey2] > 0
        input.fill_(4)
        input = input[ex1:ex2, ey1:ey2]
    else:
        goal_mask = input > 0


    traversible = (~DT_target).numpy()
    traversible_ma = ma.masked_values(traversible, 0)
    traversible_ma[goal_mask] = 0
    df = distance(traversible_ma, dx=0.02)
    input.copy_(torch.from_numpy(ma.filled(df, 4)))

def gaussian_dist_map(input, half_ratio=0.25):
    '''
    where distance/map_size=half_ratio, prob=0.5*max_prob
    '''
    global lazy_gaussian
    assert input.size(1) == input.size(2)
    size = input.size(1)
    if lazy_gaussian is None or lazy_gaussian.size(0) != size * 2 + 1:
        '''
        f(d)=1/(2*pi*s^2)*exp(-0.5*d^2/s^2)
        f(size*ratio)=f_max*0.5 means exp(-0.5*(size*ratio)^2/s^2)=0.5 
        thus (size*ratio)^2=2*s^2*log(2), s^2=(size*ratio)^2/log(4)
        '''
        sigma2 = (size * half_ratio)**2 / log(4)
        lazy_gaussian = torch.zeros(2*size+1, 2*size+1)
        for i in range(2*size+1):
            for j in range(2*size+1):
                lazy_gaussian[i, j] = -0.5 * ((i - size)**2 + (j - size)**2) / sigma2
        lazy_gaussian = torch.exp(lazy_gaussian.cuda())
    for i in range(input.size(0)):
        if input[i].sum() > 5:
            sigma2 = (size * half_ratio)**2 / log(4)
            coeff = -0.5 / sigma2
            output = distance_transform_edt(~input[i].bool().cpu().numpy())
            output = torch.exp(torch.from_numpy(output).cuda() * coeff)
        else:
            index = input[i].nonzero()
            output = torch.zeros(size, size).cuda()
            for j in range(index.size(0)):
                x, y = index[j, 0].item(), index[j, 1].item()
                output = torch.max(output, lazy_gaussian[size-x:2*size-x, size-y:2*size-y])
        input[i] = output



def global_input_transform(input, channels, method, DT_target_channels):
    assert method in ['pos', 'distance_field', 'gaussian_dist_map']
    if method == 'pos':
        return
    for channel, dt_channel in zip(channels, DT_target_channels):
        if method =='distance_field':
            distance_field(input[:, channel], input[:, dt_channel])
        else:
            gaussian_dist_map(input[:, channel])
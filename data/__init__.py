'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data
from pdb import set_trace as bp
import numpy as np
import random
import os
import sys
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
except ImportError:
    pass
import data.util as util
from torch.utils.data import DataLoader


def create_dataloader(dataset, dataset_opt, opt, sampler):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        if dataset_opt['name'] == 'Adobe_a':
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False, collate_fn=collate_function)
        else:
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    if mode == 'Adobe_a':
        from data.Adobe_arbitrary import AdobeDataset as D
    elif mode == 'Adobe':
        from data.Adobe_dataset import AdobeDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset


def collate_function(data):
    '''
    We want to create a dataloader, which would randomly select a down-sampling scale for each batch.
    If down-sampling is performed in __getitem__ function, when num_workers > 1,
    each subprocess would have a different scale, resulting in unmatched resolutions.
    Therefore, we define a collate function for down-sampling.
    In this function, we fix resolutions of down-sampled input images to (64, 64)
    and set resolutions of GT images to (64 * d_scale, 64 * d_scale), where d_scale is the randomly sampled scale.

    For the operation of temporal sampling, see line 189 - 191 in Adobe_arbitrary.py
    '''
    d_scale = random.uniform(2, 4) # randomly select down-sampling scale in [2, 4]
    LQ_size = 64 # fixed resolution for LQ images
    GT_size = int(np.floor(LQ_size * d_scale))
    
    ### Image Cropping ###
    x = random.randint(0, max(0, 720 - GT_size))
    y = random.randint(0, max(0, 1280 - GT_size))
    img_LQ_l = [np.stack([img_[0][i][x:x+GT_size,y:y+GT_size] if img_[0][i].shape[0] == 720 else img_[0][i][y:y+GT_size,x:x+GT_size] for img_ in data], axis=0) for i in range(len(data[0][0]))]
    img_GT_l = [np.stack([img_[1][i][x:x+GT_size,y:y+GT_size] if img_[1][i].shape[0] == 720 else img_[1][i][y:y+GT_size,x:x+GT_size] for img_ in data], axis=0) for i in range(len(data[0][1]))]

    ### Down-sampling ###
    img_LQ_l = [np.stack([imresize_np(img_[i], 1/(2*d_scale), True) for i in range(img_.shape[0])],axis=0) for img_ in img_LQ_l]
    img_GT_l = [np.stack([imresize_np(img_[i], 1/2, True) for i in range(img_.shape[0])],axis=0) for img_ in img_GT_l]

    img_LQ_l = [img_.astype(np.float32) / 255. for img_ in img_LQ_l]
    img_GT_l = [img_.astype(np.float32) / 255. for img_ in img_GT_l]
    img_LQs = np.stack(img_LQ_l, axis=0)
    img_GTs = np.stack(img_GT_l, axis=0)
    # augmentation - flip, rotate
    img_LQs, img_GTs = util.augment_a2(img_LQs, img_GTs, True, True)

    # BGR to RGB, HWC to CHW, numpy to tensor
    img_GTs = img_GTs[:, :, :, :, [2, 1, 0]]
    img_LQs = img_LQs[:, :, :, :, [2, 1, 0]]

    img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs, (1, 0, 4, 2, 3)))).float()
    img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs, (1, 0, 4, 2, 3)))).float()

    time_t = [torch.cat([time_[2][i][None] for time_ in data], dim=0) for i in range(len(data[0][2]))]
    
    return {'LQs': img_LQs, 'GT': img_GTs, 'scale': [[img_GTs.shape[-2]], [img_GTs.shape[-1]]], 'time': time_t}     

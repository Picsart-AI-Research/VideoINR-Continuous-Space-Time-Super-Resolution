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
                                           pin_memory=False, collate_fn=collate_function2)
        else:
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    if mode == 'Vimeo7':
        from data.Vimeo7_dataset import Vimeo7Dataset as D
    elif mode == 'Adobe':
        from data.Adobe_dataset import AdobeDataset as D
    elif mode == 'Adobe_a':
        from data.Adobe_arbitrary import AdobeDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset


def collate_function(data):
    d_scale = random.uniform(2, 8)
    # img_LQo_l, img_GTo_l: [(B,C,H,W)]
    print(data[0][3], data[1][3], data[2][3], data[3][3])
    print(data[0][0][0].shape, data[1][0][0].shape, data[2][0][0].shape, data[3][0][0].shape)
    img_LQo_l = [np.concatenate([img_[0][i][None] for img_ in data], axis=0) for i in range(len(data[0][0]))]
    img_GTo_l = [np.concatenate([img_[1][i][None] for img_ in data], axis=0) for i in range(len(data[0][1]))]

    width_l, height_l = int(np.floor(img_LQo_l[0].shape[2] / d_scale)), int(np.floor(img_LQo_l[0].shape[1] / d_scale))
    width_g, height_g = int(np.floor(img_LQo_l[0].shape[2] / 2)), int(np.floor(img_LQo_l[0].shape[1] / 2))

    scaled_width_l, scaled_height_l = int(np.floor(d_scale * width_l)), int(np.floor(d_scale * height_l))
    if len(img_LQo_l[0].shape) == 4:
        img_LQo_l = [img_[:, 0:scaled_height_l, 0:scaled_width_l, :] for img_ in img_LQo_l]
        img_GTo_l = [img_[:, 0:2 * height_g, 0:2 * width_g, :] for img_ in img_GTo_l]
    else:
        img_LQo_l = [img_[:, 0:scaled_height_l, 0:scaled_width_l] for img_ in img_LQo_l]
        img_GTo_l = [img_[:, 0:2 * height_g, 0:2 * width_g] for img_ in img_GTo_l]

    img_LQ_l = [np.concatenate([imresize_np(img_[i], 1 / d_scale, True)[None] for i in range(img_.shape[0])], axis=0) for img_ in img_LQo_l]
    img_GT_l = [np.concatenate([imresize_np(img_[i], 1 / 2, True)[None] for i in range(img_.shape[0])], axis=0) for img_ in img_GTo_l]

    img_LQ_l = [img_.astype(np.float32) / 255. for img_ in img_LQ_l]
    img_GT_l = [img_.astype(np.float32) / 255. for img_ in img_GT_l]
        
    img_LQ_l = [img_[:, :, :, :3] for img_ in img_LQ_l]
    img_GT_l = [img_[:, :, :, :3] for img_ in img_GT_l]
            
            
    # LQ_size_tuple = (3, 64, 112) if self.LR_input else (3, 256, 448)
    C, H, W = img_LQ_l[0].shape[3], img_LQ_l[0].shape[1], img_LQ_l[0].shape[2]
    LQ_size = 48
    d_scale /= 2
    GT_size = int(LQ_size * d_scale)

    rnd_h = random.randint(0, max(0, H - LQ_size))
    rnd_w = random.randint(0, max(0, W - LQ_size))
    img_LQ_l = [v[:, rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQ_l]
    rnd_h_HR, rnd_w_HR = int(rnd_h * d_scale), int(rnd_w * d_scale)
    img_GT_l = [v[:, rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :] for v in img_GT_l]

    # augmentation - flip, rotate
    img_LQ_l = img_LQ_l + img_GT_l
    rlt = util.augment_a(img_LQ_l, True, True)
    img_LQ_l = rlt[0:2]
    img_GT_l = rlt[2:]
    
    # stack LQ images to NHWC, N is the frame number
    img_LQs = np.stack(img_LQ_l, axis=0)
    img_GTs = np.stack(img_GT_l, axis=0)
    # BGR to RGB, HWC to CHW, numpy to tensor
    img_GTs = img_GTs[:, :, :, :, [2, 1, 0]]
    img_LQs = img_LQs[:, :, :, :, [2, 1, 0]]

    img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs, (1, 0, 4, 2, 3)))).float()
    img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs, (1, 0, 4, 2, 3)))).float()

    time_t = [torch.cat([time_[2][i][None] for time_ in data], dim=0) for i in range(len(data[0][2]))]
    return {'LQs': img_LQs, 'GT': img_GTs, 'shape': (GT_size, GT_size), 'time': time_t}            


def collate_function2(data):
    d_scale = random.uniform(2, 4)
    LQ_size = 64
    GT_size = int(np.floor(LQ_size * d_scale))
    
    x = random.randint(0, max(0, 720 - GT_size))
    y = random.randint(0, max(0, 1280 - GT_size))
    img_LQ_l = [np.stack([img_[0][i][x:x+GT_size,y:y+GT_size] if img_[0][i].shape[0] == 720 else img_[0][i][y:y+GT_size,x:x+GT_size] for img_ in data], axis=0) for i in range(len(data[0][0]))]
    img_GT_l = [np.stack([img_[1][i][x:x+GT_size,y:y+GT_size] if img_[1][i].shape[0] == 720 else img_[1][i][y:y+GT_size,x:x+GT_size] for img_ in data], axis=0) for i in range(len(data[0][1]))]
    img_LQ_l = [np.stack([imresize_np(img_[i], 1/(2*d_scale), True) for i in range(img_.shape[0])],axis=0) for img_ in img_LQ_l]
    img_GT_l = [np.stack([imresize_np(img_[i], 1/2, True) for i in range(img_.shape[0])],axis=0) for img_ in img_GT_l]

    img_LQ_l = [img_.astype(np.float32) / 255. for img_ in img_LQ_l]
    img_GT_l = [img_.astype(np.float32) / 255. for img_ in img_GT_l]
    img_LQs = np.stack(img_LQ_l, axis=0)
    img_GTs = np.stack(img_GT_l, axis=0)
    # augmentation - flip, rotate
    img_LQs, img_GTs = util.augment_a2(img_LQs, img_GTs, True, True)
    
    # stack LQ images to NHWC, N is the frame number
    # img_LQs = np.stack(img_LQ_l, axis=0)
    # img_GTs = np.stack(img_GT_l, axis=0)
    # BGR to RGB, HWC to CHW, numpy to tensor
    img_GTs = img_GTs[:, :, :, :, [2, 1, 0]]
    img_LQs = img_LQs[:, :, :, :, [2, 1, 0]]

    img_GTs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTs, (1, 0, 4, 2, 3)))).float()
    img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs, (1, 0, 4, 2, 3)))).float()

    time_t = [torch.cat([time_[2][i][None] for time_ in data], dim=0) for i in range(len(data[0][2]))]
    return {'LQs': img_LQs, 'GT': img_GTs, 'shape': (img_GTs.shape[-2], img_GTs.shape[-1]), 'time': time_t}     
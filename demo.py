import os
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import models.modules.Sakuya_arch as Sakuya_arch

from pdb import set_trace as bp
from data.util import imresize_np

parser = argparse.ArgumentParser()
parser.add_argument('--space_scale', type=int, default=4, help="upsampling space scale")
parser.add_argument('--time_scale', type=int, default=8, help="upsampling time scale")
parser.add_argument('--data_path', type=str, required=True, help="data path for testing")
parser.add_argument('--out_path_lr', type=str, default="./output/LR/", help="output path (Low res image)")
parser.add_argument('--out_path_bicubic', type=str, default="./output/Bicubic/", help="output path (bicubic upsampling)")
parser.add_argument('--out_path_ours', type=str, default="./output/VideoINR/", help="output path (VideoINR)")
parser.add_argument('--model_path', type=str, default="latest_G.pth", help="model parameter path")
opt = parser.parse_known_args()[0]

device = 'cuda'
model = Sakuya_arch.LunaTokis(64, 6, 8, 5, 40)
model.load_state_dict(torch.load(opt.model_path), strict=True)

model.eval()
model = model.to(device)

def single_forward(model, imgs_in, space_scale, time_scale):
    with torch.no_grad():
        b, n, c, h, w = imgs_in.size()
        h_n = int(4 * np.ceil(h / 4))
        w_n = int(4 * np.ceil(w / 4))
        imgs_temp = imgs_in.new_zeros(b, n, c, h_n, w_n)
        imgs_temp[:, :, :, 0:h, 0:w] = imgs_in

        time_Tensors = [torch.tensor([i / time_scale])[None].to(device) for i in range(time_scale)]
        model_output = model(imgs_temp, time_Tensors, space_scale, test=True)
        return model_output


os.makedirs(opt.out_path_lr, exist_ok=True)
os.makedirs(opt.out_path_bicubic, exist_ok=True)
os.makedirs(opt.out_path_ours, exist_ok=True)

path_list = [os.path.join(opt.data_path, name) for name in sorted(os.listdir(opt.data_path))]
index = 0
for ind in tqdm(range(len(path_list) - 1)):

    imgpath1 = os.path.join(path_list[ind])
    imgpath2 = os.path.join(path_list[ind + 1])

    img1 = cv2.imread(imgpath1, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(imgpath2, cv2.IMREAD_UNCHANGED)

    '''
    We apply down-sampling on the original video
    in order to avoid CUDA out of memory.
    You may skip this step if your input video
    is already of relatively low resolution.
    '''
    img1 = imresize_np(img1, 1 / 8, True).astype(np.float32) / 255.
    img2 = imresize_np(img2, 1 / 8, True).astype(np.float32) / 255.

    Image.fromarray((np.clip(img1[:, :, [2, 1, 0]], 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(opt.out_path_lr, path_list[ind].split('/')[-1]))

    imgs = np.stack([img1, img2], axis=0)[:, :, :, [2, 1, 0]]
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()[None].to(device)

    output = single_forward(model, imgs, opt.space_scale, opt.time_scale)

    '''
    Save results of VideoINR and bicubic up-sampling.
    '''
    for out_ind in range(len(output)):

        img = output[out_ind][0]
        img = Image.fromarray((img.clamp(0., 1.).detach().cpu().permute(1, 2, 0) * 255).numpy().astype(np.uint8))
        img.save(os.path.join(opt.out_path_ours, '{}.jpg'.format(index)))

        HH, WW = img1.shape[0] * 4, img1.shape[1] * 4
        img = Image.fromarray((np.clip(img1[:, :, [2, 1, 0]], 0, 1) * 255).astype(np.uint8)).resize((WW, HH),Image.BICUBIC)
        img.save(os.path.join(opt.out_path_bicubic, '{}.jpg'.format(index)))
        index += 1

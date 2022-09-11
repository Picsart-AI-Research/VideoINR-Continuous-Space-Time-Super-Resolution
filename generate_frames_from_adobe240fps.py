import cv2
import numpy as np
from PIL import Image
import glob
import cv2
import os
import shutil
from pdb import set_trace as bp

# Configuration
# ============================
'''
A hardcoded path here as an example
'''
videoFolder = '/home/users/adobe240/original_high_fps_videos'
frameFolder = '/home/users/adobe240/frame'

train_txt = 'data/adobe240fps_folder_train.txt'
valid_txt = 'data/adobe240fps_folder_valid.txt'
test_txt = 'data/adobe240fps_folder_test.txt'

# Run
# ============================

with open(train_txt) as f:
    temp = f.readlines()
    train_list = [v.strip() for v in temp]

with open(valid_txt) as f:
    temp = f.readlines()
    valid_list = [v.strip() for v in temp]

with open(test_txt) as f:
    temp = f.readlines()
    test_list = [v.strip() for v in temp]


mov_files = glob.glob(os.path.join(videoFolder, '*'))

def check_if_folder_exist(folder_path='/home/ubuntu/'):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        if not os.path.isdir(folder_path):
            print('Folder: ' + folder_path + ' exists and is not a folder!')
            exit()

name_list = []
for i, mov_path in enumerate(mov_files):
    if mov_path.split('/')[-1].split('.')[0] in valid_list:
        mov_folder = os.path.join(frameFolder, 'valid')
    elif mov_path.split('/')[-1].split('.')[0] in test_list:
        mov_folder = os.path.join(frameFolder, 'test')
    elif mov_path.split('/')[-1].split('.')[0] in train_list:
        mov_folder = os.path.join(frameFolder, 'train')
    
    image_index = 0
    video = cv2.VideoCapture(mov_path)
    success, frame = video.read()
    frame = np.transpose(frame, (0, 1, 2))
    while success:
        save_folder = os.path.join(mov_folder, mov_path.split('/')[-1].split('.')[0])
        check_if_folder_exist(save_folder)
        cv2.imwrite(os.path.join(save_folder, str(image_index) + '.png'), frame)
        print(str(i) + ' ' + str(image_index))
        image_index += 1
        success, frame = video.read()
        if not success:
            break
        frame = np.transpose(frame, (0, 1, 2))

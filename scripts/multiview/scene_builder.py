#from scripts import calc_gt_info
#from scripts import calc_gt_masks
#from scripts.multiview import depth_generator
import os
import csv
import ast

import numpy as np
import glob
import shutil

from bop_toolkit_lib.inout import save_scene_gt


cams_ids = [0, 1]
base_path = '/home/hazem/projects/multi_view_dataset'
recording = 'recordings_test/11_11_11 08_11_2021'
recording_path = os.path.join(base_path, recording)

def copy_images(req_img_id, out_path):
    for cam_id in cams_ids:
        imgs_path = os.path.join(recording_path, 'camera_'+str(cam_id), 'images')
        for img in sorted(glob.glob(os.path.join(imgs_path, '*.png'))):
            img_id = img.split('_')[-1].split('.')[-2]
            if img_id == str(req_img_id):
                new_img_id = f'{cam_id:06d}' # use cam_id as img_id
                shutil.copy2(os.path.join(imgs_path, 'image_'+str(img_id)+'.png'), os.path.join(out_path, str(new_img_id)+'.png'))
    print('Images copied')

def read_gt_csv(cam_path):
    '''
        Reads a single csv file and saves it to a dict in an intermediary format.
        output format: a list of dicts containing the GT pose and obj_id, with image id as key.
        Each dict corresponds to an object GT pose (in the same image).
    '''
    out_dict = {}
    with open (os.path.join(cam_path, 'data.csv')) as f:
        reader = csv.reader(f)
        reader_list = list(reader)
        #print(reader_list)
        img_ids = []
        content = []
        for row in reader_list[1:]:
            img_id = int(os.path.split(row[2])[-1].split('.')[-2].split('_')[-1])
            img_id = f'{img_id:06d}'
            #cam_id = int(os.path.split(row[2])[-1].split('.')[-2].split('_')[-1])
            #out_dict['cam_id'] = cam_id
            entry = {'cam_t_m2c': np.asarray(ast.literal_eval(row[3])), 'cam_R_m2c': np.asarray(ast.literal_eval(row[4])), 'obj_id': int(row[0])}
            if img_id not in img_ids:
                content = []
            content.append(entry)
            out_dict[img_id] = content
            img_ids.append(img_id)
            #print(out_dict)
    return out_dict

# TODO: add obj2cam transformation
def create_scene_gt(req_img_id, out_path):
    '''
        Combines GT values from all cameras for a particular 'snap' to create a json file
        Img_id: id of interest to concatenate from all cameras to build a scene
    '''
    out_dict = {}
    instances = 0
    for cam_id in cams_ids:
        cam_path = os.path.join(recording_path, 'camera_'+str(cam_id))
        gt_dict = read_gt_csv(cam_path)
        img_id_mod = f'{req_img_id:06d}'
        if img_id_mod in sorted(gt_dict.keys()):
            out_dict[str(cam_id)] = gt_dict[str(img_id_mod)]  # img id is the cam_id (within a single scene)
            instances += 1
        #print((out_dict))
    if instances == 0:
        print('Scene is Empty!')
    out_path = os.path.join(out_path, 'scene_gt.json')
    save_scene_gt(out_path, out_dict)
    return out_dict

#TODO: replace mode with dataset split
def build_scene(mode):
    '''
    :param mode: train or test
    :return:
    '''
    max_len = get_max_len(recording_path) # accounts for images' filtration
    for i in range(max_len):
        scene_id = f'{i:06d}'
        out_path = os.path.join(recording_path, mode, scene_id)
        for cam_id in cams_ids:
            pass

def get_max_len(recording_path):
    '''
        Gets max num of images per cam to account for invalid images during filtration
    '''
    max_val = 0
    for cam_id in cams_ids:
        imgs_path = os.path.join(recording_path, 'camera_'+str(cam_id), 'images')
        imgs_num = len(glob.glob(os.path.join(imgs_path, '*.png')))
        if imgs_num > max_val:
            max_val = imgs_num
    print(max_val)
    return max_val

if __name__ == '__main__':
    cam_id = 0
    cam_path = os.path.join(recording_path, 'camera_' + str(cam_id))
    read_gt_csv(cam_path)
    create_scene_gt(0, '/home/hazem/projects/multi_view_dataset/mv/test/000001')
    #test_dict()
    #copy_images(0, '/home/hazem/projects/multi_view_dataset/mv/test/000001')
    get_max_len(recording_path)

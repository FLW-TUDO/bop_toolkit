import copy
import os
import csv
import ast

import numpy as np
import glob
import shutil

from icecream import ic
from bop_toolkit_lib.inout import save_scene_gt, save_scene_camera
from bop_toolkit_lib import dataset_params, config
from tqdm import tqdm
ic.disable()


# PARAMETERS.
################################################################################
p = {
  # See dataset_params.py for options.
  'dataset': 'mv',

  # Dataset split. Options: 'train', 'val', 'test'.
  'dataset_split': 'test',

  # Dataset split type. None = default. See dataset_params.py for options.
  'dataset_split_type': None,

  # Folder containing the BOP datasets.
  'datasets_path': config.datasets_path,
}
################################################################################


# Load dataset parameters.
dp_split = dataset_params.get_split_params(
  p['datasets_path'], p['dataset'], p['dataset_split'], p['dataset_split_type'])

cams_ids = [0]
# recording = '13_51 25_02_2022'
calib_params_path = os.path.join(p['datasets_path'], p['dataset'], 'calib_params.csv') #only full split path available from dp_split


scene_directories = glob.glob(os.path.join(dp_split['split_path'], '*'))
for scene in scene_directories:
    shutil.rmtree(scene)


def get_cam_ids():
    # Method to get cam id from directory list in filtered images
    pass

def copy_images(req_img_id, recording_path, out_path):
    '''
        Modifies names of raw images and copies them to dataset directory
        @param req_img_id: img id of interest (i.e: scene id) in long format
    '''
    for cam_id in cams_ids:
        imgs_path = os.path.join(recording_path, 'camera_'+str(cam_id), 'images')
        for img in sorted(glob.glob(os.path.join(imgs_path, '*.png'))):
            #img_id = img.split('_')[-1].split('.')[-2]
            img_id = os.path.split(img)[-1].split('.')[-2]
            req_img_id = str(int(req_img_id)) # convert id to short format
            if img_id == req_img_id:
                new_img_id = f'{cam_id:06d}' # use cam_id as img_id
                shutil.copy2(os.path.join(imgs_path, str(img_id)+'.png'), os.path.join(out_path, str(new_img_id)+'.png'))
    #print('Images copied')

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
        keys = reader_list[0]
        for row in reader_list[1:]:
            img_id = int(os.path.split(row[keys.index('ImageName')])[-1].split('.')[-2].split('_')[-1])
            img_id = f'{img_id:06d}'
            #cam_id = int(os.path.split(row[2])[-1].split('.')[-2].split('_')[-1])
            #out_dict['cam_id'] = cam_id
            entry = {'cam_t_m2c': np.asarray(ast.literal_eval(row[keys.index('camToObjTrans')])), 'cam_R_m2c': np.asarray(ast.literal_eval(row[keys.index('camToObjRot')])), 'obj_id': int(row[keys.index('ObjectID')])}
            if img_id not in img_ids:
                content = []
            content.append(entry)
            out_dict[img_id] = content
            img_ids.append(img_id)
            #print(out_dict)
    ic(out_dict)
    return out_dict

# TODO: add obj2cam transformation
# TODO: add ability to augment on existing scenes (for multiple recordings)
def create_scene_gt(req_img_id, recording_path, out_path):
    '''
        Combines GT values from all cameras for a particular 'snap' to create a json file
        Img_id: id of interest to concatenate from all cameras to build a scene
        :param req_img_id: img id of interest (i.e: scene id) in long format
    '''
    out_dict = {}
    instances = 0
    for cam_id in cams_ids:
        cam_path = os.path.join(recording_path, 'camera_'+str(cam_id))
        gt_dict = read_gt_csv(cam_path)
        gt_dict_copy = copy.deepcopy(gt_dict)
        #img_id_mod = f'{req_img_id:06d}'
        if req_img_id in sorted(gt_dict.keys()):
            out_dict[str(cam_id)] = gt_dict_copy[str(req_img_id)]  # img_id is the cam_id (within a single scene)
            instances += 1
        #print((out_dict))
    if instances == 0:
        print('Scene is Empty!')
    out_path = os.path.join(out_path, 'scene_gt.json')
    save_scene_gt(out_path, out_dict)
    return out_dict

# TODO: add cam_t_w2c and cam_R_w2c
def read_calib_params(calib_path):
    '''
        Loads csv file with calib params and saves it to a dict
    '''
    out_dict = {}
    content = {}
    with open(calib_path) as f:
        reader = csv.reader(f)
        reader_list = list(reader)
        keys = reader_list[0]
        for row in reader_list[1:]:
            content['cam_K'] = np.asarray(ast.literal_eval(row[keys.index('cam_K')]))
            content['depth_scale'] = int(row[keys.index('depth_scale')])
            content_copy = copy.deepcopy(content)
            out_dict[row[keys.index('cam_id')]] = content_copy
        #print(out_dict)
    return out_dict

# TODO: make function independent from GT values and reliant on scene from folder structure
def create_scene_camera(gt_dict, out_path):
    '''
        creates scene_camera.json for a scene using its GT values
    '''
    out_dict = {}
    calib_params = read_calib_params(calib_params_path)
    for img_id in sorted(gt_dict.keys()): # assuming filtered recordings
        out_dict[img_id] = calib_params[img_id]
    out_path = os.path.join(out_path, 'scene_camera.json')
    save_scene_camera(out_path, out_dict)
    return out_dict

#TODO: replace mode with dataset split
def build_scene(mode, recording_path, scene_id):
    '''
        Builds all scenes of the dataset
    :param mode: train or test
    :return:
    '''
    scene_id = f'{scene_id:06d}'
    out_path = os.path.join(dp_split['split_path'], scene_id)
    #out_path = os.path.join(out_path, mode, scene_id)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    gt = create_scene_gt(scene_id, recording_path, out_path) # scene_id corresponds (initially, berore modifying images with cam id) to the requested images id
    imgs_out_path = os.path.join(out_path, 'rgb')
    if not os.path.exists(imgs_out_path):
        os.makedirs(imgs_out_path)
    copy_images(scene_id, recording_path, imgs_out_path)
    create_scene_camera(gt, out_path)

def build_dataset(mode):
    # recording_path = os.path.join(p['datasets_path'], 'recordings', recording)
    list_of_files = glob.glob(os.path.join(p['datasets_path'], 'recordings', '*'))
    recording_path = max(list_of_files, key=os.path.getctime)
    print(recording_path)
    max_len = get_max_len(recording_path) # accounts for images' filtration
    for i in tqdm(range(max_len)):
        build_scene(mode, recording_path, i)
        #print(f'scene: {i} done')
    #create mode dis - test, train


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
    #print(max_val)
    return max_val

if __name__ == '__main__':
    build_dataset(dp_split['split_path'])

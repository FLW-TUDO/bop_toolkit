import copy
import os
import csv
import ast

import numpy as np
import glob
import shutil

from icecream import ic
from bop_toolkit_lib.inout import save_scene_gt, save_scene_camera, save_depth
from bop_toolkit_lib import dataset_params, config
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from bop_toolkit_lib import renderer

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

    # Type of the renderer.
    'renderer_type': 'vispy',  # Options: 'vispy', 'cpp', 'python'.

    # Folder containing the BOP datasets.
    'datasets_path': config.datasets_path,
}
################################################################################


# Load dataset parameters.
dp_split = dataset_params.get_split_params(
    p['datasets_path'], p['dataset'], p['dataset_split'], p['dataset_split_type'])

cams_ids = [0,7]
# recording = '13_51 25_02_2022'
calib_params_path = os.path.join(p['datasets_path'], p['dataset'],
                                 'calib_params.csv')  # only full split path available from dp_split


model_type = None
dp_model = dataset_params.get_model_params(
  p['datasets_path'], p['dataset'], model_type)

# Load dataset parameters.
dp_split = dataset_params.get_split_params(
  p['datasets_path'], p['dataset'], p['dataset_split'], p['dataset_split_type'])

# The renderer has a larger canvas for generation of masks of truncated objects.
im_width, im_height = dp_split['im_size']
ren_width, ren_height = 3 * im_width, 3 * im_height
ren_cx_offset, ren_cy_offset = im_width, im_height
ren = renderer.create_renderer(
  ren_width, ren_height, p['renderer_type'], mode='depth')

for obj_id in dp_model['obj_ids']:
  model_fpath = dp_model['model_tpath'].format(obj_id=obj_id)
  ren.add_object(obj_id, model_fpath)





def get_homogenous_form(rot, trans):
    '''
    @param rot: rotation matrix
    @type rot: numpy array
    @param trans: translation vector
    @type trans: numpy
    @return: homogenous form
    @rtype: numpy array
    '''
    mat = np.column_stack((rot, trans))
    mat_homog = np.row_stack((mat, [0.0, 0.0, 0.0, 1.0]))
    return mat_homog


cam_locations_csv_file = '/media/athos/DATA-III/projects/RGB-Camera-System/cam_locations.csv'
camera2vicon = {}
with open(cam_locations_csv_file) as f:
    reader = csv.reader(f)
    reader_list = list(reader)
    for row in reader_list:
        cameraId = row[0]
        translation = np.array(ast.literal_eval(row[1]))
        rotation = np.array(ast.literal_eval(row[2]))  # rotation matrix
        transfrom = get_homogenous_form(rotation, translation)
        camera2vicon[cameraId] = transfrom

scene_directories = glob.glob(os.path.join(dp_split['split_path'], '*'))
for scene in scene_directories:
    shutil.rmtree(scene)


def get_cam_ids():
    # Method to get cam id from directory list in filtered images
    pass


def copy_images(req_img_id, recording_path, out_path, cam_id):
    '''
        Modifies names of raw images and copies them to dataset directory
        @param req_img_id: img id of interest (i.e: scene id) in long format
    '''
    imgs_path = os.path.join(recording_path, 'camera_' + str(cam_id), 'images')
    for img in sorted(glob.glob(os.path.join(imgs_path, '*.png'))):
        # img_id = img.split('_')[-1].split('.')[-2]
        img_id = os.path.split(img)[-1].split('.')[-2]
        req_img_id = str(int(req_img_id))  # convert id to short format
        if img_id == req_img_id:
            new_img_id = f'{cam_id:06d}'  # use cam_id as img_id
            shutil.copy2(os.path.join(imgs_path, str(img_id) + '.png'),
                         os.path.join(out_path, str(new_img_id) + '.png'))
    # print('Images copied')


def invert_homog_transfrom(homog_trans):
    trans = homog_trans[0:3, 3]
    rot = homog_trans[0:3, 0:3]
    rot_inv = np.linalg.inv(rot)
    homog_inv = get_homogenous_form(rot_inv, -1 * (rot_inv.dot(trans)))
    return homog_inv


def generate_gt_transformation(cam_id, obj2vicon_trans, obj2vicon_rot):
    cam2vicon_transform = camera2vicon[str(cam_id)]
    obj2vicon_rot = R.from_euler('XYZ', obj2vicon_rot, degrees=False)
    obj2vicon_rot_mat = obj2vicon_rot.as_matrix()

    obj2vicon_transform = get_homogenous_form(obj2vicon_rot_mat, obj2vicon_trans)
    ic(obj2vicon_transform)

    obj2cam_transform = invert_homog_transfrom(cam2vicon_transform).dot(obj2vicon_transform)
    ic(obj2cam_transform)
    obj2cam_trans = obj2cam_transform[0:3, 3]
    obj2cam_rot = obj2cam_transform[0:3, 0:3]

    return obj2cam_trans, obj2cam_rot


def generate_depth_image(gt, cam_id):
    K = scene_camera[str(cam_id)]['cam_K']
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # Render depth image of the object model in the ground-truth pose.
    depth_gt_large = ren.render_object(
        gt['obj_id'], gt['cam_R_m2c'], gt['cam_t_m2c'],
        fx, fy, cx + ren_cx_offset, cy + ren_cy_offset)['depth']
    depth_gt = depth_gt_large[
               ren_cy_offset:(ren_cy_offset + im_height),
               ren_cx_offset:(ren_cx_offset + im_width)]
    if np.sum(depth_gt) < 100 or np.sum(depth_gt) < 0.9 * np.sum(depth_gt_large):
        return None
    return depth_gt

def combine_depth_images(depth_images, scene_id, img_id):
    im_shape = np.shape(depth_images[0])
    combined_depth_gt = np.zeros((im_shape[0], im_shape[1]))
    # print('Combining depth images..')
    for depth_img in depth_images:
        rows, columns = np.where(depth_img > 0)
        for i, j in zip(rows, columns):
            if (combined_depth_gt[i, j] == 0):  # updated by first non-zero value in any depth image
                combined_depth_gt[i, j] = depth_img[i, j]

    combined_depth_gt = np.flipud(combined_depth_gt)
    combined_depth_gt = np.fliplr(combined_depth_gt)

    save_depth(dp_split['depth_tpath'].format(scene_id=int(scene_id), im_id=int(img_id)), combined_depth_gt)


def read_gt_csv(cam_path, cam_id, scene_id):
    '''
        Reads a single csv file and saves it to a dict in an intermediary format.
        output format: a list of dicts containing the GT pose and obj_id, with image id as key.
        Each dict corresponds to an object GT pose (in the same image).
    '''
    out_dict = {}
    with open(os.path.join(cam_path, 'data.csv')) as f:
        reader = csv.reader(f)
        reader_list = list(reader)
        # print(reader_list)
        img_ids = []
        content = []
        keys = reader_list[0]
        depth_images = []
        for row in reader_list[1:]:

            # TODO Remove this from production
            if int(row[keys.index('ObjectID')]) == 5:
                continue



            img_id = int(os.path.split(row[keys.index('ImageName')])[-1].split('.')[-2].split('_')[-1])
            img_id = f'{img_id:06d}'
            # cam_id = int(os.path.split(row[2])[-1].split('.')[-2].split('_')[-1])
            # out_dict['cam_id'] = cam_id
            obj2vicon_trans = np.asarray(ast.literal_eval(row[keys.index('camToObjTrans')]))
            obj2vicon_rot = np.asarray(ast.literal_eval(row[keys.index('camToObjRot')]))
            camToObjTrans, cam_R_m2c = generate_gt_transformation(cam_id, obj2vicon_trans, obj2vicon_rot)
            entry = {'cam_t_m2c': camToObjTrans, 'cam_R_m2c': cam_R_m2c, 'obj_id': int(row[keys.index('ObjectID')])}
            depth_image = generate_depth_image(entry, cam_id)
            if img_id not in img_ids:
                img_ids.append(img_id)
                content = []
            if depth_image is not None:
                depth_images.append(depth_image)
                content.append(entry)
                out_dict[img_id] = content
            else:
                continue

            if img_id not in img_ids and len(img_ids) > 0 and len(depth_images) > 0:
                combine_depth_images(depth_images, scene_id, img_id)
                depth_images = []

        if len(depth_images) > 0:
            combine_depth_images(depth_images, scene_id, img_id)

            # print(out_dict)
    ic(out_dict)
    return out_dict


# TODO: add obj2cam transformation
# TODO: add ability to augment on existing scenes (for multiple recordings)
def create_scene_gt(scene_id, recording_path, out_path, gt, imgs_out_path):
    '''
        Combines GT values from all cameras for a particular 'snap' to create a json file
        Img_id: id of interest to concatenate from all cameras to build a scene
        :param req_img_id: img id of interest (i.e: scene id) in long format
    '''
    out_dict = {}

    for cam_id in cams_ids:
        gt_cam = gt[cam_id]
        depth_images = []
        valid_entries = []
        if scene_id in sorted(gt_cam.keys()):
            gt_dict = gt_cam[str(scene_id)]  # img_id is the cam_id (within a single scene)
        else:
            continue

        for entry in gt_dict:
            depth_image = generate_depth_image(entry, cam_id)
            if depth_image is not None:
                depth_images.append(depth_image)
                valid_entries.append(entry)
        if len(depth_images) < 1:
            print(f'Scene {scene_id} is empty for camera {cam_id}!')
            continue

        combine_depth_images(depth_images, scene_id, cam_id)
        copy_images(scene_id, recording_path, imgs_out_path, cam_id)
        out_dict[str(cam_id)] = valid_entries

    if not out_dict:
        return

    scene_path = os.path.join(out_path, 'scene_gt.json')
    save_scene_gt(scene_path, out_dict)
    create_scene_camera(gt, out_path)


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
        # print(out_dict)
    return out_dict


# TODO: make function independent from GT values and reliant on scene from folder structure
def create_scene_camera(gt_dict, out_path):
    '''
        creates scene_camera.json for a scene using its GT values
    '''
    out_dict = {}
    calib_params = read_calib_params(calib_params_path)
    for img_id in sorted(gt_dict.keys()):  # assuming filtered recordings
        out_dict[img_id] = calib_params[str(img_id)]
    out_path = os.path.join(out_path, 'scene_camera.json')
    save_scene_camera(out_path, out_dict)
    return out_dict


# TODO: replace mode with dataset split
def build_scene(mode, recording_path, scene_id, gt):
    '''
        Builds all scenes of the dataset
    :param mode: train or test
    :return:
    '''
    scene_id = f'{scene_id:06d}'
    out_path = os.path.join(dp_split['split_path'], scene_id)
    # out_path = os.path.join(out_path, mode, scene_id)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    depth_dir_path = os.path.join(dp_split['split_path'], scene_id, 'depth')
    if not os.path.exists(depth_dir_path):
        os.makedirs(depth_dir_path)
    imgs_out_path = os.path.join(out_path, 'rgb')
    if not os.path.exists(imgs_out_path):
        os.makedirs(imgs_out_path)

    create_scene_gt(scene_id, recording_path,out_path,gt,imgs_out_path)  # scene_id corresponds (initially, berore modifying images with cam id) to the requested images id

def read_csv(recording_path):
    gt_csv = {}
    for cam_id in cams_ids:
        cam_path = os.path.join(recording_path, 'camera_' + str(cam_id))
        out_dict = {}
        with open(os.path.join(cam_path, 'data.csv')) as f:
            reader = csv.reader(f)
            reader_list = list(reader)
            img_ids = []
            content = []
            keys = reader_list[0]
            for row in reader_list[1:]:

                # TODO Remove this from production
                if int(row[keys.index('ObjectID')]) == 5:
                    continue

                img_id = int(os.path.split(row[keys.index('ImageName')])[-1].split('.')[-2].split('_')[-1])
                img_id = f'{img_id:06d}'
                # cam_id = int(os.path.split(row[2])[-1].split('.')[-2].split('_')[-1])
                # out_dict['cam_id'] = cam_id
                obj2vicon_trans = np.asarray(ast.literal_eval(row[keys.index('camToObjTrans')]))
                obj2vicon_rot = np.asarray(ast.literal_eval(row[keys.index('camToObjRot')]))
                camToObjTrans, cam_R_m2c = generate_gt_transformation(cam_id, obj2vicon_trans, obj2vicon_rot)
                entry = {'cam_t_m2c': camToObjTrans, 'cam_R_m2c': cam_R_m2c, 'obj_id': int(row[keys.index('ObjectID')])}

                if img_id not in img_ids:
                    img_ids.append(img_id)
                    content = []

                content.append(entry)
                out_dict[img_id] = content

        gt_csv[cam_id] = out_dict

    return gt_csv

def build_dataset(mode):
    # recording_path = os.path.join(p['datasets_path'], 'recordings', recording)
    list_of_files = glob.glob(os.path.join(p['datasets_path'], 'recordings', '*'))
    recording_path = max(list_of_files, key=os.path.getctime)
    gt_csv = read_csv(recording_path)
    print(recording_path)
    max_len = get_max_len(recording_path)  # accounts for images' filtration
    for i in tqdm(range(max_len)):
        build_scene(mode, recording_path, i, gt_csv)
        # print(f'scene: {i} done')
    # create mode dis - test, train


def get_max_len(recording_path):
    '''
        Gets max num of images per cam to account for invalid images during filtration
    '''
    max_val = 0
    for cam_id in cams_ids:
        imgs_path = os.path.join(recording_path, 'camera_' + str(cam_id), 'images')
        imgs_num = len(glob.glob(os.path.join(imgs_path, '*.png')))
        if imgs_num > max_val:
            max_val = imgs_num
    # print(max_val)
    return max_val


if __name__ == '__main__':
    scene_camera = read_calib_params(calib_params_path)
    build_dataset(dp_split['split_path'])

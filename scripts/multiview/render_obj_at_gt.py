import os
import csv
import ast

import numpy as np
import glob
import shutil

from bop_toolkit_lib import renderer
from scene_processor import generate_depth_image, combine_depth_images
from overlay_imgs import overlay_img


calib_params_csv = './calib_params_all.csv'
models_path = './models'
# obj2vicon_path = './location_tuner_images'
depth_imgs = './location_tuner_imgages/depth_imgs'
im_size = (1296, 1024)


def get_calib_params(calib_params_csv, cam_id):
    with open(calib_params_csv) as f:
        reader = list(csv.reader(f))
        for row in reader:
            if row[0] == str(cam_id):
                mtx = ast.literal_eval(row[1])
                newcameramatrix = ast.literal_eval(row[2])
                dist = ast.literal_eval(row[3])
                roi = ast.literal_eval(row[4])
    return mtx, newcameramatrix, dist, roi


def initialize_renderer(models_path):
    global ren, im_height, im_width, ren_cx_offset, ren_cy_offset

    model_type = None
    im_width = im_size[0]
    im_height = im_size[1]
    ren_width, ren_height = 3 * im_width, 3 * im_height
    ren_cx_offset, ren_cy_offset = im_width, im_height
    ren = renderer.create_renderer(
        ren_width, ren_height, renderer_type='vispy', mode='depth')

    obj_ids = list(range(1,5))
    for obj_id in obj_ids:
        model_path = os.path.join(models_path, 'obj_{obj_id:06d}.ply')
        model_fpath = model_path.format(obj_id=obj_id)
        ren.add_object(obj_id, model_fpath)
    return ren, ren_cx_offset, ren_cy_offset


def get_obj2vicon_transform(req_cam_id, req_img_id, req_obj_id):
    '''
    obj2vicon for a single object
    '''
    csv_path = os.path.join('./location_tuner_images', f'camera_{req_cam_id}')
    with open(csv_path, 'r') as f:
        reader = list(csv.reader(f))
        for row in reader:
            img_id = os.path.split(row[1])[-1].split('.')[-2]
            if int(img_id) == req_img_id and int(row[0]) == req_obj_id:
                return row[2], row[3]


def get_obj2cam_transform(req_cam_id, req_img_id, req_obj_id, cam2vicon_trans, cam2vicon_rot):
    obj2vicon_trans, obj2vicon_rot = get_obj2vicon_transform(req_cam_id, req_img_id, req_obj_id)
    if obj2vicon_trans == None or obj2vicon_rot == None:
        return None


def generate_depth_image(cam_id, obj_id, obj2cam_trans, obj2cam_rot):
    '''
    Generates depth image for a single object
    '''
    _, K, _, _= get_calib_params(calib_params_csv, cam_id)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # Render depth image of the object model in the ground-truth pose.
    depth_gt_large = ren.render_object(
        obj_id, obj2cam_trans, obj2cam_rot,
        fx, fy, cx + ren_cx_offset, cy + ren_cy_offset)['depth']
    depth_gt = depth_gt_large[
               ren_cy_offset:(ren_cy_offset + im_height),
               ren_cx_offset:(ren_cx_offset + im_width)]
    if np.sum(depth_gt) < 100 or np.sum(depth_gt) < 0.9 * np.sum(depth_gt_large):
        return None
    return depth_gt

def combine_depth_images(depth_images, camera_id):
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
    # save_depth(os.path.join(depth_images, ), combined_depth_gt)
    return combined_depth_gt

def get_combined_depth_img(cam_id, img_id, cam2vicon_trans, cam2vicon_rot):
    obj_ids = list(range(1,5))
    depth_imgs = []
    for obj_id in obj_ids:
        obj2cam_trans, obj2cam_rot = get_obj2cam_transform(cam_id, img_id, obj_id, cam2vicon_trans, cam2vicon_rot)
        if obj2cam_trans == None or obj2cam_rot == None:
            continue
        depth_img = generate_depth_image(cam_id, obj_id, obj2cam_trans, obj2cam_rot)
        depth_imgs.append(depth_img)
    combined_depth_img = combine_depth_images(depth_imgs)
    return combined_depth_img



def save_img():
    pass


if __name__ == '__main__':
    initialize_renderer(models_path)
    generate_depth_image()

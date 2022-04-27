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


def generate_depth_image(obj_id, obj_trans, obj_rot, cam_id):
    _, K, _, _= get_calib_params(calib_params_csv, cam_id)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # Render depth image of the object model in the ground-truth pose.
    depth_gt_large = ren.render_object(
        obj_id, obj_trans, obj_rot,
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

def save_img():
    pass


if __name__ == '__main__':
    initialize_renderer(models_path)
    generate_depth_image()

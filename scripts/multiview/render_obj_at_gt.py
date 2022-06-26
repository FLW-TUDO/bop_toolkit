import os
import csv
import ast

import cv2
import numpy as np
from icecream import ic

from bop_toolkit_lib import renderer
from scene_processor import get_homogenous_form, invert_homog_transfrom
from scipy.spatial.transform import Rotation as R
from utils import convert_to_rot_mat
from overlay_imgs import overlay_img


'''
The script renders the object models at their ground truth positions with respect to the camera using stored
absolute location of the objects and the camera parameters
'''

class DepthGenerator:
    def __init__(self, calib_params_csv_path, models_path, im_size, obj_ids):
        self.calib_params_csv_path = calib_params_csv_path
        self.models_path = models_path
        self.im_size = im_size
        self.obj_ids = obj_ids

    def get_calib_params(self, csv_path, cam_id):
        with open(csv_path, 'r') as f:
            reader = list(csv.reader(f))
            for row in reader:
                if row[0] == str(cam_id):
                    mtx = ast.literal_eval(row[1])
                    newcameramatrix = ast.literal_eval(row[2])
                    dist = ast.literal_eval(row[3])
                    roi = ast.literal_eval(row[4])
        return mtx, newcameramatrix, dist, roi

    def initialize_renderer(self):
        im_width = self.im_size[0]
        im_height = self.im_size[1]
        ren_width, ren_height = 3 * im_width, 3 * im_height
        ren = renderer.create_renderer(
            ren_width, ren_height, renderer_type='vispy', mode='depth')

        for obj_id in self.obj_ids:
            model_path = os.path.join(self.models_path, 'obj_{obj_id:06d}.ply')
            model_fpath = model_path.format(obj_id=obj_id)
            ren.add_object(obj_id, model_fpath)
        return ren

    def get_obj2vicon_transforms(self, req_cam_id, req_img_id):
        '''
        Provides translation and rotation vectors for all objects in a particular image as a dict.
        Object ID is used as a key
        '''
        csv_path = os.path.join('./location_tuner_images', f'camera_{req_cam_id}', 'data.csv')
        with open(csv_path, 'r') as f:
            reader = list(csv.reader(f))
            obj2vicon_transforms = {}
            for row in reader:
                img_id = os.path.split(row[1])[-1].split('.')[-2]
                if str(img_id) == str(req_img_id):
                    obj2vicon_trans = ast.literal_eval(row[2])
                    obj2vicon_rot = ast.literal_eval(row[3])
                    obj2vicon_transforms[str(row[0])] = [obj2vicon_trans, obj2vicon_rot]
        return obj2vicon_transforms

    def get_obj2cam_transform(self, req_cam_id, req_img_id, cam2vicon_trans, cam2vicon_rot):
        obj2vicon_transforms = self.get_obj2vicon_transforms(req_cam_id, req_img_id)
        obj2cam_transforms = {}
        for obj_id in list(obj2vicon_transforms.keys()):
            obj2vicon_rot = R.from_euler('XYZ', obj2vicon_transforms[obj_id][1], degrees=False)
            obj2vicon_rot_mat = obj2vicon_rot.as_matrix()

            obj2vicon_transform = get_homogenous_form(obj2vicon_rot_mat, obj2vicon_transforms[obj_id][0])
            ic(obj2vicon_transform)

            cam2vicon_rot_mat = convert_to_rot_mat(cam2vicon_rot)
            cam2vicon_transform = get_homogenous_form(cam2vicon_rot_mat, cam2vicon_trans)
            obj2cam_transform = invert_homog_transfrom(cam2vicon_transform).dot(obj2vicon_transform)
            ic(obj2cam_transform)
            obj2cam_trans = obj2cam_transform[0:3, 3]
            obj2cam_rot = obj2cam_transform[0:3, 0:3]
            obj2cam_transforms[obj_id] = [obj2cam_trans, obj2cam_rot]
        return obj2cam_transforms

    def generate_depth_images(self, ren, cam_id, obj2cam_transforms):
        _, K, _, _ = self.get_calib_params(self.calib_params_csv_path, cam_id)
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
        ren_cx_offset, ren_cy_offset = self.im_size[0], self.im_size[1]
        depth_imgs = []
        # Render depth image of the object model in the ground-truth pose.
        for obj_id in list(obj2cam_transforms.keys()):
            depth_gt_large = ren.render_object(
                int(obj_id), obj2cam_transforms[obj_id][1], obj2cam_transforms[obj_id][0],
                fx, fy, cx + ren_cx_offset, cy + ren_cy_offset)['depth']
            depth_gt = depth_gt_large[
                       ren_cy_offset:(ren_cy_offset + self.im_size[1]),
                       ren_cx_offset:(ren_cx_offset + self.im_size[0])]
            depth_imgs.append(depth_gt)
        # if np.sum(depth_gt) < 100 or np.sum(depth_gt) < 0.9 * np.sum(depth_gt_large):
        #     return None
        return depth_imgs

    def combine_depth_images(self, depth_images):
        combined_depth_gt = np.zeros((self.im_size[1], self.im_size[0]))
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

    def get_combined_depth_img(self, cam_id, img_id, ren, cam2vicon_trans, cam2vicon_rot):
        obj2cam_transforms = self.get_obj2cam_transform(cam_id, img_id, cam2vicon_trans, cam2vicon_rot)
        depth_imgs = self.generate_depth_images(ren, cam_id, obj2cam_transforms)
        combined_depth_img = self.combine_depth_images(depth_imgs)
        combined_depth_img[combined_depth_img > 0] = 255 # a white mask is generated for each object
        combined_depth_img = combined_depth_img.astype(np.uint8) # convert from float64 to uint8
        combined_depth_img = cv2.merge([combined_depth_img, combined_depth_img, combined_depth_img]) # convert to RGB
        return combined_depth_img

    def save_img(self, img, img_id, path):
        cv2.imwrite((os.path.join(path, f'{img_id}_depth.png')), img)



if __name__ == '__main__':
    calib_params_csv = './calib_params_all.csv'
    models_path = './models'
    im_size = (1296, 1024)
    obj_ids = list(range(1, 5))
    depth_imgs_path = './location_tuner_images/depth_imgs'

    depth_gen = DepthGenerator(calib_params_csv, models_path, im_size, obj_ids)
    ren = depth_gen.initialize_renderer()

    cam_id = 6
    img_id = 123
    depth_img = depth_gen.get_combined_depth_img(cam_id=cam_id, img_id=img_id, ren=ren, cam2vicon_trans=[526.8575593558775, -5600.212355673482, 4850],
                                       cam2vicon_rot=[-143.000,  0.0000000, -180])
    depth_gen.save_img(depth_img, img_id, depth_imgs_path)

    rgb_img = cv2.imread(f'./location_tuner_images/camera_{cam_id}/images/{img_id}.png')
    overlay_img(rgb_img, depth_img, cam_id)
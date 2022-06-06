import numpy as np
import ast
import csv
from render_obj_at_gt import DepthGenerator
from utils import convert_to_rot_mat, convert_to_euler
from overlay_imgs import overlay_img
import glob
import os
import cv2



intial_cam_locations_csv_file = 'initial_cam_locations.csv'
final_cam_locations_csv_file = './accepted_cam_locations.csv'


def get_initial_cam_location(cam_id):
    with open(intial_cam_locations_csv_file) as f:
        reader = list(csv.reader(f))
        for row in reader:
            if int(row[0])==cam_id:
                cam_trans = ast.literal_eval(row[1])
                cam_rot_mat = ast.literal_eval(row[2])
    return cam_trans, cam_rot_mat

def tune_cam_location(depth_generator, cam_id, trans_x_range, trans_y_range, trans_z_range, rot_x_range, rot_y_range, rot_z_range):
    '''
    trans/rot_range: desired amount of offsetting to the initial camera location
    '''
    ren = depth_generator.initialize_renderer()
    trans_x_inc = 1
    trans_y_inc = 1
    trans_z_inc = 1
    rot_x_inc = -1
    rot_y_inc = 1
    rot_z_inc = 1

    cam_initial_trans, cam_intial_rot_mat = get_initial_cam_location(cam_id)
    cam_initial_rot = convert_to_euler(cam_intial_rot_mat)
    cam_dir = f'./location_tuner_images/camera_{cam_id}/images'

    for img in glob.glob(os.path.join(cam_dir, '*.png')):
        for trans_x in range(trans_x_range[0], trans_x_range[1], trans_x_inc):
            for trans_y in range(trans_y_range[0], trans_y_range[1], trans_y_inc):
                for trans_z in range(trans_z_range[0], trans_z_range[1], trans_z_inc):
                    for rot_x in range(rot_x_range[0], rot_x_range[1], rot_x_inc):
                        for rot_y in range(rot_y_range[0], rot_y_range[1], rot_y_inc):
                            for rot_z in range(rot_z_range[0], rot_z_range[1], rot_z_inc):
                                new_cam_trans = [cam_initial_trans[0]+trans_x, cam_initial_trans[1]+trans_y, cam_initial_trans[2]+trans_z]
                                new_cam_rot = [cam_initial_rot[0]+rot_x, cam_initial_rot[1]+rot_y, cam_initial_rot[2]+rot_z]
                                print(new_cam_trans)
                                print(new_cam_rot)
                                img_id = os.path.split(img)[-1].split('.')[-2]
                                depth_img = depth_gen.get_combined_depth_img(cam_id=cam_id, img_id=img_id, ren=ren,
                                                                             cam2vicon_trans=new_cam_trans,
                                                                             cam2vicon_rot=new_cam_rot)
                                # convert depth image from 1 channel float64 to 3 channels uint8
                                depth_img = depth_img.astype(np.uint8)
                                depth_img = cv2.merge([depth_img, depth_img, depth_img])
                                rgb_img = cv2.imread(img)
                                overlay_img(rgb_img, depth_img, cam_id)
                                # depth_generator.save_img(depth_img, './location_tuner_images/depth_imgs')


def write_to_csv(out_csv_file, data_list):
    with open(out_csv_file, 'w') as f:  #'a' for appending entries
        writer = list(csv.writer(f))
        writer.writerow(data_list)


if __name__ == '__main__':
    calib_params_csv = './calib_params_all.csv'
    models_path = './models'
    im_size = (1296, 1024)
    obj_ids = list(range(1, 5))

    depth_gen = DepthGenerator(calib_params_csv, models_path, im_size, obj_ids)
    tune_cam_location(depth_generator=depth_gen, cam_id=6, trans_x_range=[-2, 2], trans_y_range=[-2, 2],
                      trans_z_range=[-2, 3], rot_x_range=[-2, -3], rot_y_range=[-2, 2], rot_z_range=[-2, 2])
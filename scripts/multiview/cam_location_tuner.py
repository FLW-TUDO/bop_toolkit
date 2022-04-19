import numpy as np
import ast
import csv
from scipy.spatial.transform import Rotation as R
from overlay_imgs import overlay_img
from scene_processor import build_dataset
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

def tune_cam_location(cam_id, trans_x_range, trans_y_range, trans_z_range, rot_x_range, rot_y_range, rot_z_range):
    trans_x_inc = 1
    trans_y_inc = 1
    trans_z_inc = 1
    rot_x_inc = 1
    rot_y_inc = 1
    rot_z_inc = 1

    cam_initial_trans, cam_intial_rot_mat = get_initial_cam_location(cam_id)
    cam_initial_rot = convert_to_euler(cam_intial_rot_mat)
    cam_dir = f'../images/camera_{cam_id}'

    for img in glob.glob(os.path.join(cam_dir, '*.png')):
        for trans_x in range(trans_x_range[0], trans_x_range[1], trans_x_inc):
            for trans_y in range(trans_y_range[0], trans_y_range[1], trans_y_inc):
                for trans_z in range(trans_z_range[0], trans_z_range[1], trans_z_inc):
                    for rot_x in range(rot_x_range[0], rot_x_range[1], rot_x_inc):
                        for rot_y in range(rot_y_range[0], rot_y_range[1], rot_y_inc):
                            for rot_z in range(rot_z_range[0], rot_z_range[1], rot_z_inc):
                                new_cam_trans = [cam_initial_trans[0]+trans_x, cam_initial_trans[1]+trans_y, cam_initial_trans[2]+trans_z]
                                new_cam_rot = [cam_initial_rot[0]+rot_x, cam_initial_rot[1]+rot_y, cam_initial_rot[2]+rot_z]
                                build_dataset('./images')
                                # depth_img = cv2.imread(...)
                                # overlay_img(img, depth_img)


def write_to_csv(out_csv_file, data_list):
    with open(out_csv_file, 'w') as f:  #'a' for appending entries
        writer = list(csv.writer(f))
        writer.writerow(data_list)


def convert_to_euler(rot_mat):
    angle = R.from_matrix(rot_mat)
    euler_angle = angle.as_euler("XYZ", degrees=True)
    return euler_angle

def convert_to_rot_mat(euler_angle):
    angle = R.from_euler("XYZ", euler_angle, degrees=True)
    rot_mat = angle.as_matix()
    return rot_mat
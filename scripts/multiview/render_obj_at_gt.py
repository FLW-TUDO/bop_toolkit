import os
import csv
import ast

import numpy as np
import glob
import shutil

from bop_toolkit_lib import renderer
from overlay_imgs import overlay_img


calib_params_csv = './calib_params_all.csv'
models_path = './models'
obj2vicon_path = './location_tuner_images'


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


def read_obj2vicon_csv(obj2vicon_gt_path, cam_id):
    data_csv = os.path.join(obj2vicon_path, f'camera_{cam_id}', 'data.csv')
    obj2vicon_dict = {}
    with open(data_csv) as f:
        reader = list(csv.reader(f))
        for row in reader:
            obj2vicon_



def get_obj2cam_




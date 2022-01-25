#from scripts import calc_gt_info
#from scripts import calc_gt_masks
#from scripts.multiview import depth_generator
import os
import csv
import ast

import numpy as np

from bop_toolkit_lib.inout import save_scene_gt

cams = [0]
base_path = '/home/hazem/projects/multi_view_dataset'
recording = 'recordings_test/11_11_11 08_11_2021'
recording_path = os.path.join(base_path, recording)

def read_data(path):
    pass

def read_gt_poses(path, cams):
    '''Reads csv file and saves it to a dict in intermediary format'''
    out_dict = {}
    for cam in cams:
        try:
            cam_dir = os.path.join(path, 'camera_'+str(cam))
        except:
            print(f'Camera_{cam} directory Not found!')
        with open (os.path.join(cam_dir, 'data.csv')) as f:
            reader = csv.reader(f)
            reader_list = list(reader)
            #print(reader_list)
            img_ids = []
            content = []
            for row in reader_list[1:]:
                img_id = int(os.path.split(row[2])[-1].split('.')[-2].split('_')[-1])
                img_id = f'{img_id:06d}'
                out_dict['cam_id'] = cam
                entry = {'cam_t_m2c': np.asarray(ast.literal_eval(row[3])), 'cam_R_m2c': np.asarray(ast.literal_eval(row[4])), 'obj_id': int(row[0])}
                if img_id not in img_ids:
                    content = []
                content.append(entry)
                out_dict[img_id] = content
                img_ids.append(img_id)
                print(out_dict)
    return out_dict


def create_scene_gt(data):
    '''Creates final json file from intermediary format'''
    out_dict = {}
    pass


def test_dict():
    arr_1 = np.asarray([[0.96196665, 0.00844651, 0.27303804], [-0.25964929, -0.28227396, 0.92352759], [0.08487223, -0.95929701, -0.26934539]])
    arr_2 = np.asarray([[0.72359593, -0.00844647, -0.69017283], [-0.66040191, 0.28227403, -0.69583779], [0.20069552, 0.95929699, 0.19867456]])
    arr_3 = np.asarray([87.76545575, 42.6672001, 754.33191936])
    arr_4 = np.asarray([24.18985221, -45.02596545, 754.14068623])
    test = {"0": [{"cam_R_m2c": arr_1, "cam_t_m2c": arr_3, "obj_id": 5}, {"cam_R_m2c": arr_2, "cam_t_m2c": arr_4, "obj_id": 8}]}
    print(save_scene_gt('/home/hazem/projects/test.json', test))




def build_scene(recording, vicon_data):
    pass

def save_img(img):
    pass


if __name__ == '__main__':
    read_gt_poses(recording_path, cams)
    #test_dict()
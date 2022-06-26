import json
import numpy as np
import ast
import csv
from render_obj_at_gt import DepthGenerator
from utils import convert_to_rot_mat, convert_to_euler
from overlay_imgs import overlay_img
import glob
import os
import cv2
import copy


intial_cam_locations_csv_file = 'initial_cam_locations.csv'
final_cam_locations_csv_file = './accepted_cam_locations.csv'
resutls_csv_file = './poses_res.csv'


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
    trans_x_inc = 5
    trans_y_inc = 5
    trans_z_inc = 5
    rot_x_inc = 1
    rot_y_inc = 1
    rot_z_inc = 1

    cam_initial_trans, cam_intial_rot_mat = get_initial_cam_location(cam_id)
    cam_initial_rot = convert_to_euler(cam_intial_rot_mat)
    cam_dir = f'./location_tuner_images/camera_{cam_id}/masks' #iterate over crafted masks of objects

    cand_poses = {}
    meta_data = {}
    iter = 0
    checkpt_val = 100
    hit_counter = 0

    with open(resutls_csv_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['cam_id', 'trans_x_range', 'trans_y_range', 'trans_z_range', 'rot_x_range', 'rot_y_range',
                        'rot_z_range', 'trans_x_inc', 'trans_y_inc', 'trans_z_inc'])
        writer.writerow([cam_id, trans_x_range, trans_y_range, trans_z_range, rot_x_range, rot_y_range, rot_z_range,
                        trans_x_inc, trans_y_inc, trans_z_inc])
        writer.writerow([])
        writer.writerow(['img_path', 'score', 'trans', 'rot'])


    # for img in glob.glob(os.path.join(cam_dir, '*.png')):
    for img in glob.glob(os.path.join(cam_dir, '123.png')):
        for trans_x in range(trans_x_range[0], trans_x_range[1], trans_x_inc):
            for trans_y in range(trans_y_range[0], trans_y_range[1], trans_y_inc):
                for trans_z in range(trans_z_range[0], trans_z_range[1], trans_z_inc):
                    for rot_x in range(rot_x_range[0], rot_x_range[1], rot_x_inc):
                        for rot_y in range(rot_y_range[0], rot_y_range[1], rot_y_inc):
                            for rot_z in range(rot_z_range[0], rot_z_range[1], rot_z_inc):
                                new_cam_trans = [cam_initial_trans[0]+trans_x, cam_initial_trans[1]+trans_y, cam_initial_trans[2]+trans_z]
                                new_cam_rot = [cam_initial_rot[0]+rot_x, cam_initial_rot[1]+rot_y, cam_initial_rot[2]+rot_z]
                                # print(new_cam_trans)
                                # print(new_cam_rot)
                                img_id = os.path.split(img)[-1].split('.')[-2]

                                rgb_img = cv2.imread(img)
                                rgb_img = rgb_img[:,:,0]
                                # cv2.imshow('rgb', rgb_img);cv2.waitKey(0);cv2.destroyAllWindows()

                                depth_img = depth_generator.get_combined_depth_img(cam_id=cam_id, img_id=img_id, ren=ren,
                                                                             cam2vicon_trans=new_cam_trans,
                                                                             cam2vicon_rot=new_cam_rot)
                                # convert depth image from 1 channel float64 to 3 channels uint8
                                # depth_img[depth_img > 0] = 255
                                # depth_img = depth_img.astype(np.uint8)
                                # cv2.imshow('depth', depth_img); cv2.waitKey(0);cv2.destroyAllWindows()

                                depth_img_sc =  depth_img[:,:,0] # single channel depth image
                                contours, hierarchy = cv2.findContours(image=depth_img_sc, mode=cv2.RETR_TREE,
                                                                       method=cv2.CHAIN_APPROX_NONE) # find contours works with single channel images
                                # for i in range(len(contours)):
                                depth_contoured = copy.deepcopy(depth_img_sc)
                                cv2.drawContours(depth_contoured, contours, -1, 255, thickness=-1, hierarchy=hierarchy, maxLevel=1)
                                # cv2.imshow('contours', depth_contoured); cv2.waitKey(0);cv2.destroyAllWindows()

                                depth_res = np.bitwise_and(rgb_img, depth_contoured)
                                # print(np.where(depth_res==True))
                                # cv2.imshow('res', depth_res); cv2.waitKey(0);cv2.destroyAllWindows()

                                x = rgb_img == depth_contoured
                                score = len(x[x == True])/len(x.flatten())
                                if score > 0.983:
                                    hit_counter+=1
                                    res_name = f'{int(score*10000)}_{img_id}_{hit_counter}'
                                    img_path = f'./location_tuner_images/res_imgs/{res_name}.png'
                                    data = [img_path, score, new_cam_trans, new_cam_rot]
                                    save_img(img_path, depth_res)
                                    if (iter % checkpt_val == 0):
                                        write_to_csv(resutls_csv_file, data)
                                print(f'Total Hits: {hit_counter}')
                                iter+=1
                                print(f'Total Iterations: {iter}')


def write_to_csv(csv_file_path, data_list):
    with open(csv_file_path, 'a') as f:  #'a' for appending entries
        writer = csv.writer(f)
        writer.writerow(data_list)

def save_img(path, img):
    cv2.imwrite(os.path.join(path), img)


def visualize_cam_location(poses_csv_path, req_pose_id, cam_id, depth_generator, display=True):
    '''
    Saves and visualizes the resulting camera poses from the tuning process. It generates the depth image according to
    each accepted pose and overlays it on the 'original' image
    '''
    ren = depth_generator.initialize_renderer()
    with open(poses_csv_path, 'r') as f:
        reader = list(csv.reader(f))
        for row in reader[2:]:
            pose_id = os.path.split(row[0])[-1].split('.')[-2]
            img_id = pose_id.split('_')[1]
            if pose_id == req_pose_id:
                depth_img = depth_gen.get_combined_depth_img(cam_id=cam_id, img_id=img_id, ren=ren,
                                                                   cam2vicon_trans=row[2],
                                                                   cam2vicon_rot=row[3])
    rgb_img = f'./location_tuner_images/camera_{cam_id}/{img_id}.png'
    overlaid_img = overlay_img(depth_img, rgb_img)
    save_img(f'./location_tuner_images/res_imgs/overlaid/{pose_id}_overlaid.png', overlaid_img)
    if display:
        cv2.imshow(f'{pose_id}_{row[2]}_{row[3]}', overlaid_img)
        key = cv2.waitKey(0)
        if key == 32:  # press 'q' to exit
            cv2.destroyAllWindows()




if __name__ == '__main__':
    calib_params_csv = './calib_params_all.csv'
    models_path = './models'
    im_size = (1296, 1024)
    obj_ids = list(range(1, 5))

    depth_gen = DepthGenerator(calib_params_csv, models_path, im_size, obj_ids)
    tune_cam_location(depth_generator=depth_gen, cam_id=6, trans_x_range=[-10, 10], trans_y_range=[-10, 10],
                      trans_z_range=[-10, 10], rot_x_range=[-2, 2], rot_y_range=[-2, 2], rot_z_range=[-2, 2])

    # visualize_cam_location(resutls_csv_file, )

    # ren = depth_gen.initialize_renderer()
    # depth_img = depth_gen.get_combined_depth_img(cam_id=6, img_id=123, ren=ren,
    #                                                    cam2vicon_trans=[516.8575593558775, -5610.212355673482, 4850],
    #                                                    cam2vicon_rot=[-143.00000113389902, 1.0, -182.0])
    # cv2.imwrite('./location_tuner_images/depth_imgs/test.png', depth_img)
    # depth_img = cv2.imread('./location_tuner_images/depth_imgs/test.png')
    # rgb_img = cv2.imread('/home/athos/Schreibtisch/123.png')
    # # mask_img = '/media/athos/DATA-III/projects/bop_toolkit/scripts/multiview/location_tuner_images/res_imgs/9856_12.png'
    # overlay_img(rgb_img, depth_img)

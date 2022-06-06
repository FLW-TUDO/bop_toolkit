import csv
import ast
import numpy as np
import cv2
import os


calib_params_csv_file = './calib_params_all.csv'

def get_calib_params(csv_file, cam_id):
    with open(csv_file) as f:
        reader = csv.reader(f)
        for row in list(reader):
            if row[0] == str(cam_id):
                mtx = np.array(ast.literal_eval(row[1]))
                newcameramtx = np.array(ast.literal_eval(row[2]))
                dist = np.array(ast.literal_eval(row[3]))
                roi = np.array(ast.literal_eval(row[4]))
                return mtx, dist, newcameramtx, roi


def undistort_img(img, mtx, dist, newcameramtx, roi):
    undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    # x, y, w, h = roi
    # undistorted_img = undistorted_img[y:y+h, x:x+w]
    return undistorted_img

def overlay_img(rgb_img, depth_img, cam_id, undistort=False):
    # rgb_img = cv2.imread(rgb_img_path)
    # depth_img = cv2.imread(depth_img_path)
    if undistort:
        mtx, dist, newcameramtx, roi = get_calib_params(calib_params_csv_file, cam_id)
        rgb_img = undistort_img(rgb_img, mtx, dist, newcameramtx, roi)
    res_img = cv2.add(rgb_img, depth_img) # cv2.add clips the values beyond 255 unlike numpy
    while True:
        cv2.imshow('overlay', res_img)
        key = cv2.waitKey(0)
        if key == 32: #press 'q' to exit
            break
        elif key == 115: #press 's' to save image
            path = os.path.normpath(rgb_img_path).split('/')
            img_id = int(path[-1].split('.')[-2])
            scene_id = int(path[-3])
            img_name = f'{scene_id:06d}_{img_id:06d}'
            cv2.imwrite(f'./overlaid_imgs/{img_name}.png', res_img)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    rgb_img_path = cv2.imread('/media/athos/DATA-III/projects/000000/rgb/000001.png')
    depth_img_path = cv2.imread('/media/athos/DATA-III/projects/000000/depth/000001.png')
    overlay_img(rgb_img_path, depth_img_path, cam_id=1, undistort=False) # only calibration parameters used in model projection
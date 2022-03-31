import numpy as np
import cv2


mtx = np.array([[861.59229782,   0.        , 629.39824799],
                [  0.        , 861.13244676, 526.14365127],
                [  0.        ,   0.        ,   1.        ]])
newcameramtx = np.array([[799.56201172,   0.        , 628.6002021 ],
                         [  0.        , 799.69110107, 526.90323354],
                         [  0.        ,   0.        ,   1.        ]])
dist = np.array([[-0.16757778,  0.1080619 ,  0.00078127, -0.00021056, -0.01003737]])
roi = (5, 11, 1287, 1001)

cam_locations_csv_file = '../cam_locations.csv'  # final csv file
calib_params_csv_file = '../calib_params_all.csv'

def undistort(img_path, mtx, dist, newcameramtx, roi, visualize):
    '''
    @param imgs_path: directory of images to be undistorted
    @type imgs_path: str
    ...
    @return: image matrix
    @rtype: np array
    '''
    print('Undistorting images')
    img = cv2.imread(img_path)
    undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    # x, y, w, h = roi
    # undistorted_img = undistorted_img[y:y+h, x:x+w]
    return undistorted_img

img_rgb = '/media/athos/DATA-III/projects/multiview_dataset/mv/test/000000/rgb/000001.png'
img_depth = '/media/athos/DATA-III/projects/multiview_dataset/mv/test/000000/depth/000001.png'

#img_rgb = undistort(img_rgb, mtx, dist, newcameramtx, roi, False)
img_rgb = cv2.imread(img_rgb)
img_depth = cv2.imread(img_depth)

#img_depth = np.fliplr(img_depth) ######

print(np.shape(img_rgb))
print(np.shape(img_depth))

res_img = np.add(img_rgb, img_depth)
print(np.shape(res_img))

while True:
    cv2.imshow('overlay', res_img)
    key = cv2.waitKey(0)
    if key == 32:
        break

    cv2.destroyAllWindows()

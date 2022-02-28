import numpy as np
import cv2


mtx = np.array([[859.90825492,   0.        , 654.08289216],
     [  0.        , 859.12593446, 497.68761377],
     [  0.        ,   0.        ,   1.        ]])
newcameramtx = np.array([[795.02294922,   0.        , 652.47181967],
              [  0.        , 796.08612061, 493.5983575  ],
              [  0.        ,   0.        ,   1.         ]])
dist = np.array([[-0.164644  ,  0.09254984, -0.00237873, -0.0005464 ,  0.00101041]])
roi = (5, 11, 1284, 1000)

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

# img_rgb = undistort(img_rgb, mtx, dist, newcameramtx, roi, False)
img_rgb = cv2.imread(img_rgb)
img_depth = cv2.imread(img_depth)

img_depth = np.fliplr(img_depth) ######

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

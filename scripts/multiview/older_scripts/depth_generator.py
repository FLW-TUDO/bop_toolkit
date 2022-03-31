import numpy as np
import cv2
from bop_toolkit_lib import renderer
from bop_toolkit_lib import inout
from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import misc
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


p = {
  # See dataset_params.py for options.
  'dataset': 'mv',

  # Dataset split. Options: 'train', 'val', 'test'.
  'dataset_split': 'test',

  # Dataset split type. None = default. See dataset_params.py for options.
  'dataset_split_type': None,

  # Type of the renderer.
  'renderer_type': 'vispy',  # Options: 'vispy', 'cpp', 'python'.

  # Folder containing the BOP datasets.
  'datasets_path': config.datasets_path,
}

model_type = None
dp_model = dataset_params.get_model_params(
  p['datasets_path'], p['dataset'], model_type)

# Load dataset parameters.
dp_split = dataset_params.get_split_params(
  p['datasets_path'], p['dataset'], p['dataset_split'], p['dataset_split_type'])

# The renderer has a larger canvas for generation of masks of truncated objects.
im_width, im_height = dp_split['im_size']
ren_width, ren_height = 3 * im_width, 3 * im_height
ren_cx_offset, ren_cy_offset = im_width, im_height
ren = renderer.create_renderer(
  ren_width, ren_height, p['renderer_type'], mode='depth')

for obj_id in dp_model['obj_ids']:
  model_fpath = dp_model['model_tpath'].format(obj_id=obj_id)
  ren.add_object(obj_id, model_fpath)

scene_ids = dataset_params.get_present_scene_ids(dp_split)
#for scene_id in tqdm(scene_ids):
for scene_id in tqdm(scene_ids):
  scene_id_full = f'{scene_id:06d}'
  depth_dir_path = os.path.join(dp_split['split_path'], scene_id_full, 'depth')
  if not os.path.exists(depth_dir_path):
    os.makedirs(depth_dir_path)
  # Load scene info and ground-truth poses.
  scene_camera = inout.load_scene_camera(
    dp_split['scene_camera_tpath'].format(scene_id=scene_id))
  scene_gt = inout.load_scene_gt(
    dp_split['scene_gt_tpath'].format(scene_id=scene_id))

  im_ids = sorted(scene_gt.keys())
  for im_counter, im_id in enumerate(im_ids):
    # if im_counter % 100 == 0:
    #   misc.log(
    #     'Calculating GT info - dataset: {} ({}, {}), scene: {}, im: {}'.format(
    #       p['dataset'], p['dataset_split'], p['dataset_split_type'], scene_id,
    #       im_id))

    K = scene_camera[im_id]['cam_K']
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # im_size = (depth.shape[1], depth.shape[0])

    depth_imgs = []
    for gt_id, gt in enumerate(scene_gt[im_id]):
      # Render depth image of the object model in the ground-truth pose.
      depth_gt_large = ren.render_object(
        gt['obj_id'], gt['cam_R_m2c'], gt['cam_t_m2c'],
        fx, fy, cx + ren_cx_offset, cy + ren_cy_offset)['depth']
      depth_gt = depth_gt_large[
                   ren_cy_offset:(ren_cy_offset + im_height),
                   ren_cx_offset:(ren_cx_offset + im_width)]
      if np.sum(depth_gt) < 100 or np.sum(depth_gt) < 0.9 * np.sum(depth_gt_large):
        print(gt['obj_id'], 'not in image')
        continue
      depth_imgs.append(depth_gt)

    im_shape = np.shape(depth_gt)
    combined_depth_gt = np.zeros((im_shape[0], im_shape[1]))
    #print('Combining depth images..')
    for depth_img in depth_imgs:
      rows, columns = np.where(depth_img > 0)
      for i, j in zip(rows, columns):
        if (combined_depth_gt[i, j] == 0):  # updated by first non-zero value in any depth image
          combined_depth_gt[i, j] = depth_img[i, j]

    combined_depth_gt = np.flipud(combined_depth_gt)
    combined_depth_gt = np.fliplr(combined_depth_gt)

    inout.save_depth(dp_split['depth_tpath'].format(scene_id=scene_id, im_id=im_id), combined_depth_gt)
    #combined_depth_gt = np.asarray(combined_depth_gt, dtype=np.uint16)
    #cv2.imwrite(dp_split['depth_tpath'].format(scene_id=scene_id, im_id=im_id), combined_depth_gt)
    # cv2.imwrite('/home/hazem/projects/depth_test.png', depth_gt)


    # img = cv2.imread(dp_split['depth_tpath'].format(scene_id=scene_id, im_id=im_id), -1) # read image as is
    # print(np.shape(img))
    # normed_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # color = cv2.applyColorMap(normed_img, cv2.COLORMAP_JET)
    # cv2.imshow('depth img', normed_img)
    # key = cv2.waitKey(0)
    # if key == 'q':
    #   cv2.destroyAllWindows()
    #
    # print(img.dtype)
    # vals = []
    # for row in normed_img:
    #   for elem in row:
    #     if elem != 0:
    #       print(f'Pixel value is: {elem}')
    #       vals.append(elem)
    #
    # num_bins = 20
    # n, bins, patches = plt.hist(vals, num_bins, facecolor='blue', alpha=0.5)
    # plt.show()




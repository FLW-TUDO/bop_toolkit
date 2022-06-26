import open3d as o3d
import copy
import numpy as np
from scipy.spatial.transform import Rotation as R

# setup visualizer
viewer = o3d.visualization.Visualizer()
viewer.create_window()
opt = viewer.get_render_option()
opt.point_size = 1
#opt.show_coordinate_frame = True

cloud_path = '/home/gouda/segmentation/datasets/ML2R_segmentation_dataset_BOP_format/ml2r/models/cocacola_filtered.ply'
original_pcd = o3d.io.read_triangle_mesh(cloud_path)

# remove outliers and down sample to get a better OBB
OBB_original = original_pcd.get_oriented_bounding_box()
pcd_aligned = copy.deepcopy(original_pcd)
pcd_aligned.translate(np.array([0,0,0]), relative=False)
#p = R.from_matrix(OBB_original.R)
#print(p.as_matrix())
#pcd_aligned.rotate(p.inv().as_matrix())

# rotate object to have main face facing front
rot_mat = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, 0])
pcd_aligned.rotate(rot_mat)

# calculate OBB after aligning
OBB_aligned = pcd_aligned.get_oriented_bounding_box()

# save aligned cloud
o3d.io.write_triangle_mesh(cloud_path[:-4] + '_aligned.ply', pcd_aligned, write_ascii=True)

o3d.visualization.draw_geometries([o3d.geometry.TriangleMesh.create_coordinate_frame(size=200), OBB_aligned, pcd_aligned])
from scipy.spatial.transform import Rotation as R


def convert_to_euler(rot_mat):
    angle = R.from_matrix(rot_mat)
    euler_angle = angle.as_euler("XYZ", degrees=True)
    return euler_angle

def convert_to_rot_mat(euler_angle):
    angle = R.from_euler("XYZ", euler_angle, degrees=True)
    rot_mat = angle.as_matrix()
    return rot_mat
import numpy as np
from scipy.spatial.transform import Rotation as R

camera_id = 1
new_angle = np.array([146.89910137,   2.59198324,   1.34976179])

camera_data = {
    0: {'pos': [-6686.742955566662, 5959.497944234329, 4850], 'rot': [[0.9941231956636261, -0.012012593698220785, -0.10758610243047602], [-0.06349963607862076, -0.8695996704150286, -0.489657236678838], [-0.08767478577916309, 0.4936112952584395, -0.865251998629222]]},
    1: {'pos': [-2033.8025950407948, 5869.235170710448, 4670.520733149707], 'rot': [[0.9993366472806696, -0.023546516179374647, 0.02778177419902314], [-0.004564935674398269, -0.8378350972363092, -0.545904305901056], [0.036130690058725226, 0.545415356783103, -0.8373868053779218]]},
    2: {'pos': [2290.77197003578, 5895.690499337283, 4850], 'rot': [[0.9914105998172326, -0.1104064133066334, 0.07011024512009781], [-0.05319920774886086, -0.8301467325181208, -0.5550011232280356], [0.11947747429925398, 0.546504186983277, -0.8288898037404121]]},
    3: {'pos': [7371.4194775151345, 6182.77617805888, 4850], 'rot': [[0.9606330757300722, -0.11694063014923445, -0.25200988638083566], [-0.2522154839643561, -0.747452185648069, -0.6145751213794374], [-0.11649653841212351, 0.6539419845752562, -0.7475215296884553]]},
    4: {'pos': [9108.99598421382, -5762.115808475929, 4850], 'rot': [[-0.9439490669433342, -0.1908277801821743, -0.2693416368249673], [-0.3132047862680901, 0.7754241165988348, 0.5482884288909231], [0.10422533701345771, 0.6019154406622749, -0.7917290454548013]]},
    5: {'pos': [4710.4921159219875, -5962.684706935452, 4850], 'rot': [[-0.9977634332902533, 0.059491912020423025, -0.03047693542665281], [0.017177670108667366, 0.6688328649966196, 0.7432143206034523], [0.06459921701138424, 0.7410285494531015, -0.6683589081152043]]},
    6: {'pos': [526.8575593558775, -5600.212355673482, 4850], 'rot': [[-1.0000000,  0.0000000, -0.0000000], [0.0000000,  0.7986355,  0.6018150], [ 0.0000000,  0.6018150, -0.7986355]]},
    7: {'pos': [-4071.0293359016882, -5713.337420494916, 4850], 'rot': [[-0.9962205744824704, 0.0856512050665356, -0.014437383712686474], [0.05698702792596087, 0.7699524046372579, 0.6355515504201606], [0.06555185448164555, 0.6323267870839845, -0.7719234344869073]]},
}

data = camera_data[camera_id]
original = R.from_matrix(data["rot"])
original_angle = original.as_euler("XYZ", degrees=True)

print(f"Original value: {original_angle}")
print(f"New value: {new_angle}")

objrot = R.from_euler('XYZ', new_angle, degrees=True)
objrot_mat = objrot.as_matrix()
objrot_mat[np.abs(objrot_mat) < 1e-5] = 0.0
print("Output:")
print(f'{camera_id},"{repr(data["pos"])}","{repr([list(x) for x in objrot_mat])}"')

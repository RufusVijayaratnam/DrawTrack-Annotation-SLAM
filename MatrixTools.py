import numpy as np

def gen_rotation_matrix(alpha, beta, gamma):
    rz = np.matrix([[np.cos(gamma), -np.sin(gamma),  0],
                    [np.sin(gamma),  np.cos(gamma),  0],
                    [0,                          0,  1]])

    ry = np.matrix([[np.cos(beta),  0, np.sin(beta)],
                    [0,             1,             0],
                    [-np.sin(beta), 0, np.cos(beta)]])


    rx = np.matrix([[1,            0,               0],
                    [0, np.cos(alpha), -np.sin(alpha)], 
                    [0, np.sin(alpha), np.cos(alpha)]])

    rotation_matrix = rz * ry * rx
    return rotation_matrix

def signed_angle_between_2d_vec(vec1, vec2):
    dot = np.dot(vec1, vec2)
    det = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    angle = np.arctan2(det, dot)
    return angle
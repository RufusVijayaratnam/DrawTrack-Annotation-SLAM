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
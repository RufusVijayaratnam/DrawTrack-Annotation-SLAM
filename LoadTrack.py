import numpy as np
from MatrixTools import *

def load_cones(file, cone_separation_m=2.5):
    track_direction_ws, trackpoints_ws = load_track(file)
    blue_cones_ws = []
    yellow_cones_ws = []
    up = [0 ,0, 1]

    for point, direction in zip(trackpoints_ws, track_direction_ws):
        chord_vec = np.cross(direction, up)
        yellow_cone_ws = point - 0.5 * cone_separation_m * chord_vec
        yellow_cones_ws.append(yellow_cone_ws)
        blue_cone_ws = point + 0.5 * cone_separation_m * chord_vec
        blue_cones_ws.append(blue_cone_ws)

    return blue_cones_ws, yellow_cones_ws

def load_track(file):
    f = open(file, 'r')
    trackpoints_worldspace = []
    isClosed = False

    lines = f.readlines()
    for line in lines:
        line.rstrip()
        chars = line.split()
        if chars[0] == 'p':
            x = float(chars[1])
            y = float(chars[2])
            trackpoints_worldspace.append([x, y, 0])
        elif chars[0] == "closed":
            isClosed = True

    track_direction_worldspace = []
    for i in range(0, len(trackpoints_worldspace) - 1):
        direction = np.subtract(trackpoints_worldspace[i + 1], trackpoints_worldspace[i])
        direction = direction / np.linalg.norm(direction)
        track_direction_worldspace.append(direction)

    if isClosed == False:
        direction = track_direction_worldspace[-1]
        track_direction_worldspace.append(direction)
    else:
        direction = np.subtract(trackpoints_worldspace[-1], trackpoints_worldspace[0])
        direction = direction / np.linalg.norm(direction)
        track_direction_worldspace.append(direction)
    
    return track_direction_worldspace, trackpoints_worldspace

def stereo_cam_ext(filepath, substeps=2, cam_spacing=0.5):
    #Track direction are all unit vectors
    track_direction, track_points_ws = load_track(filepath)
    render_points = np.ndarray((substeps * len(track_points_ws), 3))

    render_pairs = substeps * len(track_direction)
    #Next three lines are all in blender space
    left_cam_points_ws = np.ndarray((render_pairs, 3))
    right_cam_points_ws = np.ndarray((render_pairs, 3))
    blender_cam_rotation_ws = np.ndarray((render_pairs, 3))
    
    step_count = 0
    up_unit_vec = [0, 0, 1]
    forward_unit = np.array([0, -1, 0])
    for i in range(render_pairs - 2):
        p1 = track_points_ws[i + 1]
        p0 = track_points_ws[i]
        d1 = track_direction[i + 1]
        d0 = track_direction[i]
        print("d0: \n", d0)
        total_turn_angle_rad = np.arccos(np.dot(d1, d0)) #About Z blender space
        print("turn angle is %f degrees" % np.rad2deg(total_turn_angle_rad))
        step_turn_angle_rad = total_turn_angle_rad / substeps
        vec = np.subtract(p1, p0)
        step_vec = vec / substeps
        for j in range(substeps):
            rotation_matrix = gen_rotation_matrix(0, 0, -step_turn_angle_rad * (j + 1))
            print("current turn angle is: ", np.rad2deg((step_turn_angle_rad * (j + 1))))
            direction = rotation_matrix * np.matrix(d0).transpose()
            direction = np.asarray(direction.transpose())[0]
            cone_chord = np.cross(up_unit_vec, direction)
            print("cone_chord: \n", cone_chord)
            point = p0 + step_vec * j
            right_cam_loc_ws = point - 0.5 * cam_spacing * cone_chord 
            right_cam_loc_ws[2] = 0.5
            left_cam_loc_ws = point + 0.5 * cam_spacing * cone_chord
            left_cam_loc_ws[2] = 0.5
            rz = 180 - np.rad2deg(np.arccos(np.dot(direction, forward_unit) / (np.linalg.norm(direction))))
            print("rx: ", rz)
            cam_rot = np.array([90, 0, rz])
            left_cam_points_ws[step_count] = left_cam_loc_ws
            right_cam_points_ws[step_count] = right_cam_loc_ws
            blender_cam_rotation_ws[step_count] = cam_rot
            render_points[step_count] = point
            step_count += 1
  
    return left_cam_points_ws, right_cam_points_ws, blender_cam_rotation_ws
    
        
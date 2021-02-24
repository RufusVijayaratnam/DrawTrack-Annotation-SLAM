import numpy as np

def gen_rotation_matrix(alpha, beta, gamma):
    rz = np.matrix([[np.cos(gamma), -np.sin(gamma),  0],
                    [np.sin(gamma),  np.cos(gamma),  0],
                    [0,                          0,  0]])

    ry = np.matrix([[np.cos(beta),  0, np.sin(beta)],
                    [0,             1,             0],
                    [-np.sin(beta), 0, np.cos(beta)]])


    rx = np.matrix([[1,            0,               0],
                    [0, np.cos(alpha), -np.sin(alpha)], 
                    [0, np.sin(alpha), np.cos(alpha)]])

    rotation_matrix = rz * ry * rx
    return rotation_matrix

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
    track_direction, track_points_ws = load_track(filepath)
    render_points = np.ndarray((substeps * len(track_points_ws), 3))
    step_count = 0
    for i in range(0, (len(track_points_ws) - 1)):
        p1 = track_points_ws[i + 1]
        p0 = track_points_ws[i]
        vec = np.subtract(p1, p0)
        step_vec = vec / substeps
        for j in range(substeps):
            point = p0 + step_vec * j
            render_points[step_count] = point
            step_count += 1

    up_unit_vec = [0, 0, 1]
    render_pairs = len(render_points) - 1
    left_cam_points_ws = np.ndarray((render_pairs, 3))
    right_cam_points_ws = np.ndarray((render_pairs, 3))
    blender_cam_rotation_ws = np.ndarray((render_pairs, 3))
    annotation_cam_rotation_ws = np.ndarray((render_pairs, 3))


    for i in range(render_pairs):
        p1 = render_points[i + 1]
        p0 = render_points[i]
        vec = np.subtract(p1, p0)
        direction = vec / np.linalg.norm(vec)
        cross_vec = np.cross(up_unit_vec, direction)
        right_cam_loc_ws = render_points[i] - 0.5 * cam_spacing * cross_vec
        right_cam_loc_ws[2] = 0.5
        left_cam_loc_ws = render_points[i] + 0.5 * cam_spacing * cross_vec
        left_cam_loc_ws[2] = 0.5
        y_unit = np.array([0, 1, 0])
        rz = 360 - np.arccos(np.dot(y_unit, vec) / (np.linalg.norm(vec) * np.linalg.norm(y_unit))) * 180 / np.pi
        cam_rot = np.array([90, 0, rz])
        anno_cam_rot = np.array([0, rz, 0]) #Definitely not the best way of doing this
        left_cam_points_ws[i] = left_cam_loc_ws
        right_cam_points_ws[i] = right_cam_loc_ws
        blender_cam_rotation_ws[i] = cam_rot
        annotation_cam_rotation_ws[i] = anno_cam_rot
        #set_camera_ext(right_cam, right_cam_loc_ws, cam_rot)
        #render_save(right_cam)
        #set_camera_ext(left_cam, left_cam_loc_ws, cam_rot)
        #render_save(left_cam)

    return left_cam_points_ws, right_cam_points_ws, blender_cam_rotation_ws
    
        

import numpy as np
import cv2 as cv
import sys
sys.path.append("/mnt/c/Users/Rufus Vijayaratnam/Driverless/Draw and Annotate/")
sys.path.append("/mnt/c/Users/Rufus Vijayaratnam/Driverless/Location and Mapping/")
import LoadTrack as lt
import Annotate as anno
import Detections
import Matching
import Colour
import Mapping
import DebugTools as dbgt
import matplotlib.pyplot as plt
import MatrixTools

rsrc_path = "/mnt/c/Users/Rufus Vijayaratnam/Driverless/Blender/Resources/Renders/video/"

angle = np.pi - np.deg2rad(182.121151)
coords = []
location_ws = Detections.Point(-0.049966, 0, -0.001851)
coords.append(location_ws)
initial_direction = (MatrixTools.gen_rotation_matrix(0, angle, 0) * location_ws).unit()
up_unit = Detections.Point(0, -1, 0)

def rms(arry):
    squared = arry ** 2
    mean = np.mean(squared)
    return np.sqrt(mean)

def derive_motion(train, query):
    if len(train) < 2:
        print("Uh oh spaghetti-o, we need at least two matched cones")
        #Do some frame skipping protocol
        return
    
    t1 = train[0].loc_cs
    t2 = train[1].loc_cs

    q1 = query[0].loc_cs
    q2 = query[1].loc_cs
    tz_1 = t1.z
    tx_1 = t1.x
    tx_2 = t2.x
    tz_2 = t2.z

    qx_1 = q1.x
    qz_1 = q1.z
    qx_2 = q2.x
    qz_2 = q2.z


    x_t = (-qx_1*(tx_1*tx_2 - tx_2**2 + tz_1*tz_2 - tz_2**2) + qx_2*(tx_1**2 - tx_1*tx_2 + tz_1**2 - tz_1*tz_2) + qz_1*(tx_1*tz_2 - tx_2*tz_1) - qz_2*(tx_1*tz_2 - tx_2*tz_1))/(tx_1**2 - 2*tx_1*tx_2 + tx_2**2 + tz_1**2 - 2*tz_1*tz_2 + tz_2**2)
    z_t = (-qx_1*(tx_1*tz_2 - tx_2*tz_1) + qx_2*(tx_1*tz_2 - tx_2*tz_1) - qz_1*(tx_1*tx_2 - tx_2**2 + tz_1*tz_2 - tz_2**2) + qz_2*(tx_1**2 - tx_1*tx_2 + tz_1**2 - tz_1*tz_2))/(tx_1**2 - 2*tx_1*tx_2 + tx_2**2 + tz_1**2 - 2*tz_1*tz_2 + tz_2**2)
    sin_theta = (-qx_1*(tz_1 - tz_2) + qx_2*(tz_1 - tz_2) + qz_1*(tx_1 - tx_2) - qz_2*(tx_1 - tx_2))/(tx_1**2 - 2*tx_1*tx_2 + tx_2**2 + tz_1**2 - 2*tz_1*tz_2 + tz_2**2)
    cos_theta = (qx_1*(tx_1 - tx_2) - qx_2*(tx_1 - tx_2) + qz_1*(tz_1 - tz_2) - qz_2*(tz_1 - tz_2))/(tx_1**2 - 2*tx_1*tx_2 + tx_2**2 + tz_1**2 - 2*tz_1*tz_2 + tz_2**2)

    theta = np.arcsin(sin_theta)
    return x_t, z_t, theta

def get_real_change(frame):
    path = rsrc_path + "labels/simple-left_ext.txt"
    with open(path) as f:
        idx = frame - 1
        lines = f.readlines()
        train_line = lines[idx]
        train_line = train_line.split()
        train_line = [float(val) for val in train_line]
        query_line = lines[idx + 1]
        query_line = query_line.split()
        query_line = [float(val) for val in query_line]
        xt = train_line[1]
        zt = train_line[2]
        theta_t = train_line[6]
        xq = query_line[1]
        zq = query_line[2]
        theta_q = query_line[6]
        delta_x = abs(xt - xq)
        delta_z = abs(zt - zq)
        delta_theta = abs(theta_t - theta_q)
        theta = np.deg2rad(delta_theta)
        delta_x *= np.sin(theta)
        delta_z *= np.cos(theta)
        print("Real: xt: %f, zt: %f, theta_deg: %f" % (delta_x, delta_z, delta_theta))

def get_track_map(positions):
    im_size = 720
    blank_image = np.zeros((im_size, im_size, 3), np.uint8)
    blank_image[:] = (128, 128, 128)
    cv.circle(blank_image, (int(im_size / 2), int(im_size - 20)), 5, (0, 255, 0), thickness=-1)
    for pos in positions:
        x_cs = pos.x
        z_cs = pos.z
        im_y = int(im_size - z_cs / 80 * (im_size - 20))
        im_x = int(im_size / 2 + x_cs / 40 * im_size)
        cv.circle(blank_image, (im_x, im_y), 4, (0, 0, 0), thickness=-1)
    return blank_image

for num in range(1, 140):
    train_real_path = "labels/Real-simple-Left_Cam-Render-%d.txt" % num
    query_real_path = "labels/Real-simple-Left_Cam-Render-%d.txt" % (num + 1)
    train_image_path = "images/simple-Left_Cam-Render-%d.png" % num
    query_image_path = "images/simple-Left_Cam-Render-%d.png" % (num + 1)

    train_im = np.array(cv.imread(rsrc_path + train_image_path))
    query_im = np.array(cv.imread(rsrc_path + query_image_path))
    im_width = np.shape(train_im)[1]
    im_height = np.shape(train_im)[0]


    detections = []
    with open(rsrc_path + train_real_path) as r:
        with open(rsrc_path + train_real_path.replace("Real-", "")) as f:
            for real, box in zip(r.readlines(), f.readlines()):
                real = real.split()
                box = box.split()
                box = [float(val) for val in box]
                w = int(box[3] * im_width)
                h = int(box[4] * im_height)
                cx = int(box[1] * im_width)
                cy = int(box[2] * im_height)
                colour = real[0]
                detected_cone = Detections.DetectedCone(cx, cy, w, h)
                if colour == "blue":
                    detected_cone.colour = Colour.blue
                elif colour == "yellow":
                    detected_cone.colour = Colour.yellow
                
                detected_cone.loc_cs = Detections.Point(float(real[1]), float(0), float(real[3]))
                detections.append(detected_cone)
    train = Detections.Detections(detections, max_dist=100)
    train.image = train_im


    detections = []
    with open(rsrc_path + query_real_path) as r:
        with open(rsrc_path + query_real_path.replace("Real-", "")) as f:
            for real, box in zip(r.readlines(), f.readlines()):
                real = real.split()
                box = box.split()
                box = [float(val) for val in box]
                w = int(box[3] * im_width)
                h = int(box[4] * im_height)
                cx = int(box[1] * im_width)
                cy = int(box[2] * im_height)
                colour = real[0]
                detected_cone = Detections.DetectedCone(cx, cy, w, h)
                if colour == "blue":
                    detected_cone.colour = Colour.blue
                elif colour == "yellow":
                    detected_cone.colour = Colour.yellow
                
                detected_cone.loc_cs = Detections.Point(float(real[1]), float(0), float(real[3]))
                detections.append(detected_cone)
    query = Detections.Detections(detections, max_dist=100)
    query.image = query_im

    train.init_detections()
    query.init_detections()
    train = train.filter_distance()
    query = query.filter_distance()

    print("frame: ", num)

    for i, t in enumerate(train):
        train[i].depth = t.loc_cs.z

    for i, q in enumerate(query):
        query[i].depth = q.loc_cs.z

    matcher = Matching.FrameMatcher(train, query)
    matcher.find_subsequent_matches()
    train_matched, query_matched = matcher.get_matched()
    if len(train_matched) < 2:
        print("frame: %d Oh shit no matches" % num)

    #get_real_change(num)
    x, z, theta = derive_motion(train_matched, query_matched)
    angle += theta
    rotation_matrix = MatrixTools.gen_rotation_matrix(0, angle, 0)
    pos_change = Detections.Point(x, 0, z)
    location_ws = location_ws + rotation_matrix * -pos_change
    direction_unit = rotation_matrix * initial_direction
    normal_unit = direction_unit * up_unit
    car_pos = location_ws + 0.05 * normal_unit
    coords.append(car_pos)
    
image = get_track_map(coords)
dbgt.show_image("Path", image)



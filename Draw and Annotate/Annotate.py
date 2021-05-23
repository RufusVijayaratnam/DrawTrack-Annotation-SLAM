import numpy as np
import sys
import LoadTrack as lt
from MatrixTools import *
import cv2 as cv
import os
import sys
sys.path.append(os.path.abspath("../Location and Mapping"))
from Detections import *
from Colour import *
import argparse


def axis_transform(point):
    #OpenCV cardinal vectors in blender coordinate system
    cv_z = np.matrix([0, -1, 0]).transpose()
    cv_x = np.matrix([-1, 0, 0]).transpose()
    cv_y = np.matrix([0, 0, -1]).transpose()
    M = np.linalg.inv(np.c_[cv_x, cv_y, cv_z])
    rotation_matrix = M.transpose()
    point = np.matrix(point).transpose()
    point = rotation_matrix * point
    return point

def normalise_wrt_indx(point, indx=2): #Default to normalise by Z
    normalised = np.matrix(np.array([val / point[indx] for val in point])).transpose()
    return normalised

def cone_annotation_bounds(cam_loc_ws, cone_loc_ws, intrinsic_matrix, rotation_matrix):
    cam_direction_unit = rotation_matrix.T * np.matrix(cam_initial_direction).transpose()
    cone_to_cam_vec = np.subtract(cone_loc_ws, cam_loc_ws)
    cone_to_cam_vec_unit = cone_to_cam_vec / np.linalg.norm(cone_to_cam_vec)    

    normal_unit = np.cross(up_vec, cam_direction_unit.transpose())
    top_left_point = cone_loc_ws.transpose() + cone_height_m * up_vec + cone_widht_m / 2 * normal_unit + cone_widht_m / 2 * cone_to_cam_vec_unit.transpose()
    bottom_right_point = cone_loc_ws.transpose() - cone_widht_m / 2 * normal_unit - cone_widht_m / 2 * cone_to_cam_vec_unit.transpose()

    p1_cs = np.subtract(top_left_point, cam_loc_ws.T).transpose()
    p2_cs = np.subtract(bottom_right_point, cam_loc_ws.T).transpose()
    cam_direction_unit = np.squeeze(np.asarray(cam_direction_unit))
    cone_to_cam_vec_unit = np.squeeze(np.asarray(cone_to_cam_vec_unit))
    angle_between_direction_and_cone_cam = np.arccos(np.dot(cone_to_cam_vec_unit, cam_direction_unit))

    if angle_between_direction_and_cone_cam >= np.pi / 2:
        return -1, -1, -1, -1
    
    img_point1 = intrinsic_matrix * rotation_matrix * p1_cs
    if img_point1[2] == 0:
        return -1, -1, -1, -1
    img_point1 = normalise_wrt_indx(img_point1)
    img_point2 = intrinsic_matrix * rotation_matrix * p2_cs
    if img_point2[2] == 0:
        return -1, -1, -1, -1
    img_point2 = normalise_wrt_indx(img_point2)
    u1 = img_point1[0]
    v1 = img_point1[1]
    u2 = img_point2[0]
    v2 = img_point2[1]
    x = int((u1 + u2) / 2)
    y = int((v1 + v2) / 2)
    w = abs(u1 - u2)
    h = abs(v1 - v2)
    return x, y, w, h

def annotate_image(render, cam_loc_ws, cam_rotation, blue_cone_ws, yellow_cone_ws, fx, fy, rsrc, track_name, res_x=1920, res_y=1080):
    sep = os.sep
    #img_points = ImagePoints(len(blue_cone_ws), res_x, res_y)
    num_cones = len(blue_cone_ws)
    cam_rotation_rad = np.deg2rad(cam_rotation)
    alpha = float(cam_rotation_rad[0])
    #beta = rotation about z
    beta = float(cam_rotation_rad[1])
    #gamma = rotation about y
    gamma = float(cam_rotation_rad[2] - np.pi)
    rotation_matrix = gen_rotation_matrix(0, gamma, 0)
    
    intrinsic_matrix = np.matrix([[fx, 0, res_x / 2],
                                [0, fy, res_y / 2],
                                [0, 0, 1]])
    cam_direction_unit = rotation_matrix.T * np.matrix(cam_initial_direction).transpose()
    cones = []
    c_loc_ws = Vec3(cam_loc_ws[0], cam_loc_ws[1], cam_loc_ws[2])
    for i, (blue_cone, yellow_cone) in enumerate(zip(blue_cone_ws, yellow_cone_ws)):
        x, y, w, h = cone_annotation_bounds(cam_loc_ws, blue_cone, intrinsic_matrix, rotation_matrix)
        cone = DetectedCone(x, y, w, h)
        cone.colour = blue
        x_ws = blue_cone[0]
        y_ws = blue_cone[1]
        z_ws = blue_cone[2]
        cone.loc_cs = rotation_matrix * (Vec3(x_ws, y_ws, z_ws) - c_loc_ws)
        cones.append(cone)
        x, y, w, h = cone_annotation_bounds(cam_loc_ws, yellow_cone, intrinsic_matrix, rotation_matrix)
        cone = DetectedCone(x, y, w, h)
        x_ws = yellow_cone[0]
        y_ws = yellow_cone[1]
        z_ws = yellow_cone[2]
        cone.colour = yellow
        cone.loc_cs = rotation_matrix * (Vec3(x_ws, y_ws, z_ws) - c_loc_ws)
        cones.append(cone)
    
    print("Annotating: %s" % (render))
    image_folder = "Renders" + sep + train_or_val + sep + "images" + sep
    labels_folder = "Renders" + sep + train_or_val + sep + "labels" + sep
    file_path = rsrc + sep + "Renders" + sep + train_or_val + sep + "Annotated-images" + sep + "%s" % (render.replace(".png", ".jpg"))
    active_image = cv.imread(os.path.join(rsrc, image_folder, render))
    annotations = Detections(cones)
    annotations.image = active_image
    annotations.init_detections()
    annotations = annotations.remove_off_screen()
    im_width = np.shape(active_image)[1]
    im_height = np.shape(active_image)[0]   
    anno_name = render.replace(".png", ".txt")
    f_yolo = open(rsrc + sep + labels_folder + "%s" % (anno_name), "w+")
    f_real = open(rsrc + sep + labels_folder + "%s" % ("Real-" + anno_name), "w+")
    for anno in annotations:
        cx = anno.cx / im_width
        cy = anno.cy / im_height
        w = anno.w / im_width
        h = anno.h / im_height
        yolo_string = "0 %f %f %f %f\n" % (cx, cy, w, h)
        f_yolo.write(yolo_string)
        loc = anno.loc_cs
        x_cs, y_cs, z_cs = loc.x, loc.y, loc.z
        colour = anno.colour.name
        f_real.write("%s %f %f %f\n" % (colour, x_cs, y_cs, z_cs))
    cv.imwrite(file_path, np.array(annotations.get_annotated_image()))
    f_yolo.close()
    f_real.close()
        
    return

def annotate_track(track_name):
    info_f = open(os.path.join(rsrc, "Renders", "Track Info", track_name + "_info.txt"))
    info_lines = info_f.readlines()
    substeps = int(info_lines[0].split()[1])
    global train_or_val
    train_or_val = info_lines[1].split()[1]
    sep = os.sep
    file_path = rsrc + sep + "Tracks" + sep + track_name + ".txt"
    blue_cone_ws,  yellow_cone_ws = lt.load_cones(file_path)
    if train_or_val == "video":
        left_label = rsrc + sep + "Renders" + sep + "video" + sep + "labels" + sep + "%s-left_ext.txt" % track_name
        right_label = rsrc + sep +"Renders" + sep + "video" + sep + "labels" + sep + "%s-right_ext.txt" % track_name
        left_cam_points_ws = []
        right_cam_points_ws = []
        cam_rotation_ws = []
        with open(left_label) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                line = [float(val) for val in line]
                left_cam_points_ws.append(np.array([line[1], line[2], line[3]]))
                cam_rotation_ws.append(np.array([(line[4]), (line[5]), (line[6])]))
                
        with open(right_label) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                line = [float(val) for val in line]
                right_cam_points_ws.append(np.array([line[1], line[2], line[3]]))
    else:
        left_cam_points_ws, right_cam_points_ws, cam_rotation_ws = lt.stereo_cam_ext(file_path, substeps=substeps)
        
    blue_cone_ws = [axis_transform(point) for point in blue_cone_ws]
    yellow_cone_ws = [axis_transform(point) for point in yellow_cone_ws]
    left_cam_points_ws = [axis_transform(point) for point in left_cam_points_ws]
    right_cam_points_ws = [axis_transform(point) for point in right_cam_points_ws]
    images_folder = os.path.join(rsrc, "Renders", train_or_val, "images")
    renders = os.listdir(images_folder)
    renders = [render for render in renders if "%s-" % track_name in render]
    imgWidth = cv.imread(os.path.join(images_folder, renders[0])).shape[1]
    focalLength_mm = float(info_lines[2].split()[1])
    sensorWidth_mm = float(info_lines[3].split()[1])
    sensorHeight_mm = float(info_lines[4].split()[1])
    focalLength_pixels = (focalLength_mm / sensorWidth_mm) * imgWidth
    fx = focalLength_pixels
    fy = fx

    for i in range(len(renders)):
        render = renders[i]
        left = "%s-Left_Cam" % track_name
        right = "%s-Right_Cam" % track_name
        cam_info = render.split("-Render-") #cam_info[0] = left / right, cam_info[1] = cam index
        cam = cam_info[0]
        cam_indx = int(cam_info[1].replace(".png", ""))
        print("Render = " + render + "cam_indx: " + str(cam_indx))
        if cam == left:
            cam_loc_ws = left_cam_points_ws[cam_indx]
            cam_rotation = cam_rotation_ws[cam_indx]
            annotate_image(render, cam_loc_ws, cam_rotation, blue_cone_ws, yellow_cone_ws, fx, fy, rsrc, track_name)
        elif cam == right:
            cam_loc_ws = right_cam_points_ws[cam_indx]
            cam_rotation = cam_rotation_ws[cam_indx]
            annotate_image(render, cam_loc_ws, cam_rotation, blue_cone_ws, yellow_cone_ws, fx, fy, rsrc, track_name)
        else:
            print("Couldn't determine which camera, skipping this render.")
        

if __name__ == "__main__":
    cone_height_m = 0.262
    cone_widht_m = 0.324
    up_vec = np.array([0, -1, 0]) #CV
    cam_initial_direction = np.array([0, 0, 1]) #CV
    train_or_val = "test" #Don't delete this, just send as parameter
    rsrc = os.path.abspath("../Blender/Resources")

    parser = argparse.ArgumentParser()
    parser.add_argument("trackname", help="Name of track to annotate.")
    parser.add_argument("--images", metavar="'False'", help="Set to 'False' if you don't want the annotation to create annotated images (not yet implemented)")
    args = parser.parse_args()
    track_name = args.trackname
    annotate_track(track_name)

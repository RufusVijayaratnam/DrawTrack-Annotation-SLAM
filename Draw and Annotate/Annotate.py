import numpy as np
import sys
import LoadTrack as lt
from MatrixTools import *
import cv2 as cv
import os

cone_height_m = 0.262
cone_widht_m = 0.324
up_vec = np.array([0, -1, 0]) #CV
cam_initial_direction = np.array([0, 0, 1]) #CV
train_or_val = "train"


class ImagePoints:
    
    def __init__(self, num_cone_points, res_x, res_y):
        self.blue_points = np.ndarray(num_cone_points, dtype=(tuple, 2))
        self.yellow_points = np.ndarray(num_cone_points, dtype=(tuple, 2))
        self.res_x = res_x
        self.res_y = res_y
    
    def on_screen(self, point):
        if point[0] > 0 and point[1] > 0 and point[0] <= self.res_x and point[1] <= self.res_y:
            return True
        else:
            return False
    
    def swap_points(self, arry):
        arry = np.asarray([((int(p[0][1]), int(p[0][0])), (int(p[1][1]), int(p[1][0]))) for p in arry], dtype=(tuple, 2))
        return arry
        
    def clean(self):
        blue_points_cleaned = np.asarray([point for point in self.blue_points if self.on_screen(point[0]) and self.on_screen(point[1])])
        yellow_points_cleaned = np.asarray([point for point in self.yellow_points if self.on_screen(point[0]) and self.on_screen(point[1])])
        self.blue_points = np.array(blue_points_cleaned)
        self.yellow_points = np.array(yellow_points_cleaned)
        #Should get rid of this and make a better way of casting to int
        self.blue_points = self.swap_points(self.blue_points)
        self.yellow_points = self.swap_points(self.yellow_points)
        self.blue_points = self.swap_points(self.blue_points)
        self.yellow_points = self.swap_points(self.yellow_points)

    

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
        return ((-1, -1), (-1, -1))
    
    img_point1 = intrinsic_matrix * rotation_matrix * p1_cs
    img_point1 = normalise_wrt_indx(img_point1)
    img_point2 = intrinsic_matrix * rotation_matrix * p2_cs
    img_point2 = normalise_wrt_indx(img_point2)

    u1 = img_point1[0]
    v1 = img_point1[1]
    u2 = img_point2[0]
    v2 = img_point2[1]

    return ((u1, v1), (u2, v2))

def annotate_image(render, cam_loc_ws, cam_rotation, blue_cone_ws, yellow_cone_ws, fx, fy, resource_folder, track_name, file, res_x=1920, res_y=1080):
    sep = os.sep
    img_points = ImagePoints(len(blue_cone_ws), res_x, res_y)
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
    for i, (blue_cone, yellow_cone) in enumerate(zip(blue_cone_ws, yellow_cone_ws)):
        blue_point = cone_annotation_bounds(cam_loc_ws, blue_cone, intrinsic_matrix, rotation_matrix)
        yellow_point = cone_annotation_bounds(cam_loc_ws, yellow_cone, intrinsic_matrix, rotation_matrix)
        img_points.blue_points[i] = blue_point
        img_points.yellow_points[i] = yellow_point

    img_points.clean()
    print("Annotating: %s" % (render))
    image_folder = "Renders" + sep + train_or_val + sep + "images" + sep
    labels_folder = "Renders" + sep + train_or_val + sep + "labels" + sep
    file_path = resource_folder + "Renders" + sep + train_or_val + sep + "Annotated-images" + sep + "%s" % (render.replace(".png", ".jpg"))
    active_image = cv.imread(resource_folder + image_folder + render)
    im_width = np.shape(active_image)[1]
    im_height = np.shape(active_image)[0]
    anno_name = render.replace(".png", ".txt")

    f_yolo = open(resource_folder + labels_folder + "%s" % (anno_name), "w+")

    total_cones = len(img_points.blue_points) + len(img_points.yellow_points)
    if total_cones == 0:
        file.write("0")
        return

    file.write("%i  " % total_cones) #Double space
    count = 0
    for rects in img_points.blue_points:
        p1 = rects[0]
        p2 = rects[1]
        #For Haar Cascade
        file.write("%i %i %i %i   " %(p1[0], p1[1], p2[0], p2[1])) 
        #For Yolo:
        height = float(abs(p2[1] - p1[1]) / im_height)
        width = float(abs(p2[0] - p1[0])  / im_width)
        x = float((p1[0] + p2[0]) / 2) / im_width
        y = float((p1[1] + p2[1]) / 2) / im_height
        yolo_string = "0 %f %f %f %f\n" % (x, y, width, height)
        f_yolo.write(yolo_string)
        cv.rectangle(active_image, p1, p2, (255, 255, 0))
        
    
    for rects in img_points.yellow_points:
        p1 = rects[0]
        p2 = rects[1]
        #For Haar Cascade
        file.write("%i %i %i %i   " %(p1[0], p1[1], p2[0], p2[1]))
        #For Yolo:
        height = float(abs(p2[1] - p1[1]) / im_height)
        width = float(abs(p2[0] - p1[0])  / im_width)
        x = float((p1[0] + p2[0]) / 2) / im_width
        y = float((p1[1] + p2[1]) / 2) / im_height
        yolo_string = "0 %f %f %f %f\n" % (x, y, width, height)
        f_yolo.write(yolo_string)
        cv.rectangle(active_image, p1, p2, (255, 0, 0))
        
    cv.imwrite(file_path, active_image)
    f_yolo.close()
        
    return

def annotate_track(resource_folder, track_name, substeps, tov="train"):
    global train_or_val
    train_or_val = tov
    sep = os.sep
    file_path = resource_folder + "Tracks" + sep + track_name + ".txt"
    blue_cone_ws,  yellow_cone_ws = lt.load_cones(file_path)
    left_cam_points_ws, right_cam_points_ws, cam_rotation_ws = lt.stereo_cam_ext(file_path, substeps=substeps)
    blue_cone_ws = [axis_transform(point) for point in blue_cone_ws]
    yellow_cone_ws = [axis_transform(point) for point in yellow_cone_ws]
    left_cam_points_ws = [axis_transform(point) for point in left_cam_points_ws]
    right_cam_points_ws = [axis_transform(point) for point in right_cam_points_ws]

    images_folder = "Renders" + sep + train_or_val + sep + "images" + sep
    labels_folder = "Renders" + sep + train_or_val + sep + "labels" + sep
    naming_pattern = "%s-R%i.png"
    renders = os.listdir(resource_folder + images_folder)
    renders = renders[1:]
    renders = [render for render in renders if "%s-" % track_name in render]

    imgWidth = cv.imread(resource_folder + images_folder + renders[0]).shape[1]
    focalLength_mm = 50 #Should use more realistic values
    sensorWidth_mm = 36
    sensorHeight_mm = 24
    focalLength_pixels = (focalLength_mm / sensorWidth_mm) * imgWidth
    fx = focalLength_pixels
    fy = fx


    f = open(resource_folder + labels_folder + "%s-info.dat" % track_name, "w+")


    for i in range(len(renders)):
        render = renders[i]
        f.write(resource_folder + images_folder + resource_folder + "  ")
        left = "%s-Left_Cam" % track_name
        right = "%s-Right_Cam" % track_name
        cam_info = render.split("-Render-") #cam_info[0] = left / right, cam_info[1] = cam index
        cam = cam_info[0]
        cam_indx = int(cam_info[1].replace(".png", ""))
        if cam == left:
            cam_loc_ws = left_cam_points_ws[cam_indx]
            cam_rotation = cam_rotation_ws[cam_indx]
            annotate_image(render, cam_loc_ws, cam_rotation, blue_cone_ws, yellow_cone_ws, fx, fy, resource_folder, track_name, f)
            f.write("\n")
        elif cam == right:
            cam_loc_ws = right_cam_points_ws[cam_indx]
            cam_rotation = cam_rotation_ws[cam_indx]
            annotate_image(render, cam_loc_ws, cam_rotation, blue_cone_ws, yellow_cone_ws, fx, fy, resource_folder, track_name, f)
            f.write("\n")
        else:
            print("Couldn't determine which camera, skipping this render.")
        
    f.close()







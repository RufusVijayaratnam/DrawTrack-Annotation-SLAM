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

rsrc_path = "/mnt/c/Users/Rufus Vijayaratnam/Driverless/Blender/Resources/Renders/test/"
left_image_path = "images/track1-Left_Cam-Render-18.png"
right_image_path = "images/track1-Right_Cam-Render-18.png"
left_label_path = "labels/track1-Left_Cam-Render-18.txt"
left_real_path = "labels/Real-track1-Left_Cam-Render-18.txt"
right_label_path = "labels/track1-Right_Cam-Render-18.txt"



train_im = np.array(cv.imread(rsrc_path + left_image_path))
query_im = np.array(cv.imread(rsrc_path + right_image_path))

im_width = np.shape(train_im)[1]
im_height = np.shape(train_im)[0]

detections = []
with open(rsrc_path + left_label_path) as f:
    for box in f.readlines():
        box = box.split()
        box = [float(val) for val in box]
        cx = int(box[1] * im_width)
        cy = int(box[2] * im_height)
        w = int(box[3] * im_width)
        h = int(box[4] * im_height)
        detected_cone = Detections.DetectedCone(cx, cy, w, h)
        detections.append(detected_cone)
train = Detections.Detections(detections, train_im, max_dist=100)

detections = []
with open(rsrc_path + right_label_path) as f:
    for box in f.readlines():
        box = box.split()
        box = [float(val) for val in box]
        w = int(box[3] * im_width)
        h = int(box[4] * im_height)
        cx = int(box[1] * im_width)
        cy = int(box[2] * im_height)
        detected_cone = Detections.DetectedCone(cx, cy, w, h)
        detections.append(detected_cone)
query = Detections.Detections(detections, query_im, max_dist=100)

detections = []
with open(rsrc_path + left_real_path) as r:
    for real in r.readlines():
        real = real.split()
        colour = real[0]
        detected_cone = Detections.DetectedCone(0, 0, 0, 0)
        if colour == "blue":
            detected_cone.colour = Colour.blue
        elif colour == "yellow":
            detected_cone.colour = Colour.yellow
        
        detected_cone.loc_cs = Detections.Point(float(real[1]), float(real[2]), float(real[3]))
        detections.append(detected_cone)
real_cones = Detections.Detections(detections, max_dist=100)

real_cones.image = train.image
for i, t in enumerate(train):
    real_cones[i].cx = t.cx
    real_cones[i].cy = t.cy
    real_cones[i].w = t.w
    real_cones[i].h = t.h

train.init_detections()
query.init_detections()
train.filter_distance()
query.filter_distance()
train.colour_estimation()
query.colour_estimation()

matcher = Matching.StereoMatcher(train, query)
matcher.find_stereo_matches()
%timeit matcher.calculate_depth()
train_matched, _ = matcher.get_matched()
train_matched.locate_cones()
matched_local = train_matched.get_local_map()
real_local = real_cones.get_local_map()
dbgt.hstack_images(np.array(real_local), np.array(matched_local))



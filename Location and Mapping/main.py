import numpy as np
import cv2 as cv
import torch
from Colour import *
from Detections import *
from StereoMatching import StereoMatcher
from Mapping import Mapper

yolov5_path = "/mnt/c/Users/Rufus Vijayaratnam/yolov5/runs/train/exp8/weights/best.pt"
weights_path = yolov5_path + ""
model = torch.hub.load("ultralytics/yolov5", "custom", path_or_model=weights_path)

im_left = "/mnt/c/Users/Rufus Vijayaratnam/Driverless/Blender/Resources/Renders/test/images/track7-Left_Cam-Render-10.png"
im_right = "/mnt/c/Users/Rufus Vijayaratnam/Driverless/Blender/Resources/Renders/test/images/track7-Right_Cam-Render-10.png"

image_left = np.array(cv.imread(im_left))
image_right = np.array(cv.imread(im_right))

imgs = [image_left, image_right]
imgs = [img[:, :, ::-1] for img in imgs]
results = model(imgs, size=640)
print("got here")

detections = []
for cone in results.xywh[0]:
    cx = int(cone[0])
    cy = int(cone[1])
    w = int(cone[2])
    h = int(cone[3])
    detected_cone = DetectedCone(cx, cy, w, h)
    detections.append(detected_cone)
lds = Detections(detections, imgs[0][:, :, ::-1], max_dist=20)

detections = []
for cone in results.xywh[1]:
    cx = int(cone[0])
    cy = int(cone[1])
    w = int(cone[2])
    h = int(cone[3])
    detected_cone = DetectedCone(cx, cy, w, h)
    detections.append(detected_cone)
rds = Detections(detections, imgs[1][:, :, ::-1], max_dist=20)

if len(lds) == 0 or len(rds) == 0:
    raise Exception("No Detections")
    

lds.init_detections()
rds.init_detections()
lds = lds.filter_distance()
rds = rds.filter_distance()
lds.colour_estimation()
rds.colour_estimation()

stereo_matcher = StereoMatcher(lds, rds)
stereo_matcher.find_stereo_matches()
stereo_matcher.calculate_depth()
#Should find out if i can do a pass by reference kind of thing in Python
lds_matched, rds_matched = stereo_matcher.get_located() #type(train_matched) = Detections
mapper = Mapper()
mapper.new_cones = lds_matched
mapper.locate_cones()
lds_located = mapper.get_localised_cones()
frame_matcher = FrameMatcher(
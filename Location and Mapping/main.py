import numpy as np
import cv2 as cv
import ColourEstimation as ce
import StereoMatching as sm
import torch
from Colour import *

yolov5_path = "/mnt/c/Users/Rufus Vijayaratnam/yolov5/runs/train/exp8/weights/best.pt"
weights_path = yolov5_path + ""
model = torch.hub.load("ultralytics/yolov5", "custom", path_or_model=weights_path)

im_left = "/mnt/c/Users/Rufus Vijayaratnam/Driverless/Blender/Resources/Renders/test/images/track7-Left_Cam-Render-16.png"
im_right = "/mnt/c/Users/Rufus Vijayaratnam/Driverless/Blender/Resources/Renders/test/images/track7-Right_Cam-Render-16.png"

image_left = np.array(cv.imread(im_left))
image_right = np.array(cv.imread(im_right))

imgs = [image_left, image_right]
imgs = [img[:, :, ::-1] for img in imgs]
results = model(imgs, size=640)

#ce.estimate_cone_colours(image_right, results.xyxy[1])


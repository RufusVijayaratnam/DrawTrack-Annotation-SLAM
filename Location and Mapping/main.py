import numpy as np
import cv2 as cv
import ColourEstimation as ce
import StereoMatching as sm
import torch

yolov5_path = "/mnt/c/Users/Rufus Vijayaratnam/yolov5/runs/train/exp8/weights/best.pt"
weights_path = yolov5_path + ""
model = torch.hub.load("ultralytics/yolov5", "custom", path_or_model=weights_path)

im_path = "/mnt/c/Users/Rufus Vijayaratnam/Driverless/Blender/Resources/Renders/test/images/track7-Right_Cam-Render-16.png"

image = cv.imread(im_path)
image = np.array(image)

imgs = [image[:, :, ::-1]]
results = model(imgs, size=640)

ce.estimate_cone_colours(image, results.xyxy[0])
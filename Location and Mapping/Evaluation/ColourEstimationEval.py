
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


rsrc = "/mnt/c/Users/Rufus Vijayaratnam/Driverless/Blender/Resources/"
label_real_path = "Renders/test/labels/Real-track3-Left_Cam-Render-107.txt"
train_im = cv.imread(rsrc + "Renders/test/images/track3-Left_Cam-Render-107.png")
im_width = np.shape(train_im)[1]
im_height = np.shape(train_im)[0]


detections = []
with open(rsrc + label_real_path) as r:
    with open(rsrc + label_real_path.replace("Real-", "")) as f:
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
            """ if colour == "blue":
                detected_cone.colour = Colour.blue
            elif colour == "yellow":
                detected_cone.colour = Colour.yellow """
            
            detected_cone.loc_cs = Detections.Point(float(real[1]), float(0), float(real[3]))
            detections.append(detected_cone)
train = Detections.Detections(detections, max_dist=100)
train.image = train_im
train.init_detections()
train.colour_estimation()
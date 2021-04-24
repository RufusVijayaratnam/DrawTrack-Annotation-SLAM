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
import ColourEstimation as ce

image = cv.imread("blue_cone.png")
_ = ce.estimate_colour(image)
import numpy as np
from Colour import *
from Constants import *



class DetectedCone():
    #This holds attributes to describe a detected cone
    def __init__(self, cx, cy, w, h, colour, imgWidth):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.area = w * h
        self.hypotenuse = np.sqrt((0.5 * w) ** 2 + (0.5 * h) ** 2)
        self.colour = colour
        self.imgWidth = imgWidth
        #Unique identifier used for easier index matching
        self.uid = hash("%s%s%s" % (colour.name, str(cx), str(cy)))

    def find_center_distance(self, cone):
        #Returns pixel distance between self and center of another cone
        dcx = abs(self.cx - cone.cx)
        dcy = abs(self.cy - cone.cy)
        dist = np.sqrt(dcx ** 2 + dcy ** 2)
        return dist

    def monocular_distance_estimate(self):
        #Should return an upper and lower bound for disparity (in pixels)
        #This is a very rough estimate and should be taken with a bucket of salt
        #Unlikely to work well for obscured cones
        margin = 1.1 #Percent for bounds, this can be changed to improve
        F = 5 * 134.470 / 0.262
        depth = F * 0.262 / self.h
        depth_max_mm = depth * margin * 1000 
        depth_min_mm = depth / margin * 1000 
        #return distance (z cv space), lower_disparity_pix, upper_disparity_pix
        #Disparity decreases with depth, so depth_max_mm is used to calculate disp_min
        focalLength_pixels = (focalLength_mm / sensorWidth_mm) * self.imgWidth
        disp_min = baseline_mm * focalLength_pixels / depth_max_mm
        disp_max = baseline_mm * focalLength_pixels / depth_min_mm
        return depth, np.floor(disp_min), np.ceil(disp_max)





rc1 = DetectedCone(500, 500, 50, 100, yellow, 1080)
rc2 = DetectedCone(200, 200, 45, 110, yellow, 1080) 
rc3 = DetectedCone(340, 800, 45, 212, yellow, 1080)

rds = np.array([rc1, rc2, rc3], dtype=DetectedCone)

lc1 = DetectedCone(500, 500, 50, 100, yellow, 1080)
lc2 = DetectedCone(200 + 120, 200, 45, 110, yellow, 1080) 
lc3 = DetectedCone(310, 800, 45, 212, yellow, 1080)

lds = np.array([lc1, lc2, lc3], dtype=DetectedCone)
import numpy as np
from Detections import DetectedCone, Detections
from StereoMatching import Matcher
from Colour import Colour
import Constants as consts
import cv2 as cv

class Point():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z =z

class Mapper():
    def __init__(self):
        self.prev_cones = None
        self.new_cones = None
        self.im_size = 300
        self.blank_image = np.zeros((self.im_size, self.im_size, 3), np.uint8)
        self.blank_image[:] = (255, 255, 255)
        
        cv.circle(self.blank_image, (int(self.im_size / 2), int(self.im_size - 20)), 5, (0, 255, 0))

    def visualise_local_map(self):
        im_width = self.new_cones[0].im_width
        for i, cone in enumerate(self.new_cones):
            if cone.cx < im_width / 2:
                multiplier = -1
            else:
                multiplier = 1
            sensor_loc_mm = abs(im_width / 2 - cone.cx) * (consts.sensorWidth_mm / im_width)
            angle = np.arctan(sensor_loc_mm / consts.focalLength_mm) * multiplier
            x_cs = cone.depth * np.tan(angle)
            #CONVENTION: Before the car moves, the car space coordinate system is aligned with the OpenCv coordinate system, the origins are at the same point initially.
            point = Point(x_cs, 0, cone.depth)
            self.new_cones[i].loc_cs = point
            im_y = int(self.im_size - cone.depth / 20 * (self.im_size - 20))
            im_x = int(self.im_size / 2 + x_cs / 10 * self.im_size)
            cv.circle(self.blank_image, (im_x, im_y), 4, cone.colour.colour, thickness=-1)

        cv.imshow("visualisation", self.blank_image)
        cv.waitKey(0)
        cv.destroyWindow("visualisation")
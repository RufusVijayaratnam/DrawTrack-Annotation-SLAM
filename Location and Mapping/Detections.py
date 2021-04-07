import numpy as np
from Colour import *
from Constants import *
import ColourEstimation as ce
import Constants as consts
import cv2 as cv



class DetectedCone():
    #This holds attributes to describe a detected cone
    #This class does not have image context
    def __init__(self, cx, cy, w, h):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.area = w * h
        self.hypotenuse = np.sqrt((0.5 * w) ** 2 + (0.5 * h) ** 2)
        self.colour = ambiguous
        self.im_width = 0
        #Unique identifier used for easier index matching
        self.uid = hash("%s%s%s" % (self.colour.name, str(cx), str(cy)))
        self.depth = None
        self.angle = None
        self.loc_cs = None
        self.unmapped = True
        self.loc_ws = None

    def find_center_distance(self, cone):
        #Returns pixel distance between self and center of another cone
        dcx = abs(self.cx - cone.cx)
        dcy = abs(self.cy - cone.cy)
        dist = np.sqrt(dcx ** 2 + dcy ** 2)
        return dist

    def in_range(self, max_dist):
        dist, _, _ = self.monocular_distance_estimate()
        if dist <= max_dist:
            return True
        else:
            return False

    def monocular_distance_estimate(self):
        #Should return an upper and lower bound for disparity (in pixels)
        #This is a very rough estimate and should be taken with a bucket of salt
        #Unlikely to work well for obscured cones
        margin = 1.1 #Percent for bounds, this can be changed to improve
        focalLength_pixels = (focalLength_mm / sensorWidth_mm) * self.im_width
        angle = self.angle
        #F = 5 * consts.ref_height_pix / 0.262 #0.262 = cone height m
        depth = focalLength_pixels * 0.262 / self.h
        distance = depth / np.cos(angle)
        depth_max_mm = distance * margin * 1000 
        depth_min_mm = distance / margin * 1000 
        #return distance (z cv space), lower_disparity_pix, upper_disparity_pix
        #Disparity decreases with distance, so depth_max_mm is used to calculate disp_min
        disp_min = baseline_mm * focalLength_pixels / depth_max_mm
        disp_max = baseline_mm * focalLength_pixels / depth_min_mm
        return distance, np.floor(disp_min), np.ceil(disp_max)

    def get_sub_image(self, image):
       x1 = int(self.cx - self.w / 2)
       y1 = int(self.cy - self.h / 2)
       x2 = int(self.cx + self.w / 2)
       y2 = int(self.cy + self.h / 2)
       sub_image = image[y1:y2, x1:x2]
       return sub_image


class Detections(np.ndarray):
    def __new__(cls, detections_array, image, max_dist=2):
        obj = np.asarray(detections_array, dtype=DetectedCone).view(cls)
        obj.max_dist = max_dist
        obj.image = image
        return obj

    def __set_im_width(self):
        im_width = np.shape(self.image)[1]
        for i in range(len(self)):
            self[i].im_width = im_width

    def __set_angle(self):
        for i, cone in enumerate(self):
            sensor_loc_mm = abs(im_width / 2 - cone.cx) * (consts.sensorWidth_mm / im_width)
            angle = np.arctan(sensor_loc_mm / consts.focalLength_mm)
            self[i].angle = angle


    def __array_finalize__(self, obj):
        if obj is None: return
        self.max_dist = getattr(obj, 'max_dist', None)
        self.image = getattr(obj, 'image', None)

    def init_detections(self):
        self.__set_im_width()
        self.__set_angle()

    def filter_distance(self):
        in_range_idx = np.array([v.in_range(self.max_dist) for v in self])
        out_of_range_idx = np.where(in_range_idx == False)[0]
        return np.delete(self, out_of_range_idx)


    def colour_estimation(self):
        for i, cone in enumerate(self):
            sub_image = cone.get_sub_image(self.image)
            self[i].colour = ce.estimate_colour(sub_image)

    def get_annotated_image(self):
        image = np.array(self.image)
        for cone in self:
            x1 = int(cone.cx - cone.w / 2)
            y1 = int(cone.cy - cone.h / 2)
            x2 = int(cone.cx + cone.w / 2)
            y2 = int(cone.cy + cone.h / 2)
            p1 = tuple([x1, y1])
            p2 = tuple([x2, y2])
            cv.rectangle(image, p1, p2, cone.colour.colour)
            #cv.putText(image, str(cone.depth), (cone.cx, cone.cy), cv.FONT_HERSHEY_SIMPLEX, 1,color=(0,0,0),)
        return image

    def show_annotated_image(self, name="hi"):
        image = self.get_annotated_image()
        cv.imshow(name, image)
        cv.waitKey(0)
        cv.destroyWindow(name)


    def show_image(self):
        image = self.image
        cv.imshow("hi", image)
        cv.waitKey(0)
        cv.destroyAllWindows()


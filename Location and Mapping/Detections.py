import numpy as np
from Colour import *
from Constants import *
import ColourEstimation as ce
import Constants as consts
import cv2 as cv


class Point():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def vec(self):
        return np.array([self.x, self.y, self.z])

    def __add__(self, obj):
        return Point(self.x + obj.x, self.y + obj.y, self.z + obj.z)

    def __sub__(self, obj):
        return Point(self.x - obj.x, self.y - obj.y, self.z - obj.z)

    def __str__(self):
        prnt = "[%f, %f, %f]" % (self.x, self.y, self.z)
        return prnt

    def __rmul__(self, obj):
        #for np array
        vec = np.matrix(self.vec()).transpose()
        vec = obj * vec
        return Point(vec[0], vec[1], vec[2])

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
        self.im_height = 0
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

    def on_screen(self):
        cx = self.cx
        cy = self.cy
        im_height = self.im_height
        im_width = self.im_width
        if (cx <= im_width and cx >= 0 and cy <= im_height and cy >= 0):
            return True
        else:
            return False



class Detections(np.ndarray):
    def __new__(cls, detections_array, image=None, max_dist=2):
        obj = np.asarray(detections_array, dtype=DetectedCone).view(cls)
        obj.max_dist = max_dist
        obj.image = image
        return obj

    def __set_im_width(self):
        im_width = np.shape(self.image)[1]
        im_height = np.shape(self.image)[0]
        for i in range(len(self)):
            self[i].im_width = im_width
            self[i].im_height = im_height

    def __set_angle(self):
        for i, cone in enumerate(self):
            im_width = cone.im_width
            if cone.cx < im_width / 2:
                multiplier = -1
            else:
                multiplier = 1
            sensor_loc_mm = abs(im_width / 2 - cone.cx) * (consts.sensorWidth_mm / im_width)
            angle = np.arctan(sensor_loc_mm / consts.focalLength_mm) * multiplier
            self[i].angle = angle


    def remove_off_screen(self):
        on_screen_idx = np.array([v.on_screen() for v in self])
        off_screen_idx = np.where(on_screen_idx == False)[0]
        return np.delete(self, off_screen_idx)
                

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
            #cv.putText(image, str(cone.loc_cs), (cone.cx, cone.cy), cv.FONT_HERSHEY_SIMPLEX, 1,color=(0,0,0),)
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

    def get_local_map(self):
        im_size = 720
        blank_image = np.zeros((im_size, im_size, 3), np.uint8)
        blank_image[:] = (255, 255, 255)
        cv.circle(blank_image, (int(im_size / 2), int(im_size - 20)), 5, (0, 255, 0), thickness=-1)
        for cone in self:
            x_cs = cone.loc_cs.x
            z_cs = cone.loc_cs.z
            im_y = int(im_size - z_cs / 80 * (im_size - 20))
            im_x = int(im_size / 2 + x_cs / 40 * im_size)
            cv.circle(blank_image, (im_x, im_y), 4, cone.colour.colour, thickness=-1)
        return blank_image

    def show_local_map(self):
        image = self.get_local_map()
        cv.imshow("Local Map", image)
        cv.waitKey(0)
        cv.destroyWindow("Local Map")

    def locate_cones(self):
        im_width = self[0].im_width
        for i, cone in enumerate(self):
            sensor_loc_mm = abs(im_width / 2 - cone.cx) * (consts.sensorWidth_mm / im_width)
            angle = cone.angle
            x_cs = cone.depth * np.tan(angle) - consts.camera_spacing_m / 2
            #CONVENTION: Before the car moves, the car space coordinate system is aligned with the OpenCv coordinate system, the origins are at the same point initially.
            point = Point(x_cs, 0, cone.depth)
            self[i].loc_cs = point



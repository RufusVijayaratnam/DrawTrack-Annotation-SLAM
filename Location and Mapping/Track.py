import numpy as np
from Detections import DetectedCone
import cv2 as cv
from Colour import Colour

class Track():
    
    def __init__(self):
        self.cones = []

    def update_track(self, cones):
        print("received %i cones" % len(cones))
        for cone in cones:
            self.cones.append(cone)

            
        
    def draw_track(self):
        im_size = 720
        blank_image = np.zeros((im_size, im_size, 3), np.uint8)
        blank_image[:] = (255, 255, 255)
        cv.circle(blank_image, (int(im_size / 2), int(im_size / 2)), 5, (0, 255, 0), thickness=-1)

        for cone in self.cones:
            cone_x = cone.loc_ws.x
            cone_z = cone.loc_ws.z
            colour = cone.colour.colour
            max_x = 40
            max_z = 40
            im_x = int(cone_x / max_x * (im_size / 2) + im_size / 2)
            im_y = int(cone_z / max_z * (im_size / 2) + im_size / 2)
            cv.circle(blank_image, (im_x, im_y), 5, colour, thickness=-1)

        cv.imshow("Track", blank_image)
        cv.waitKey(0)
        cv.destroyWindow("Track")

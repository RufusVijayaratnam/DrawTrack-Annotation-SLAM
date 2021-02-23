import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

im_size = 800
area_size_m = 80 #The area is a area_size_m x area_size_m area. Therefore each pixel represents im_size / area_size_m meters
blank_image = np.zeros((im_size, im_size, 3), np.uint8)
blank_image[:] = (255, 255, 255)

indicator_radius_px = 5 #radius of indicator size when clicking. 

track_directory = "/mnt/c/Users/Rufus Vijayaratnam/Documents/University/Year 3/IP/Blender/Resources/Tracks/"
track_name = "track2.txt"

#This is unencessary but I don't want to remove it because of the file reading.
isClosed = False
f = open(track_directory+track_name, 'w+')
f.write("Track\n")
if isClosed:
    f.write("closed\n")
else:
    f.write("open\n")

#Lines below sets 0.0 0.0 0.0 origin point
center_point = (int(im_size / 2), int(im_size / 2))
cv.circle(blank_image, center_point, indicator_radius_px, (0, 255, 0), -1)

f.write("p 0.0 0.0 0.0\n")

def write_point(file, point_x, point_y):
    point_x -= float(im_size / 2)
    point_y -= float(im_size / 2)
    """ point_x = point_x * grid_representation / grid_size * -1 #Just because
    point_y = point_y * grid_representation / grid_size """
    point_x /= (im_size / area_size_m)
    point_y /= (im_size / area_size_m)
    f.write("p %f %f 0.0 \r\n" % (point_x, point_y))


def add_point(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONUP:
        cv.circle(blank_image, (x, y), indicator_radius_px, (0, 0, 0), -1)
        write_point(f, x, y)
        cv.imshow("blank image", blank_image)

cv.imshow("blank image", blank_image)
cv.setMouseCallback("blank image", add_point)
cv.waitKey(0)
cv.destroyAllWindows()
f.close()
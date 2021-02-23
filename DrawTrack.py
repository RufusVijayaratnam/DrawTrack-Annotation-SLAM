import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

im_size = 720 * 3
blank_image = np.zeros((im_size, im_size, 3), np.uint8)
num_grids = 720 #Ideally this must be an integer factor of im_size
grid_size = int(im_size / num_grids)
blank_image[:] = (255, 255, 255)

blank_image[0:im_size:grid_size, :] = (0, 0, 0)
blank_image[:, 0:im_size:grid_size] = (0, 0, 0)


grid_representation = 2.5 #Physical distance representation between the centre points
# of two adjacent grid spaces in meters.
grid_center_points = np.zeros(num_grids - 1)
for i in range(num_grids - 1):
    grid_center_points[i] = grid_size / 2 + i * grid_size

track_directory = "/mnt/c/Users/Rufus Vijayaratnam/Documents/University/Year 3/IP/Blender/Resources/Tracks/"
track_name = "track5.txt"


isClosed = False
f = open(track_directory+track_name, 'w+')
f.write("Track\n")
if isClosed:
    f.write("closed\n")
else:
    f.write("open\n")

#Lines below sets 0.0 0.0 0.0 origin point
p1 = int(im_size / 2 - int(grid_size / 2))
p2 = int(im_size / 2 + int(grid_size / 2))
blank_image[p1:p2, p1:p2] = (0, 0, 0)

f.write("p 0.0 0.0 0.0\n")

def write_point(file, point_x, point_y):
    point_x -= float(im_size / 2)
    point_y -= float(im_size / 2)
    point_x = point_x * grid_representation / grid_size * -1 #Just because
    point_y = point_y * grid_representation / grid_size
    f.write("p %f %f 0.0 \r\n" % (point_x, point_y))


def add_point(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONUP:
        x_idx = (np.abs(grid_center_points - x)).argmin()
        y_idx = (np.abs(grid_center_points - y)).argmin()
        point_x = grid_center_points[x_idx]
        point_y = grid_center_points[y_idx]
        p1x = int(point_x - grid_size / 2)
        p1y = int(point_y - grid_size / 2)
        p2x = int(point_x + grid_size / 2)
        p2y = int(point_y + grid_size / 2)
        blank_image[p1y:p2y, p1x:p2x] = (0, 0, 0)
        write_point(f, point_x, point_y)
        cv.imshow("blank image", blank_image)


cv.imshow("blank image", blank_image)
cv.setMouseCallback("blank image", add_point)
cv.waitKey(0)
cv.destroyAllWindows()
f.close()
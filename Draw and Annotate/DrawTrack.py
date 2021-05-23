import numpy as np
import cv2 as cv
import argparse
import os
#from matplotlib import pyplot as plt

def write_point(point_x, point_y):
    point_x -= float(im_size / 2)
    point_y -= float(im_size / 2)
    """ point_x = point_x * grid_representation / grid_size * -1 #Just because
    point_y = point_y * grid_representation / grid_size """
    point_x /= -(im_size / area_size_m) #Coordinate convention
    point_y /= (im_size / area_size_m)
    f.write("p %f %f 0.0\n" % (point_x, point_y))


def add_point(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONUP:
        cv.circle(blank_image, (x, y), indicator_radius_px, (0, 0, 0), -1)
        write_point(x, y)
        cv.imshow("blank image", blank_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trackname", type=str, help="Name of the track to be created")
    parser.add_argument("--startpos", metavar=("x_initial", "y_initial"), nargs=2, type=float, default=[0.0, 0.0], help="Desired start position of the track, represented by two floats between -50.0 and 50.0")
    parser.add_argument("--initialpos", metavar="'False'", type=bool, default=True, help="Add initial point where green point is shown. Default is True")
    args = parser.parse_args()
    im_size = 800
    area_size_m = 80 #The area is a area_size_m x area_size_m area. Therefore each pixel represents im_size / area_size_m meters
    blank_image = np.zeros((im_size, im_size, 3), np.uint8)
    blank_image[:] = (255, 255, 255)
    indicator_radius_px = 5 #radius of indicator size when clicking. 

    track_name = args.trackname + ".txt"
    track_directory = os.path.abspath("../Blender/Resources/Tracks")
    #track_name = "example.txt"

    #This is unencessary but I don't want to remove it because of the file reading.
    isClosed = False
    f = open(os.path.join(track_directory, track_name), 'w+')
    f.write("Track\n")
    if isClosed:
        f.write("closed\n")
    else:
        f.write("open\n")

    #Lines below sets  origin point
    start_pos = args.startpos
    sx = -start_pos[0]
    sy = -start_pos[1]
    center_point = (int(im_size / 2 + sx / area_size_m * im_size / 2), int(im_size / 2 - sy / area_size_m * im_size / 2))
    cv.circle(blank_image, center_point, indicator_radius_px, (0, 255, 0), -1)

    
    if args.initialpos == True: write_point(int(im_size / 2 + sx / area_size_m * im_size / 2), int(im_size / 2 - sy / area_size_m * im_size / 2))

    cv.imshow("blank image", blank_image)
    cv.setMouseCallback("blank image", add_point)
    while True:
        k = cv.waitKey(0) & 0xFF
        if k == 27:
            cv.destroyAllWindows()
            break

    f.close()
    
    


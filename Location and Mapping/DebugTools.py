import cv2 as cv
import numpy as np
import sys
import os
sys.path.append(os.path.abspath("~/../../Draw and Annotate/"))
import LoadTrack as lt

rsrc = os.path.abspath("~/../../Blender/Resources/")

def hstack_images(im1, im2, name="hstack"):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = cv.resize(im1, (0, 0), None, .5, .5)
    im2 = cv.resize(im2, (0, 0), None, .5, .5)
    stacked = np.hstack((im1, im2))
    return stacked

def vstack_images(im2, im1, name="hstack"):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = cv.resize(im1, (0, 0), None, 0.25, 0.25)
    im2 = cv.resize(im2, (0, 0), None, 0.25, 0.25)
    stacked = np.vstack((im1, im2))
    return stacked

def hvstack_images(t1, t2, t3, t4, name="4 stack"):
    t1 = cv.resize(t1, (0, 0), None, 0.25, 0.25)
    t2 = cv.resize(t2, (0, 0), None, 0.25, 0.25)
    t3 = cv.resize(t3, (0, 0), None, 0.25, 0.25)
    t4 = cv.resize(t4, (0, 0), None, 0.25, 0.25)
    hstack = np.hstack((t1, t2))
    vstack = np.vstack((t3, t4))
    stacked = np.vstack((vstack, hstack))
    cv.imshow(name, stacked)
    cv.waitKey(0)
    cv.destroyWindow(name)

def show_image(name, image):
    cv.imshow(name, image)
    while True:
        k = cv.waitKey(0) & 0xFF
        if k == 27:
            cv.destroyWindow(name)
            break

def show_video(name, image):
    cv.imshow("Video", image)

def get_track_overhead(track_name, image=None):
    #From track file so using Blender space for simplicity
    path = rsrc + "/Tracks/" + track_name + ".txt"
    blue_cones, yellow_cones = lt.load_cones(path)
    im_size = 720
    if type(image) == np.ndarray:
        blank_image = np.array(image)
    else:
        blank_image = np.zeros((im_size, im_size, 3), np.uint8)
        blank_image[:] = (128, 128, 128)

    area = 80
    mid = int(im_size / 2)
    cv.circle(blank_image, (mid, mid), 4, (0, 255, 0), thickness=-1)
    for b, y in zip(blue_cones, yellow_cones):
        bx = -b[0]
        bz = b[1]
        yx = -y[0]
        yz = y[1]

        cx = int(mid + bx / (area / 2) * mid)
        cz = int(mid + bz / (area / 2) * mid)
        cv.circle(blank_image, (cx, cz), 4, (255, 0, 0), thickness=-1)
        cx = int(mid + yx / (area / 2) * mid)
        cz = int(mid + yz / (area / 2) * mid)
        cv.circle(blank_image, (cx, cz), 4, (0, 255, 255), thickness=-1)
    return blank_image
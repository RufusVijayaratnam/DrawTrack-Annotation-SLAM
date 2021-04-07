import cv2 as cv
import numpy as np

def hstack_images(im1, im2, name="hstack"):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = cv.resize(im1, (0, 0), None, .5, .5)
    im2 = cv.resize(im2, (0, 0), None, .5, .5)
    stacked = np.hstack((im1, im2))
    cv.imshow(name, stacked)
    cv.waitKey(0)
    cv.destroyWindow(name)

def vstack_images(im2, im1, name="hstack"):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = cv.resize(im1, (0, 0), None, .5, .5)
    im2 = cv.resize(im2, (0, 0), None, .5, .5)
    stacked = np.vstack((im1, im2))
    cv.imshow(name, stacked)
    cv.waitKey(0)
    cv.destroyWindow(name)

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
import numpy as np
import cv2 as cv
import sys
sys.path.append("/mnt/c/Users/Rufus Vijayaratnam/Driverless/Draw and Annotate/")
sys.path.append("/mnt/c/Users/Rufus Vijayaratnam/Driverless/Location and Mapping/")
import LoadTrack as lt
import Annotate as anno
import Detections
import Matching
import Colour
import Mapping
import DebugTools as dbgt
import matplotlib.pyplot as plt

eval = "custom"
fig, ax1 = plt.subplots()
fig.set_size_inches(8, 6)
ax1.set_ylabel("Error (m)")
ax1.set_xlabel("Distance (m)")

left_im = ["images/track1-Left_Cam-Render-1.png", "images/track1-Left_Cam-Render-18.png", "images/track1-Left_Cam-Render-34.png", "images/track2-Left_Cam-Render-1.png", "images/track2-Left_Cam-Render-6.png", "images/track3-Left_Cam-Render-1.png", "images/track3-Left_Cam-Render-49.png", "images/track3-Left_Cam-Render-104.png", "images/track4-Left_Cam-Render-1.png", "images/track4-Left_Cam-Render-36.png"]
right_im = ["images/track1-Right_Cam-Render-1.png","images/track1-Left_Cam-Render-19.png", "images/track1-Right_Cam-Render-34.png", "images/track2-Right_Cam-Render-1.png", "images/track2-Right_Cam-Render-6.png", "images/track3-Right_Cam-Render-1.png", "images/track3-Right_Cam-Render-49.png", "images/track3-Right_Cam-Render-104.png", "images/track4-Right_Cam-Render-1.png", "images/track4-Right_Cam-Render-36.png"]
left_label = ["labels/track1-Left_Cam-Render-1.txt", "labels/track1-Left_Cam-Render-18.txt", "labels/track1-Left_Cam-Render-34.txt", "labels/track2-Left_Cam-Render-1.txt", "labels/track2-Left_Cam-Render-6.txt", "labels/track3-Left_Cam-Render-1.txt", "labels/track3-Left_Cam-Render-49.txt", "labels/track3-Left_Cam-Render-104.txt", "labels/track4-Left_Cam-Render-1.txt", "labels/track4-Left_Cam-Render-36.txt"]
right_label = ["labels/track1-Right_Cam-Render-1.txt", "labels/track1-Left_Cam-Render-19.txt", "labels/track1-Right_Cam-Render-34.txt", "labels/track2-Right_Cam-Render-1.txt", "labels/track2-Right_Cam-Render-6.txt", "labels/track3-Right_Cam-Render-1.txt", "labels/track3-Right_Cam-Render-49.txt", "labels/track3-Right_Cam-Render-104.txt", "labels/track4-Right_Cam-Render-1.txt", "labels/track4-Right_Cam-Render-36.txt"]
left_real = ["labels/Real-track1-Left_Cam-Render-1.txt", "labels/Real-track1-Left_Cam-Render-18.txt", "labels/Real-track1-Left_Cam-Render-34.txt", "labels/Real-track2-Left_Cam-Render-1.txt", "labels/Real-track2-Left_Cam-Render-6.txt", "labels/Real-track3-Left_Cam-Render-1.txt", "labels/Real-track3-Left_Cam-Render-49.txt", "labels/Real-track3-Left_Cam-Render-104.txt", "labels/Real-track4-Left_Cam-Render-1.txt", "labels/Real-track4-Left_Cam-Render-36.txt"]
colours = ["blue", "yellow", "green", "red", "purple", "orange", "brown", "pink", "grey", "cyan"]

rsrc_path = "/mnt/c/Users/Rufus Vijayaratnam/Driverless/Blender/Resources/Renders/test/"

"""
track1 1, 18, 34
track2 1, 6
track3 1, 49, 104
track4 1, 36
"""
max = 100
matches = []

for run in range(2, 3):

    left_image_path = left_im[run]
    right_image_path = right_im[run]
    left_label_path = left_label[run]
    right_label_path = right_label[run]
    left_real_path = left_real[run]
    plot_colour = colours[run]

    def sort_arrays(arry1, arry2):
        arry1, arry2 = (list(t) for t in zip(*sorted(zip(arry1, arry2))))
        return arry1, arry2

    train_im = np.array(cv.imread(rsrc_path + left_image_path))
    query_im = np.array(cv.imread(rsrc_path + right_image_path))
    im_width = np.shape(train_im)[1]
    im_height = np.shape(train_im)[0]
    detections = []
    with open(rsrc_path + left_label_path) as f:
        for box in f.readlines():
            box = box.split()
            box = [float(val) for val in box]
            cx = int(box[1] * im_width)
            cy = int(box[2] * im_height)
            w = int(box[3] * im_width)
            h = int(box[4] * im_height)
            detected_cone = Detections.DetectedCone(cx, cy, w, h)
            detections.append(detected_cone)
    train = Detections.Detections(detections, train_im, max_dist=max)
    detections = []

    with open(rsrc_path + right_label_path) as f:
        for box in f.readlines():
            box = box.split()
            box = [float(val) for val in box]
            w = int(box[3] * im_width)
            h = int(box[4] * im_height)
            cx = int(box[1] * im_width)
            cy = int(box[2] * im_height)
            detected_cone = Detections.DetectedCone(cx, cy, w, h)
            detections.append(detected_cone)
    query = Detections.Detections(detections, query_im, max_dist=max)
    detections = []

    with open(rsrc_path + left_real_path) as r:
        for real in r.readlines():
            real = real.split()
            colour = real[0]
            detected_cone = Detections.DetectedCone(0, 0, 0, 0)
            if colour == "blue":
                detected_cone.colour = Colour.blue
            elif colour == "yellow":
                detected_cone.colour = Colour.yellow
            
            detected_cone.loc_cs = Detections.Point(float(real[1]), float(0), float(real[3]))
            detections.append(detected_cone)
    real_cones = Detections.Detections(detections, max_dist=max)
    real_cones.image = train.image

    for i, t in enumerate(train):
        real_cones[i].cx = t.cx
        real_cones[i].cy = t.cy
        real_cones[i].w = t.w
        real_cones[i].h = t.h
    for i, real in enumerate(real_cones):
        train[i].colour = real.colour
        query[i].colour = real.colour
        train[i].depth = real.loc_cs.x
        query[i].depth = real.loc_cs.x - 0.1
        real_cones[i].cx = train[i].cx
        real_cones[i].cy = train[i].cy
        real_cones[i].w = train[i].w
        real_cones[i].h = train[i].h

    train.init_detections()
    query.init_detections()
    real_cones.init_detections()
    real_cones = real_cones.filter_distance()
    train = train.filter_distance()
    query = query.filter_distance()
    real_local = real_cones.get_local_map()


    ##################################
    #Custom Matching Evaluation
    if eval == "custom":
        matcher = Matching.FrameMatcher(train, query)
        matcher.find_subsequent_matches()
        le_im = matcher.get_matches_image()
        #matcher.calculate_depth()
        train_matched, _ = matcher.get_matched()
        train_matched.locate_cones()
        matched_local = train_matched.get_local_map()
        #dbgt.hstack_images(np.array(real_local), np.array(matched_local))
        cv.imwrite("real_overhead.png", real_local)
        cv.imwrite("custom_overhead.png", matched_local)
        errors = []
        dists = []
        hashes = np.array([q.uid for q in real_cones])
        indices = []
        for cone in train_matched:
            index = np.where(hashes == cone.uid)[0][0]
            indices.append(index)
            dist = cone.loc_cs - real_cones[index].loc_cs
            abs_dist = dist.mag()
            errors.append(abs_dist)

        areas = matcher.draw_search_areas()
        

        avg_error = np.average(np.array(errors))
        #print("Test: %d, average error: %f" % ((run + 1, avg_error)))
        real_detects = real_cones.take(indices)
        dists = [cone.loc_cs.mag() for cone in real_detects]
        dists, errors = sort_arrays(dists, errors)
        avg = np.average(np.array(errors))
        ax1.plot(dists, errors, color=plot_colour, label="Test %d" % (run + 1))
        ax1.legend()
        ax1.set_title("Error vs Distance Crom Car for Cone Matching Method")

        #fig.savefig("ConeLocationError-%s.png" % eval, dpi=200)
    #percent = len(train_matched) / len(real_cones) * 100
    #print("Total cones: %d, matched percent: %d" % (len(real_cones), percent))



    ################### Evaluation

    if eval == "orb":
        orb = cv.ORB_create(nfeatures=5000)
        kp_query = orb.detect(query_im, None)
        kp_train = orb.detect(train_im, None)
        kp_query, des_query = orb.compute(query_im, kp_query)
        kp_train, des_train = orb.compute(train_im, kp_train)
        point_matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        point_matches = point_matcher.match(des_query, des_train)
        point_matches = sorted(point_matches, key=lambda x:x.distance)

        img3 = cv.drawMatches(train_im,kp_train,query_im,kp_query,point_matches[:100],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        dbgt.show_image("matches", img3)

        for match in point_matches:
            tidx = match.trainIdx
            qidx = match.queryIdx
            Q_x = kp_query[qidx].pt[0]
            T_x = kp_train[tidx].pt[0]
            Q_y = kp_query[qidx].pt[1]
            T_y = kp_train[tidx].pt[1]
            train.point_depth((T_x, T_y), (Q_x, Q_y))
        for i, cone in enumerate(train):
            depths = np.array(cone.depths)
            m = 0.2
            if len(depths) > 0:
                depths = depths[abs(depths - np.mean(depths)) < m * np.std(depths)]
            if len(depths) > 0:
                avg_depth = np.average(depths)
                train[i].depth = avg_depth
        train.locate_cones()
        located = train.remove_unlocated()
        local_map = located.get_local_map()

        percent = len(located) / len(real_cones) * 100
        matches.append(percent)
        #print("Test: %d,Total cones: %d, matched percent: %d" % ((run + 1), len(real_cones), percent))

        cv.imwrite("orb_overhead.png", local_map)

        errors = []
        dists = []
        hashes = np.array([q.uid for q in real_cones])
        indices = []
        for cone in located:
            index = np.where(hashes == cone.uid)[0][0]
            indices.append(index)
            dist = cone.loc_cs - real_cones[index].loc_cs
            abs_dist = dist.mag()
            errors.append(abs_dist)
        avg_error = np.average(np.array(errors))
        #print("Test: %d, average error: %f" % ((run + 1, avg_error)))
        print("cones: %d" % len(real_cones))
        real_detects = real_cones.take(indices)
        dists = [cone.loc_cs.mag() for cone in real_detects]
        dists, errors = sort_arrays(dists, errors)
        avg = np.average(np.array(errors))
        ax1.plot(dists, errors, color=plot_colour, label="Test %d" % (run + 1))
        ax1.legend()
                
        ax1.set_title("Error vs Distance Crom Car for Full ORB Matching Method")
        #fig.savefig("ConeLocationError-%s.png" % eval, dpi=200)

        #1,  18

        average = np.mean(np.array(matches))
        #print("average percent: ", average)
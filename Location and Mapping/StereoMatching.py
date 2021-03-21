import numpy as np
import cv2 as cv
import ColourEstimation as ce
from Constants import *
from Detections import *

class Matcher():
    def __init__(self, train, query):
        self.train = train
        self.query = query
        self.matches = []

    def __in_range(self, test, lower, upper):
        if test <= upper and test >= lower:
            return True
        else:
            return False

    def __match_hash(self, hash, array):
        hashes = np.array([q.uid for q in array])
        index = np.where(hashes == hash)[0][0]
        return index

    def __find_match(self, train, query_original):
        for i, t in enumerate(train):
            query = query_original
            tcx = t.cx
            tcy = t.cy
            hash = t.uid
            _, disp_lower, disp_upper = t.monocular_distance_estimate()
            dists = np.array([abs(q.cx - tcx) for q in query])
            in_range = np.array([self.__in_range(d, disp_lower, disp_upper) for d in dists])
            out_range = np.where(in_range == False)
            query = np.delete(query, out_range)
            train_index = self.__match_hash(hash, self.train)
            if len(query) != 0:
                cy_dists = np.array([abs(q.cy - tcy) for q in query])
                min_cy_dist = np.min(cy_dists)
                match_idx = np.where(cy_dists == min_cy_dist)[0][0]
                match_hash = query[match_idx].uid
                query_index = self.__match_hash(match_hash, self.query)
                self.matches.append((train_index, query_index))
            else:
                self.matches.append((train_index, -1))

        return 0

    def find_stereo_matches(self):
        colours = np.array([c.colour.name for c in self.query])
        yellow_indices = np.where(colours != "yellow")
        blue_indices = np.where(colours != "blue")
        query_yellow = np.delete(self.query, blue_indices)
        query_blue = np.delete(self.query, yellow_indices)

        colours = np.array([c.colour.name for c in self.train])
        yellow_indices = np.where(colours != "yellow")
        blue_indices = np.where(colours != "blue")
        train_yellow = np.delete(self.train, blue_indices)
        train_blue = np.delete(self.train, yellow_indices)

        self.__find_match(train_blue, query_blue)
        self.__find_match(train_yellow, query_yellow)

    def __match_keypoints(self, train_im, query_im):
        print("Attempting to find matching points")
        train_im = np.array(train_im)
        query_im = np.array(query_im)
        orb = cv.ORB_create(nfeatures=20)
        kp_query = orb.detect(query_im, None)
        kp_train = orb.detect(train_im, None)

        kp_query, des_query = orb.compute(query_im, kp_query)
        kp_train, des_train = orb.compute(train_im, kp_train)

        point_matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        point_matches = point_matcher.match(des_query, des_train)
        point_matches = sorted(point_matches, key=lambda x:x.distance)
        

        """ cv.imshow("train", train_im)
        cv.imshow("query", query_im)
        img3 = cv.drawMatches(train_im,kp_train,query_im,kp_query,point_matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow("matches", img3)
        cv.waitKey(0)
        cv.destroyAllWindows() """

        return point_matches

    def deriveDepth(key_points):    
        depths = []
        for key_point in key_points:
            imgWidth = QueryCam.shape[1]
            focalLength_pixels = (focalLength_mm / sensorWidth_mm) * imgWidth
            tidx = key_point.trainIdx
            qidx = key_point.queryIdx
            imIdx = key_point.imgIdx
            Q_x = kpQuery[qidx].pt[0]
            T_x = kpTrain[tidx].pt[0]
            disparity = abs(Q_x - T_x)
            print("tidx: {}, qidx: {}".format(tidx, qidx))
            depth = baseline_mm * focalLength_pixels / disparity
            depths.append(depth)

        return depths

    def __key_points_to_image_coords(self):
        #the matching function returns key points with their coords in the sub image
        #These must be transformed back into the original image coords
        return 0

    def calculate_depth(self):
        point_matches = []
        print("what up dog")
        for i, match in enumerate(self.matches):
            train_im = np.array(self.train.image)
            query_im = np.array(self.query.image)
            train_idx = match[0]
            query_idx = match[1]
            if train_idx != -1 and query_idx != -1:
                train_sub_im = np.array(self.train[train_idx].get_sub_image(train_im))
                query_sub_im = np.array(self.query[query_idx].get_sub_image(query_im))
                point_matches.append(self.__match_keypoints(train_sub_im, query_sub_im))
        print(point_matches)

                


""" if train_idx != -1 and query_idx != -1:
        x1 = int(train_sub_im.cx - train_sub_im.w / 2)
        y1 = int(train_sub_im.cy - train_sub_im.h / 2)
        x2 = int(train_sub_im.cx + train_sub_im.w / 2)
        y2 = int(train_sub_im.cy + train_sub_im.h / 2)
        cv.rectangle(train_im, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        print(x1, y1, x2, y2)
        x1 = int(query_sub_im.cx - query_sub_im.w / 2)
        y1 = int(query_sub_im.cy - query_sub_im.h / 2)
        x2 = int(query_sub_im.cx + query_sub_im.w / 2)
        y2 = int(query_sub_im.cy + query_sub_im.h / 2)
        print(x1, y1, x2, y2)
        cv.rectangle(query_im, (x1, y1), (x2, y2), (0, 255, 255), thickness=2)

        cv.imshow("train", train_im)
        cv.imshow("query", query_im)
        cv.waitKey(0)
        cv.destroyAllWindows() """
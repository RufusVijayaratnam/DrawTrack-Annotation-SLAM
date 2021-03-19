import numpy as np
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
            print("disp lower: %f \n disp upper: %f" % (disp_lower, disp_upper))
            dists = np.array([abs(q.cx - tcx) for q in query])
            print("distances: \n", dists)
            in_range = np.array([self.__in_range(d, disp_lower, disp_upper) for d in dists])
            print("in range: \n", in_range)
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

        return self.matches


def find_stereo_depth(query_im, train_im):
    orb = cv.ORB_create(nfeatures=20)
    kp_query = orb.detect(query_img, None)
    kp_train = orb.detect(train_im, None)

    kp_query, des_query = orb.compute(query_img, kp_query)
    kp_train, des_train = orb.compute(train_im, kp_train)

    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_query, des_train)
    matches = sorted(matches, key=lambda x:x.distance)
    matches = matches[:20]

def deriveDepth(matchPoint):
    imgWidth = QueryCam.shape[1]
    focalLength_pixels = (focalLength_mm / sensorWidth_mm) * imgWidth
    tidx = matchPoint.trainIdx
    qidx = matchPoint.queryIdx
    imIdx = matchPoint.imgIdx
    Q_x = kpQuery[qidx].pt[0]
    T_x = kpTrain[tidx].pt[0]
    disparity = abs(Q_x - T_x)
    print("tidx: {}, qidx: {}".format(tidx, qidx))
    depth = baseline_mm * focalLength_pixels / disparity
    return depth

yo = Matcher(rds, lds)
yo.find_stereo_matches()
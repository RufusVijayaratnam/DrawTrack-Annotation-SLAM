import numpy as np
import cv2 as cv
import ColourEstimation as ce
from Constants import *
from Detections import DetectedCone, Detections
from Colour import Colour
import DebugTools as dbgt

class Matcher():
    def __init__(self, train, query):
        self.train = train
        self.query = query
        self.matches = []

    def in_range(self, test, lower, upper):
        if test <= upper and test >= lower:
            return True
        else:
            return False

    def match_hash(self, hash, array):
        hashes = np.array([q.uid for q in array])
        index = np.where(hashes == hash)[0][0]
        return index

    def match_keypoints(self, train_im, query_im):
        train_im = np.array(train_im)
        query_im = np.array(query_im)
        orb = cv.ORB_create(nfeatures=5)
        kp_query = orb.detect(query_im, None)
        kp_train = orb.detect(train_im, None)

        kp_query, des_query = orb.compute(query_im, kp_query)
        kp_train, des_train = orb.compute(train_im, kp_train)

        point_matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        point_matches = point_matcher.match(des_query, des_train)
        point_matches = sorted(point_matches, key=lambda x:x.distance)
        
        """ img3 = cv.drawMatches(train_im,kp_train,query_im,kp_query,point_matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow("matches", img3)
        cv.waitKey(0)
        cv.destroyWindow("matches") """


        if len(point_matches) > 0:
            tidx = point_matches[0].trainIdx
            qidx = point_matches[0].queryIdx
            Q_x = kp_query[qidx].pt[0]
            T_x = kp_train[tidx].pt[0]
            return T_x, Q_x
        else:
            tcx = np.shape(train_im)[1] / 2
            qcx = np.shape(query_im)[1] / 2
            return tcx, qcx

    def get_colour_filtered_detections(self):
        colours = np.array([c.colour.name for c in self.query])
        not_yellow_indices = np.where(colours != "yellow")
        not_blue_indices = np.where(colours != "blue")
        query_yellow = np.delete(self.query, not_yellow_indices)
        query_blue = np.delete(self.query, not_blue_indices)

        colours = np.array([c.colour.name for c in self.train])
        not_yellow_indices = np.where(colours != "yellow")
        not_blue_indices = np.where(colours != "blue")
        train_yellow = np.delete(self.train, not_yellow_indices)
        train_blue = np.delete(self.train, not_blue_indices)
        return train_blue, query_blue, train_yellow, query_yellow
    
    """ def kp_matching(self):
        train_im = self.train.image
        query_im = self.query.image
        matches = self.matches
        for match in matches:
            train = self.train[match[0]]
            query = self.query[match[1]]
            ty1 = int(train.cy - train.h / 2)
            ty2 = int(train.cy + train.h / 2)
            tx1 = int(train.cx - train.w / 2)
            tx2 = int(train.cx + train.w / 2)
            qy1 = int(query.cy - query.h / 2)
            qy2 = int(query.cy + query.h / 2)
            qx1 = int(query.cx - query.w / 2)
            qx2 = int(query.cx + query.w / 2)
            train_sub = train_im[ty1:ty2, tx1:tx2]
            query_sub = query_im[qy1:qy2, qx1:qx2]
            #dbgt.hstack_images(train_sub, query_sub)
            tx, qx = self.__match_keypoints(train_sub, query_sub)
            cx1 = tx1 + tx
            cx2 = qx1 + qx
            depth
            """
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
            depth = baseline_mm * focalLength_pixels / disparity
            depths.append(depth)

        return depths

    def center_depth(self, cx1, cx2):
        disparity = abs(cx1 - cx2)
        focalLength_pixels = (focalLength_mm / sensorWidth_mm) * self.train[0].im_width
        depth = baseline_mm * focalLength_pixels / disparity
        return depth / 1000


    def get_matched(self):
        train_matched = [self.train[val[0]] for val in self.matches]
        query_matched = [self.query[val[1]] for val in self.matches]
        train_located = Detections(train_matched, self.train.image)
        query_located = Detections(query_matched, self.query.image)
        return train_located, query_located

    def show_single_matches(self):
        for i, match in enumerate(self.matches):
            train_im = self.train.image.copy()
            query_im = self.query.image.copy()
            cone = self.train[match[0]]
            x1 = int(cone.cx - cone.w / 2)
            y1 = int(cone.cy - cone.h / 2)
            x2 = int(cone.cx + cone.w / 2)
            y2 = int(cone.cy + cone.h / 2)
            p1 = tuple([x1, y1])
            p2 = tuple([x2, y2])
            cv.rectangle(train_im, p1, p2, (0, 255, 0))
            
            cone = self.query[match[0]]
            x1 = int(cone.cx - cone.w / 2)
            y1 = int(cone.cy - cone.h / 2)
            x2 = int(cone.cx + cone.w / 2)
            y2 = int(cone.cy + cone.h / 2)
            p1 = tuple([x1, y1])
            p2 = tuple([x2, y2])
            cv.rectangle(query_im, p1, p2, (0, 255, 0))

            name = "Match: %d of %d" % (i, len(self.matches))
            dbgt.hstack_images(train_im, query_im, name=name)

           

class StereoMatcher(Matcher):
    def __init__(self, train, query):
        super().__init__(train, query)

    def __find_match(self, train, query_original):
        for i, t in enumerate(train):
            query = query_original
            tcx = t.cx
            tcy = t.cy
            hash = t.uid
            _, disp_lower, disp_upper = t.monocular_distance_estimate()
            dists = np.array([abs(q.cx - tcx) for q in query])
            in_range = np.array([self.in_range(d, disp_lower, disp_upper) for d in dists])
            out_range = np.where(in_range == False)
            query = np.delete(query, out_range)
            train_index = self.match_hash(hash, self.train)

            if len(query) != 0:
                cy_dists = np.array([abs(q.cy - tcy) for q in query])
                min_cy_dist = np.min(cy_dists)
                match_idx = np.where(cy_dists == min_cy_dist)[0][0]
                match_hash = query[match_idx].uid
                query_index = self.match_hash(match_hash, self.query)
                self.matches.append((train_index, query_index))
            else:
                self.matches.append((train_index, -1))

    def find_stereo_matches(self):
        train_blue, query_blue, train_yellow, query_yellow = self.get_colour_filtered_detections()
        self.__find_match(train_blue, query_blue)
        self.__find_match(train_yellow, query_yellow)
        self.matches = [val for val in self.matches if val[1] != -1]

    def calculate_depth(self):
        train_im = np.array(self.train.image)
        query_im = np.array(self.query.image)
        for i, match in enumerate(self.matches):
            train_idx = match[0]
            query_idx = match[1]
            """
            cx_train = self.train[train_idx].cx
            cx_query = self.query[query_idx].cx
            depth = self.center_depth(cx_train, cx_query)
            self.train[train_idx].depth = depth
            self.query[query_idx].depth = depth """

            train = self.train[match[0]]
            query = self.query[match[1]]
            ty1 = int(train.cy - train.h / 2)
            ty2 = int(train.cy + train.h / 2)
            tx1 = int(train.cx - train.w / 2)
            tx2 = int(train.cx + train.w / 2)
            qy1 = int(query.cy - query.h / 2)
            qy2 = int(query.cy + query.h / 2)
            qx1 = int(query.cx - query.w / 2)
            qx2 = int(query.cx + query.w / 2)
            train_sub = train_im[ty1:ty2, tx1:tx2]
            query_sub = query_im[qy1:qy2, qx1:qx2]
            #dbgt.hstack_images(train_sub, query_sub)
            tx, qx = self.match_keypoints(train_sub, query_sub)
            cx1 = tx1 + tx
            cx2 = qx1 + qx
            depth = self.center_depth(cx1, cx2)
            self.train[train_idx].depth = depth
            self.query[query_idx].depth = depth

    def show_matches(self, name="subsequent matches"):
        train_im = self.train.get_annotated_image()
        query_im = self.query.get_annotated_image()
        train_im = cv.resize(train_im, (0, 0), None, .5, .5)
        query_im = cv.resize(query_im, (0, 0), None, .5, .5)

        stacked = np.hstack((query_im, train_im))
        im_width = np.shape(stacked)[1] / 2
        train, query = self.get_matched()
        for t, q in zip(train, query):
            p1 = (int(t.cx / 2), int(t.cy / 2))
            p2 = (int(q.cx / 2 + im_width), int(q.cy / 2))
            cv.line(stacked, p1, p2, t.colour.colour, thickness=1)

        cv.imshow(name, stacked)
        cv.waitKey(0)
        cv.destroyWindow(name)
            

class FrameMatcher(Matcher):
    def __init__(self, train, query):
        #When this is called all cones should have depth and point information filled in.
        super().__init__(train, query)

    def __find_match(self, train, query_original):
        #self.get_car_speed()
        #Here I should use current vehicle speed to calculate maximum possible
        #travel of the car, and thereforce can estimate a viable depth for the cone 
        #because at this stage we have stereo depth information on the cones.
        #Possibility to also use IMU for angle change, something for next year probably.
        #For track 8, used 16 substeps, max cone spacing was 5 meters
        #Therefor max travel is 5 / 16
        #Train represents original frame (prev frame)
        #Query represents new frame
        #distance to a cone MUST be negative, or it should be discarded
        frame_time = 13 / 320
        v = 5
        dist = frame_time * v
        delta_depth_max = dist * 2
        delta_depth_min =  dist / 2
        
        for i, t in enumerate(train):
            """ if i == 1:
                train[:1].show_annotated_image()
                query_original.show_annotated_image() """
            t_depth = t.depth
            query = query_original
            hash = t.uid
            delta_depths = [(q.depth - t_depth) for q in query_original]
            #Line below will discard all positive depth change values.
            in_range = np.array([self.in_range(-d, delta_depth_min, delta_depth_max) for d in delta_depths])
            out_range = np.where(in_range == False)
            query = np.delete(query, out_range)
            train_index = self.match_hash(hash, self.train)

            if len(query) == 1:
                match_hash = query[0].uid
                query_index = self.match_hash(match_hash, self.query)
                self.matches.append((train_index, query_index))
            elif len(query) > 1:
                self.matches.append((train_index, -1))
            else:
                self.matches.append((train_index, -1))

    def handle_matched_cones(self):
        for val in self.matches:
            if not self.train[val[0]].unmapped:
                self.query[val[1]].unmapped = False

    def find_subsequent_matches(self):
        train_blue, query_blue, train_yellow, query_yellow = self.get_colour_filtered_detections()
        self.__find_match(train_blue, query_blue)
        self.__find_match(train_yellow, query_yellow)
        self.matches = [val for val in self.matches if val[1] != -1]
        self.handle_matched_cones()

    def show_matches(self, name="subsequent matches"):
        train, query = self.get_matched()
        train_im = train.get_annotated_image()
        query_im = query.get_annotated_image()
        train_im = cv.resize(train_im, (0, 0), None, .5, .5)
        query_im = cv.resize(query_im, (0, 0), None, .5, .5)

        stacked = np.vstack((query_im, train_im))
        im_height = int(np.shape(stacked)[0] / 2)
        for t, q in zip(train, query):
            p1 = (int(t.cx / 2), int((t.cy / 2) + im_height))
            p2 = (int(q.cx / 2), int(q.cy / 2))
            if t.unmapped == False:
                cv.circle(stacked, p1, 4, (0, 255, 0), thickness=-1)
            if q.unmapped == False:
                cv.circle(stacked, p2, 4, (0, 255, 0), thickness=-1)
            cv.line(stacked, p1, p2, t.colour.colour, thickness=2)

        cv.imshow(name, stacked)
        cv.waitKey(0)
        cv.destroyWindow(name)

        

            

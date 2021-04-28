import numpy as np
from Detections import DetectedCone, Detections, Vec3
import Matching
from Colour import Colour
import Constants as consts
import cv2 as cv
import torch
import importlib
import Track
import DebugTools as dbgt
import os
importlib.reload(Matching)
importlib.reload(Track)
importlib.reload(dbgt)

def gen_rotation_matrix(alpha, beta, gamma):
    rz = np.matrix([[np.cos(gamma), -np.sin(gamma),  0],
                    [np.sin(gamma),  np.cos(gamma),  0],
                    [0,                          0,  1]])

    ry = np.matrix([[np.cos(beta),  0, np.sin(beta)],
                    [0,             1,             0],
                    [-np.sin(beta), 0, np.cos(beta)]])


    rx = np.matrix([[1,            0,               0],
                    [0, np.cos(alpha), -np.sin(alpha)], 
                    [0, np.sin(alpha), np.cos(alpha)]])

    rotation_matrix = rz * ry * rx
    return rotation_matrix
        

class Mapper():
    def __init__(self, src_train, src_query):
        self.prev_cones = None
        self.new_cones = None
        self.mapped_cones = None
        
        
        #DETECTION
        yolov5_path = os.path.abspath("~/../../../../Rufus Vijayaratnam/yolov5/runs/train/exp8/weights/best.pt")
        weights_path = yolov5_path + ""
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path_or_model=weights_path)

        #VID PATHS
        self.src_train = src_train
        self.src_query = src_query

        #MOTION DERIVATION
        self.car_loc_ws = Vec3(0, 0, 0)
        self.car_rotation = 0
        self.positions = [Vec3(0, 0, 0)]
        self.frame_skips = 1 #No frames skipped




    def stereo_process_new_frames(self, imgs):
        imgs = [np.array(img) for img in imgs]
        results = self.model(imgs, size=640)

        detections = []
        for cone in results.xywh[0]:
            cx = int(cone[0])
            cy = int(cone[1])
            w = int(cone[2])
            h = int(cone[3])
            detected_cone = DetectedCone(cx, cy, w, h)
            detections.append(detected_cone)
        lds = Detections(detections, imgs[0][:, :, ::-1], max_dist=20)

        detections = []
        for cone in results.xywh[1]:
            cx = int(cone[0])
            cy = int(cone[1])
            w = int(cone[2])
            h = int(cone[3])
            detected_cone = DetectedCone(cx, cy, w, h)
            detections.append(detected_cone)
        rds = Detections(detections, imgs[1][:, :, ::-1], max_dist=20)

        if len(lds) == 0 or len(rds) == 0:
            raise Exception("No Detections")
            
        lds.init_detections()
        rds.init_detections()
        lds = lds.filter_distance()
        rds = rds.filter_distance()
        lds.colour_estimation()
        rds.colour_estimation()

        #vid_image = dbgt.hstack_images(lds.image, rds.image)
        #dbgt.show_video("stereo", vid_image)

        stereo_matcher = Matching.StereoMatcher(lds, rds)
        stereo_matcher.find_stereo_matches()
        stereo_matcher.calculate_depth()
        #Should find out if i can do a pass by reference kind of thing in Python
        lds_matched, rds_matched = stereo_matcher.get_matched() #type(train_matched) = Detections
        lds_anno = lds_matched.get_annotated_image()
        rds_anno = rds_matched.get_annotated_image()
        matches = dbgt.hstack_images(lds_anno, rds_anno)
        #cv.waitKey(1)
        #cv.imshow("matches", matches)
        self.new_cones = lds_matched
        self.locate_cones()
        


    def derive_motion(self, train, query):

        if len(train) < 2:
            print("Uh oh spaghetti-o, we need at least two matched cones")
            #Do some frame skipping protocol
            self.frame_skips += 1
            return
        else:
            self.frame_skips = 1

        
        #Esures closes cones (with lowest error are used)
        dists = np.array([cone.loc_cs.z for cone in train])
        min_idxs = dists.argsort()
        train = train[min_idxs]
        query = query[min_idxs]

        t1 = train[0].loc_cs
        t2 = train[1].loc_cs

        q1 = query[0].loc_cs
        q2 = query[1].loc_cs
        tz_1 = t1.z
        tx_1 = t1.x
        tx_2 = t2.x
        tz_2 = t2.z

        qx_1 = q1.x
        qz_1 = q1.z
        qx_2 = q2.x
        qz_2 = q2.z

        x_t = (-qx_1*(tx_1*tx_2 - tx_2**2 + tz_1*tz_2 - tz_2**2) + qx_2*(tx_1**2 - tx_1*tx_2 + tz_1**2 - tz_1*tz_2) + qz_1*(tx_1*tz_2 - tx_2*tz_1) - qz_2*(tx_1*tz_2 - tx_2*tz_1))/(tx_1**2 - 2*tx_1*tx_2 + tx_2**2 + tz_1**2 - 2*tz_1*tz_2 + tz_2**2)
        z_t = (-qx_1*(tx_1*tz_2 - tx_2*tz_1) + qx_2*(tx_1*tz_2 - tx_2*tz_1) - qz_1*(tx_1*tx_2 - tx_2**2 + tz_1*tz_2 - tz_2**2) + qz_2*(tx_1**2 - tx_1*tx_2 + tz_1**2 - tz_1*tz_2))/(tx_1**2 - 2*tx_1*tx_2 + tx_2**2 + tz_1**2 - 2*tz_1*tz_2 + tz_2**2)
        sin_theta = (-qx_1*(tz_1 - tz_2) + qx_2*(tz_1 - tz_2) + qz_1*(tx_1 - tx_2) - qz_2*(tx_1 - tx_2))/(tx_1**2 - 2*tx_1*tx_2 + tx_2**2 + tz_1**2 - 2*tz_1*tz_2 + tz_2**2)
        cos_theta = (qx_1*(tx_1 - tx_2) - qx_2*(tx_1 - tx_2) + qz_1*(tz_1 - tz_2) - qz_2*(tz_1 - tz_2))/(tx_1**2 - 2*tx_1*tx_2 + tx_2**2 + tz_1**2 - 2*tz_1*tz_2 + tz_2**2)
        frame_time = 1 / 30
        v = 5
        dist = frame_time * v * self.frame_skips * 1.5
        theta = np.arcsin(sin_theta)
        #Teleportation protection
        if np.rad2deg(theta) > 5:
            #maybe we should just return?
            print("to much rotation")
            self.frame_skips += 1
            return

        self.car_rotation += theta
        rotation_matrix = gen_rotation_matrix(0, self.car_rotation, 0)
        pos_change = Vec3(x_t, 0, z_t)
        self.car_loc_ws = self.car_loc_ws + rotation_matrix * -pos_change
        """ if (self.car_loc_ws - self.positions[-1]).mag() > dist:
            print("translation: ", np.sqrt(x_t ** 2 + z_t ** 2))
            self.frame_skips += 1
            return """
        self.positions.append(self.car_loc_ws)
            
    def get_unmapped_cones(self, cones):
        cones = Detections([cone for cone in cones if cone.unmapped == True], cones.image)
        return cones

    def set_cone_ws(self, cones):
        ry_mat = np.matrix([[np.cos(self.ry),  0, np.sin(self.ry)],
                    [0,             1,             0],
                    [-np.sin(self.ry), 0, np.cos(self.ry)]])
        for i, cone in enumerate(cones):
            loc_cs = np.matrix(cone.loc_cs.vec()).transpose()
            loc_cs = np.array((ry_mat * loc_cs).transpose()[0])
            loc_cs = Vec3(loc_cs[0][0], loc_cs[0][1], -loc_cs[0][2])
            loc_ws = loc_cs + self.car_pos_ws
            cones[i].loc_ws = loc_ws
            cones[i].unmapped = False
        return cones

    def get_track(self):
        return self.track

    def begin(self):
        #Begin camera and capture frames
        
        cap_train = cv.VideoCapture(self.src_train)
        cap_query = cv.VideoCapture(self.src_query)

        slam_initialised = False
        i = 1
        print("cap_train:", cap_train.isOpened())
        print("cap_query:", cap_query.isOpened())
        cv.waitKey(10)
        while cap_train.isOpened() and cap_query.isOpened():
            print("Frame: ", i)
            
            rett, train_frame = cap_train.read()
            retq, query_frame = cap_query.read()

            if rett and retq:
                vid_im = dbgt.hstack_images(train_frame, query_frame)
                #print("train frame: ", cap_train.get(1))
                #print("query frame: ", cap_query.get(1))
                #dbgt.show_image("vid im", vid_im)
                

            train_frame = np.array(train_frame)
            query_frame = np.array(query_frame)

            imgs = [np.array(train_frame), np.array(query_frame)]
            imgs = [img[:, :, ::-1] for img in imgs]

            if not slam_initialised:
                print("did initialise")
                slam_initialised = True
                self.stereo_process_new_frames(imgs)
                self.prev_cones = self.new_cones
                i += 1
                continue
                #mapped_cones = self.set_cone_ws(self.new_cones)
                #self.track.update_track(mapped_cones)
            
            if i > 1000:
                break
            #Below will be in a loop
            #####################
            if self.frame_skips == 1:
                self.prev_cones = self.new_cones
                
            i += 1    
                
            self.stereo_process_new_frames(imgs)
            #Now we have two sets of located cones
            #Previous = Train, New = Query
            #FrameMatcher(Train, Query) matches subsequent frames to derive motion
            
            prev_anno = self.prev_cones.get_annotated_image()
            new_anno = self.new_cones.get_annotated_image()
            #dbgt.vstack_images(prev_anno, new_anno)

            frame_matcher = Matching.FrameMatcher(self.prev_cones, self.new_cones)
            frame_matcher.find_subsequent_matches(n=self.frame_skips)
            #print("frame matches: \n", frame_matcher.matches)
            prev_matched, new_matched = frame_matcher.get_matched()
            matches = frame_matcher.get_matches_image()
            cv.waitKey(1)
            cv.imshow("Frame matches", matches)
            if len(prev_matched) == 0:
                self.frame_skips += 1
                continue
            prev_matched.locate_cones()
            new_matched.locate_cones()
            self.derive_motion(prev_matched, new_matched)
            #print("now car travel is: ", self.car_pos_ws)
            """ unmapped_cones = self.get_unmapped_cones(new_matched)
            self.mapped_cones = self.set_cone_ws(unmapped_cones)
            self.track.update_track(self.mapped_cones) """
            #self.show_live_output()
            ####################
            #Now we have new cones and old cones    


    def show_local_map(self):
        img = self.new_cones.get_local_map()
        dbgt.show_image("new_cones local map", img)


    def get_cones(self):
        return self.new_cones, self.prev_cones

    def locate_cones(self):
        im_width = self.new_cones[0].im_width
        for i, cone in enumerate(self.new_cones):
            
            sensor_loc_mm = abs(im_width / 2 - cone.cx) * (consts.sensorWidth_mm / im_width)
            angle = cone.angle
            x_cs = cone.depth * np.tan(angle) - consts.camera_spacing_m / 2
            #CONVENTION: Before the car moves, the car space coordinate system is aligned with the OpenCv coordinate system, the origins are at the same point initially.
            point = Vec3(x_cs, 0, cone.depth)
            self.new_cones[i].loc_cs = point

    def get_track_map(self):
        im_size = 720
        area = 80
        mid = int(im_size / 2)
        blank_image = np.zeros((im_size, im_size, 3), np.uint8)
        blank_image[:] = (128, 128, 128)
        blank_image = dbgt.get_track_overhead("example")
        #cv.circle(blank_image, (int(im_size / 2), int(im_size - 20)), 5, (0, 255, 0), thickness=-1)
        for pos in self.positions:
            x_cs = pos.x
            z_cs = pos.z
            im_y = int(mid - z_cs / (area / 2) * mid)
            im_x = int(mid + x_cs / (area / 2) * mid)
            cv.circle(blank_image, (im_x, im_y), 4, (0, 0, 0), thickness=-1)
        return blank_image

    def show_track_map(self):
        img = self.get_track_map()
        cv.waitKey(1)
        dbgt.show_image("Track Map", img)

    def show_live_output(self):
        map = self.get_track_map()
        query_im = self.new_cones.get_annotated_image()
        train_im = self.prev_cones.get_annotated_image()
        stacked = dbgt.vstack_images(train_im, query_im)
        map = cv.resize(map, (0, 0), None, .25, .25)
        stacked[0:180, 0:180] = map
        cv.waitKey(1)
        cv.imshow("video", stacked)

render_path = os.path.abspath("~/../../Blender/Resources/Renders/Videos/")
left_vid = os.path.join(render_path, "example-left.mp4")
right_vid = os.path.join(render_path, "example-right.mp4")

slam = Mapper(left_vid, right_vid)
slam.begin()
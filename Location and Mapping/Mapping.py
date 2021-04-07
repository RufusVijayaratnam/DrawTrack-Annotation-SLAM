import numpy as np
from Detections import DetectedCone, Detections
import Matching
from Colour import Colour
import Constants as consts
import cv2 as cv
import torch
import importlib
import Track
import DebugTools as dbgt
importlib.reload(Matching)
importlib.reload(Track)
importlib.reload(dbgt)

class Point():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def vec(self):
        return np.array([self.x, self.y, self.z])

    def __add__(self, obj):
        return Point(self.x + obj.x, self.y + obj.y, self.z + obj.z)

    def __str__(self):
        prnt = "[%f, %f, %f]" % (self.x, self.y, self.z)
        return prnt
        

class Mapper():
    def __init__(self, src_train, src_query):
        self.prev_cones = None
        self.new_cones = None
        self.mapped_cones = None
        self.im_size = 300
        self.blank_image = np.zeros((self.im_size, self.im_size, 3), np.uint8)
        self.blank_image[:] = (255, 255, 255)
        cv.circle(self.blank_image, (int(self.im_size / 2), int(self.im_size - 20)), 5, (0, 255, 0))
        
        self.car_pos_ws = Point(0, 0, 0)
        self.ry = 0
        self.track = Track.Track()
        
        yolov5_path = "/mnt/c/Users/Rufus Vijayaratnam/yolov5/runs/train/exp8/weights/best.pt"
        weights_path = yolov5_path + ""
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path_or_model=weights_path)

        self.src_train = src_train
        self.src_query = src_query

        self.motion_updated = True

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
        lds = Detections(detections, imgs[0][:, :, ::-1], max_dist=10)

        detections = []
        for cone in results.xywh[1]:
            cx = int(cone[0])
            cy = int(cone[1])
            w = int(cone[2])
            h = int(cone[3])
            detected_cone = DetectedCone(cx, cy, w, h)
            detections.append(detected_cone)
        rds = Detections(detections, imgs[1][:, :, ::-1], max_dist=10)

        if len(lds) == 0 or len(rds) == 0:
            raise Exception("No Detections")
            
        lds.init_detections()
        rds.init_detections()
        lds = lds.filter_distance()
        rds = rds.filter_distance()
        lds.colour_estimation()
        rds.colour_estimation()

        stereo_matcher = Matching.StereoMatcher(lds, rds)
        stereo_matcher.find_stereo_matches()
        stereo_matcher.calculate_depth()
        #Should find out if i can do a pass by reference kind of thing in Python
        lds_matched, rds_matched = stereo_matcher.get_matched() #type(train_matched) = Detections
        self.new_cones = lds_matched
        self.locate_cones()
        
        
    def show_single_matches(self, train, query):
        for i in range(1, len(train) + 1):
            train[i - 1:i].show_annotated_image()
            query[i - 1:i].show_annotated_image()

    def rms(self, arry):
        arry = np.array(arry)
        arry_sum = np.sum(arry ** 2)
        sum_avg = arry_sum / len(arry)
        return np.sqrt(sum_avg)

    def derive_motion(self, train, query):
        if len(train) == 0:
            self.motion_updated = False
            return
        self.motion_updated = True
        dzs = np.ndarray(len(train))
        for i, (t, q) in enumerate(zip(train, query)):
            dz = t.depth - q.depth
            dzs[i] = dz
        
        avg_dz = np.average(dzs)
        print("average dz = ", avg_dz)
        rotations = np.ndarray(len(train))
        for i, (t, q) in enumerate(zip(train, query)):
            gamma_train = t.angle
            tz = t.loc_cs.z
            tx = t.loc_cs.x
            qz = tz - avg_dz
            #gamma_query is the expected difference in gamma for no car rotation
            gamma_query = np.arctan(tx / qz)
            #gamma_e is expected change in angle due to forward motion with no car rotation
            gamma_e = gamma_query - gamma_train
            gamma_a = q.angle - t.angle
            rotation = gamma_a - gamma_e
            rotations[i] = rotation
        rms_ry = self.rms(rotations)
        print("rms ry: ", rms_ry)
        
        self.ry += rms_ry
        car_travel_x = avg_dz * np.sin(self.ry)
        car_travel_z = avg_dz * np.cos(self.ry)
        car_travel_ws = Point(car_travel_x, 0, car_travel_z)
        self.car_pos_ws += car_travel_ws
            
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
            loc_cs = Point(loc_cs[0][0], loc_cs[0][1], -loc_cs[0][2])
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
        i = 0
        while cap_train.isOpened() and cap_query.isOpened():
            print("Frame: ", i)
            
            rett, train_frame = cap_train.read()
            retq, query_frame = cap_query.read()
            train_frame = np.array(train_frame)
            query_frame = np.array(query_frame)

            imgs = [np.array(train_frame), np.array(query_frame)]
            imgs = [img[:, :, ::-1] for img in imgs]

            if not slam_initialised:
                slam_initialised = True
                self.stereo_process_new_frames(imgs)
                self.prev_cones = self.new_cones
                #mapped_cones = self.set_cone_ws(self.new_cones)
                #self.track.update_track(mapped_cones)

            if i > 10:
                break
            #Below will be in a loop
            #####################
            if self.motion_updated:
                self.prev_cones = self.new_cones
                
            self.stereo_process_new_frames(imgs)
            #Now we have two sets of located cones
            #Previous = Train, New = Query
            #FrameMatcher(Train, Query) matches subsequent frames to derive motion
            print("len prev cones: ", len(self.prev_cones))
            print("len new cones: ", len(self.new_cones))
            prev_anno = self.prev_cones.get_annotated_image()
            new_anno = self.new_cones.get_annotated_image()
            #dbgt.vstack_images(prev_anno, new_anno)

            frame_matcher = Matching.FrameMatcher(self.prev_cones, self.new_cones)
            frame_matcher.find_subsequent_matches()
            #print("frame matches: \n", frame_matcher.matches)
            prev_matched, new_matched = frame_matcher.get_matched()
            frame_matcher.show_matches(name="Frame: %d" % i)
            self.derive_motion(prev_matched, new_matched)
            #print("now car travel is: ", self.car_pos_ws)
            unmapped_cones = self.get_unmapped_cones(new_matched)
            self.mapped_cones = self.set_cone_ws(unmapped_cones)
            self.track.update_track(self.mapped_cones)

            ####################
            #Now we have new cones and old cones    
            i += 1


    def show_local_map(self):
        for cone in self.new_cones:
            x_cs = cone.loc_cs.x
            im_y = int(self.im_size - cone.depth / 20 * (self.im_size - 20))
            im_x = int(self.im_size / 2 + x_cs / 10 * self.im_size)
            cv.circle(self.blank_image, (im_x, im_y), 4, cone.colour.colour, thickness=-1)

        cv.imshow("Local Map", self.blank_image)
        cv.waitKey(0)
        cv.destroyWindow("Local Map")



    def get_localised_cones(self):
        return self.new_cones

    def locate_cones(self):
        im_width = self.new_cones[0].im_width
        for i, cone in enumerate(self.new_cones):
            
            sensor_loc_mm = abs(im_width / 2 - cone.cx) * (consts.sensorWidth_mm / im_width)
            angle = cone.angle
            x_cs = cone.depth * np.tan(angle) - consts.camera_spacing_m / 2
            #CONVENTION: Before the car moves, the car space coordinate system is aligned with the OpenCv coordinate system, the origins are at the same point initially.
            point = Point(x_cs, 0, cone.depth)
            self.new_cones[i].loc_cs = point

render_path = "/mnt/c/Users/Rufus Vijayaratnam/Driverless/Blender/Resources/Renders/"
left_vid = render_path + "track8-left.avi"
right_vid = render_path + "track8-right.avi"

slam = Mapper(left_vid, right_vid)
slam.begin()
track = slam.get_track()
track.draw_track()
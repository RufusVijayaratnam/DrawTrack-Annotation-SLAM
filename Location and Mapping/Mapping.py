import numpy as np
from Detections import DetectedCone, Detections
import StereoMatching as matching
from Colour import Colour
import Constants as consts
import cv2 as cv
import torch
import importlib
importlib.reload(matching)

class Point():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z =z

class Mapper():
    def __init__(self):
        self.prev_cones = None
        self.new_cones = None
        self.im_size = 300
        self.blank_image = np.zeros((self.im_size, self.im_size, 3), np.uint8)
        self.blank_image[:] = (255, 255, 255)
        yolov5_path = "/mnt/c/Users/Rufus Vijayaratnam/yolov5/runs/train/exp8/weights/best.pt"
        weights_path = yolov5_path + ""
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path_or_model=weights_path)
        
        cv.circle(self.blank_image, (int(self.im_size / 2), int(self.im_size - 20)), 5, (0, 255, 0))

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

        stereo_matcher = matching.StereoMatcher(lds, rds)
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
        avg_rotation = self.rms(rotations)
        print("rms rotation: ", avg_rotation)
            


    def begin(self):
        #Begin camera and capture frames
        
        im_left = "/mnt/c/Users/Rufus Vijayaratnam/Driverless/Blender/Resources/Renders/test/images/track8-Left_Cam-Render-0.png"
        im_right = "/mnt/c/Users/Rufus Vijayaratnam/Driverless/Blender/Resources/Renders/test/images/track8-Right_Cam-Render-0.png"
        im_left1 = "/mnt/c/Users/Rufus Vijayaratnam/Driverless/Blender/Resources/Renders/test/images/track8-Left_Cam-Render-1.png"
        im_right1 = "/mnt/c/Users/Rufus Vijayaratnam/Driverless/Blender/Resources/Renders/test/images/track8-Right_Cam-Render-1.png"

        image_left = np.array(cv.imread(im_left))
        image_right = np.array(cv.imread(im_right))
        image_left1 = np.array(cv.imread(im_left1))
        image_right1 = np.array(cv.imread(im_right1))

        imgs = [image_left, image_right]
        imgs = [img[:, :, ::-1] for img in imgs]
        imgs1 = [image_left1, image_right1]
        imgs1 = [img[:, :, ::-1] for img in imgs1]
        print("about to process frames for first image")
        self.stereo_process_new_frames(imgs)

        #Below will be in a loop
        #####################
        self.prev_cones = self.new_cones
        print("about to process frames for second image")
        self.stereo_process_new_frames(imgs1)
        #Now we have two sets of located cones
        #Previous = Train, New = Query
        frame_matcher = matching.FrameMatcher(self.prev_cones, self.new_cones)
        frame_matcher.find_subsequent_matches()
        print("frame matches: \n", frame_matcher.matches)
        prev_matched, new_matched = frame_matcher.get_matched()
        self.derive_motion(prev_matched, new_matched)


        ####################
        #Now we have new cones and old cones    


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
            if cone.cx < im_width / 2:
                multiplier = -1
            else:
                multiplier = 1
            sensor_loc_mm = abs(im_width / 2 - cone.cx) * (consts.sensorWidth_mm / im_width)
            angle = np.arctan(sensor_loc_mm / consts.focalLength_mm) * multiplier
            x_cs = cone.depth * np.tan(angle)
            #CONVENTION: Before the car moves, the car space coordinate system is aligned with the OpenCv coordinate system, the origins are at the same point initially.
            point = Point(x_cs, 0, cone.depth)
            self.new_cones[i].angle = angle
            self.new_cones[i].loc_cs = point

slam = Mapper()
slam.begin()
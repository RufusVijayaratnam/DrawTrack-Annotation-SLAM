import cv2 as cv
import torch 
import numpy as np
import os
import matplotlib.pyplot as plt


baseline_mm = 100
focalLength_mm = 5.5
sensorWidth_mm = 8.5
sensorHeight_mm = 4.78
camera_spacing_m = 0.1
ref_height_pix = 63.976


yolov5_path = "/mnt/c/Users/Rufus Vijayaratnam/yolov5/runs/train/exp8/weights/best.pt"
weights_path = yolov5_path + ""
model = torch.hub.load("ultralytics/yolov5", "custom", path_or_model=weights_path)




def monocular_distance_estimate(h):
        #Should return an upper and lower bound for disparity (in pixels)
        #This is a very rough estimate and should be taken with a bucket of salt
        #Unlikely to work well for obscured cones
        margin = 1.1 #Percent for bounds, this can be changed to improve
        focalLength_pixels = (focalLength_mm / sensorWidth_mm) * 1920
        #F = 5 * consts.ref_height_pix / 0.262 #0.262 = cone height m
        depth = focalLength_pixels * 0.262 / h
        depth_max_mm = depth * margin * 1000 
        depth_min_mm = depth / margin * 1000 
        #return distance (z cv space), lower_disparity_pix, upper_disparity_pix
        #Disparity decreases with depth, so depth_max_mm is used to calculate disp_min
        disp_min = baseline_mm * focalLength_pixels / depth_max_mm
        disp_max = baseline_mm * focalLength_pixels / depth_min_mm
        return depth, np.floor(disp_min), np.ceil(disp_max)

true_dists0 = []
est_dists0 = []
img_dir = "/mnt/c/Users/Rufus Vijayaratnam/Driverless/Blender/Resources/Renders/Experiments/Monocular Evaluation/CRotation10/"
for img_f in os.listdir(img_dir):
    img = np.array(cv.imread(img_dir + img_f))
    img = img[:, :, ::-1]
    f = img_f.split("MonocEval-")[1]
    true_dist = float(f.split("m.png")[0])
    results = model(img, size=640)
    
    if len(results.xywh[0]) > 0:
        for cone in results.xywh[0]:
            cx = int(cone[0])
            cy = int(cone[1])
            w = int(cone[2])
            h = int(cone[3])
            depth, _, _ = monocular_distance_estimate(h)
            true_dists0.append(true_dist)
            est_dists0.append(depth)

errors0 = [abs(t - e) / t * 100 for t, e in zip(true_dists0, est_dists0)]
true_dists0, errors0 = (list(t) for t in zip(*sorted(zip(true_dists0, errors0))))

true_dists15 = []
est_dists15 = []
img_dir = "/mnt/c/Users/Rufus Vijayaratnam/Driverless/Blender/Resources/Renders/Experiments/Monocular Evaluation/CRotation20/"
for img_f in os.listdir(img_dir):
    img = np.array(cv.imread(img_dir + img_f))
    img = img[:, :, ::-1]
    f = img_f.split("MonocEval-")[1]
    true_dist = float(f.split("m.png")[0])
    results = model(img, size=640)
    
    if len(results.xywh[0]) > 0:
        for cone in results.xywh[0]:
            cx = int(cone[0])
            cy = int(cone[1])
            w = int(cone[2])
            h = int(cone[3])
            depth, _, _ = monocular_distance_estimate(h)
            true_dists15.append(true_dist)
            est_dists15.append(depth)

errors15 = [abs(t - e) / t * 100 for t, e in zip(true_dists15, est_dists15)]
true_dists15, errors15 = (list(t) for t in zip(*sorted(zip(true_dists15, errors15))))

true_dists45 = []
est_dists45 = []
img_dir = "/mnt/c/Users/Rufus Vijayaratnam/Driverless/Blender/Resources/Renders/Experiments/Monocular Evaluation/CRotation30/"
for img_f in os.listdir(img_dir):
    img = np.array(cv.imread(img_dir + img_f))
    img = img[:, :, ::-1]
    f = img_f.split("MonocEval-")[1]
    true_dist = float(f.split("m.png")[0])
    results = model(img, size=640)
    
    if len(results.xywh[0]) > 0:
        for cone in results.xywh[0]:
            cx = int(cone[0])
            cy = int(cone[1])
            w = int(cone[2])
            h = int(cone[3])
            depth, _, _ = monocular_distance_estimate(h)
            true_dists45.append(true_dist)
            est_dists45.append(depth)

errors45 = [abs(t - e) / t * 100 for t, e in zip(true_dists45, est_dists45)]
tcopy = true_dists45
true_dists45, errors45 = (list(t) for t in zip(*sorted(zip(true_dists45, errors45))))
tcopy, est_dists45 = (list(t) for t in zip(*sorted(zip(tcopy, est_dists45))))

#%%
m0, c0 = np.polyfit(true_dists0, errors0, 1)
m15, c15 = np.polyfit(true_dists15, errors15, 1)
m45, c45 = np.polyfit(true_dists45, errors45, 1)


#%%
fig, ax1 = plt.subplots()
fig.set_size_inches(8, 6)
ax1.set_ylabel("Error (%)")
ax1.set_xlabel("Distance (m)")
ax1.plot(true_dists0, errors0, "bx")
plt.plot(true_dists0, m0* np.array(true_dists0) + c0, color="blue", label="0 Rotation")
ax1.plot(true_dists0, errors15, 'ro')
plt.plot(true_dists15, m15* np.array(true_dists15) + c15, color="red", label="15 Rotation")
ax1.plot(true_dists0, errors45, 'gx')
plt.plot(true_dists45, m45* np.array(true_dists45) + c45, color="green", label="45 Rotation")
ax1.legend()
plt.close(fig)
ax1.set_title("Error vs Distance")
fig.savefig("Error.png", dpi=200)
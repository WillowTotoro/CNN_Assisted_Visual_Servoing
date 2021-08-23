import sys
sys.path.append('/home/jyx/darknet/')
import math
import cv2
import numpy as np
from visual_servo import cal_robot_vel
from visual_servo_null_space import cal_robot_vel_ns
from visual_servo_combine import cal_robot_vel_f
from visual_servo_test import cal_robot_vel_t
import pyrealsense2 as rs
import darknet
import json
import subprocess
from matplotlib.pyplot import plot
import pandas as pd
from collections import defaultdict
from keyboard_input import getch
from datetime import datetime

global image_height, image_width
image_height = 1280
image_width = 720

global darknet_config
darknet_config = 608

# intrin_para = [641.0947265625,354.866577148438,644.009826660156,644.009826660156]
intrin_para = [637.992919921875,361.408325195312,927.108520507812,927.300964355469]
# intrin_para = [956.989318847656,542.112487792969,1390.66284179688,1390.95141601562]#1920x1080
# intrin_para = [318.661926269531,240.938873291016,618.072387695312,618.200622558594]
# intrin_para = [320.656829833984,236.919952392578, 386.405883789062, 386.405883789062]

cx = intrin_para[0]
cy = intrin_para[1]
fx = intrin_para[2]
fy = intrin_para[3]

# Intrinsic of "Color" / 1920x1080 / {YUYV/RGB8/BGR8/RGBA8/BGRA8/Y16/RAW16}
#   Width:      	1920
#   Height:     	1080
#   PPX:        	956.989318847656
#   PPY:        	542.112487792969
#   Fx:         	1390.66284179688
#   Fy:         	1390.95141601562
#   Distortion: 	Inverse Brown Conrady
#   Coeffs:     	0  	0  	0  	0  	0  
#   FOV (deg):  	69.24 x 42.43

# Intrinsic of "Depth" / 1280x720 / {Z16}
#   Width:      	1280
#   Height:     	720
#   PPX:        	641.0947265625
#   PPY:        	354.866577148438
#   Fx:         	644.009826660156
#   Fy:         	644.009826660156
#1280*720 Color
# PPX:        	637.992919921875
#   PPY:        	361.408325195312
#   Fx:         	927.108520507812
#   Fy:         	927.300964355469

#640*480 color
# PPX:        	318.661926269531
#   PPY:        	240.938873291016
#   Fx:         	618.072387695312
#   Fy:         	618.200622558594

#640*480 depth
# PPX:        	320.656829833984
# PPY:        	236.919952392578
# Fx:         	386.405883789062
# Fy:         	386.405883789062



# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, image_height,image_width, rs.format.z16, 15)
config.enable_stream(rs.stream.color, image_height, image_width, rs.format.bgr8, 15)
colorizer = rs.colorizer()
align = rs.align(rs.stream.color)
spat_filter = rs.spatial_filter()          # Spatial    - edge-preserving spatial smoothing
temp_filter = rs.temporal_filter()   # Temporal   - reduces temporal noise
hole_fill = rs.hole_filling_filter()
# Start streaming
profile = pipeline.start(config)

# converting BB coordinates in yolo txt format in pixels


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


# drawing bounding boxes in image from detections

def cvDrawBoxes(detections, img):
    global object_flag
    object_flag = 0
    if len(detections) == 0:
        print('no detection')
        # x =int(cx/image_height*416)
        # y =int(cy/image_width*416)
        x =int(cx/image_height*darknet_config)
        y =int(cy/image_width*darknet_config)
        return(img, x, y, x, y, 0)
    else:
        # bottle_flag = 0
        for label, confidence, bbox in detections:
            if label == 'crack':
                object_flag = 1
                x, y, w, h = (bbox[0],bbox[1], bbox[2],bbox[3])
                name_tag = label
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
                pt1 = (xmin, ymin)
                pt2 = (xmax, ymax)
                # print((x,y))
                cv2.rectangle(img, pt1, pt2, (255, 255, 255), 1)
                # cv2.rectangle(depth_img, pt1, pt2, (255, 255, 255), 1)
                cv2.putText(img,str(confidence), (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], 2)
                return(img, x, y, xmin,ymin,confidence)

        # x = int(cx/image_height*416)
        # y = int(cy/image_width*416)
        x = int(cx/image_height*darknet_config)
        y = int(cy/image_width*darknet_config)
        return(img, x, y, x, y,0)

# IP_configPath = "/home/jyx/darknet/cfg/yolov3.cfg"
# IP_weightPath = "/home/jyx/darknet/yolov3.weights"
# IP_metaPath = "/home/jyx/darknet/cfg/coco.data"

IP_configPath = "/home/jyx/darknet/tile_crack/yolov3_tiles.cfg"
IP_weightPath = "/home/jyx/darknet/tile_crack/yolov3_tiles_best.weights"
IP_metaPath = "/home/jyx/darknet/tile_crack/detector.data"

darknet.set_gpu(0)

network, class_names, class_colors = darknet.load_network(IP_configPath,  IP_metaPath, IP_weightPath, batch_size=1)

result_dict = defaultdict(list)

save_result_flag = True
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
line_count = 0
try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        
        depth_frame = frames.get_depth_frame()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # filtered = spat_filter.process(depth_frame)
        filtered = temp_filter.process(depth_frame)
        filtered = hole_fill.process(filtered)

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        bgr_color_image = np.asanyarray(color_frame.get_data())
        # rgb_color_image = cv2.cvtColor(bgr_color_image, cv2.COLOR_BGR2RGB)

        colorized_depth_image = np.asanyarray(
            colorizer.colorize(filtered).get_data())
        
        width = darknet.network_width(network)
        height = darknet.network_height(network)
        darknet_image = darknet.make_image(width, height, 3)

        image_rgb = cv2.cvtColor(bgr_color_image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.2)
        # print(detections)
        darknet.free_image(darknet_image)
        
        # image = darknet.draw_boxes(detections, image_resized, class_colors)
        # labeled_image, bbcx, bbcy= cvDrawBoxes(detections, image_resized)
        # labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
        # print(detections)
        
        labeled_image, bbcx, bbcy, x1, y1, confidence = cvDrawBoxes(detections, image_resized)
        labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
        
        print('Bounding box center is at:',(bbcx/608*1280,bbcy/608*720))

        # pt1 =(int(bbcx/416*image_height-20),int(bbcy/416*image_width-30))
        # pt2 =(int(bbcx/416*image_height+20),int(bbcy/416*image_width+30))
        pt1 =(int(bbcx/darknet_config*image_height-20),int(bbcy/darknet_config*image_width-30))
        pt2 =(int(bbcx/darknet_config*image_height+20),int(bbcy/darknet_config*image_width+30))
        # print((pt1,pt2))

        colorized_depth_image = cv2.rectangle(colorized_depth_image, pt1,pt2, (0,0,0), 2)
        colorized_depth_image = cv2.resize(colorized_depth_image,(darknet_config,darknet_config)) 
        
        # Stack both mages horizontally
        # images = np.hstack((labeled_image, colorized_depth_image))
        
        images = np.hstack((labeled_image, colorized_depth_image))

        # Crop depth data:
        depth = np.asanyarray(depth_frame.get_data())
        # print('depth shape', depth.shape)
        depth = depth[pt1[1]:pt2[1],pt1[0]:pt2[0]].astype(float)
        # print(depth)

        # Get data scale from the device and convert to meters
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth = depth * depth_scale
        #dist = np.mean(np.asarray(depth))
        dist, _, _, _ = cv2.mean(depth)

        print("Detected {0} meters away...".format(dist))
        
        # print('after correction is {} meter away...'.format(dist))

        # robot_vel = cal_robot_vel(intrin_para, bbcx, bbcy, dist, gain=0.2)
        
        robot_vel = cal_robot_vel_f(intrin_para, bbcx, bbcy, x1,y1, dist, gain=-0.05,gain_p=-1e-9)

        # robot_vel = cal_robot_vel_t(intrin_para, bbcx, bbcy, x1,y1, dist, gain=0.2,gain_p=0.001)

        # robot_vel = cal_robot_vel_ns(intrin_para, bbcx, bbcy, x1,y1, dist, gain=0.1,gain_null=1e6)

        vx,vy,vz,wx,wy,wz = robot_vel

        print('robot linear velocity vx:{},vy:{},vz:{}, angular velocity wx:{},wy:{},wz:{}'.format(vx,vy,vz,wx,wy,wz))
        print('\n')
        
        FILE = 'upload_scout.json'

        with open('upload_scout.json','w') as f:
            # f.write(' '.join(str(v[0]) for v in robot_vel))
            f.write(' '.join(str(v) for v in robot_vel))
            f.close()
        
        # subprocess.run(["scp", FILE, "scout@10.7.5.88:/home/scout/jyx_code"])
        # char = getch()

        if save_result_flag and object_flag:
            result_dict['confidence'].append(confidence)
            result_dict['distance'].append(dist)
            result_dict['bbcx'].append(bbcx/608*1280)
            result_dict['bbcy'].append(bbcy/608*720)
            result_dict['bbw'].append(2*(bbcx-x1))
            result_dict['bbh'].append(2*(bbcy-y1))
            result_dict['vx'].append(vx)
            result_dict['vy'].append(vy)
            result_dict['vz'].append(vz)
            result_dict['wx'].append(wx)
            result_dict['wy'].append(wy)
            result_dict['wz'].append(wz)
            df = pd.DataFrame(dict(result_dict))
            df.to_csv('Results/'+str('full_-+_')+str(now)+'.csv',index=False)
        
        # Show images
        # plot(range(len(result_dict['bbcx'])), result_dict['bbcx'], color='green', marker='o', linestyle='dashed',linewidth=2, markersize=12)
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', images)
        cv2.waitKey(500)

except KeyboardInterrupt:
    print('ctrl-c pressed, experiment ends, saving results...')
    
    # Stop streaming
    pipeline.stop()

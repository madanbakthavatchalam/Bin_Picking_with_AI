#!/usr/bin/env python3

from concurrent.futures import process
import rospy
# import pcl
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
# import ros_numpy
import open3d
import open3d as o3d
from ctypes import *
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.utils import normalize
from tensorflow import keras
from tensorflow.keras import callbacks
import cv2
import os
import glob
from matplotlib import colors, cm , pyplot as plt
import xmlrpc.client
import random
import time




model = keras.models.load_model(r"/home/madan/Master_thesis/model_border_filled_cw/checkpoint/")
model.summary()


s = xmlrpc.client.ServerProxy('http://localhost:8030')  #('http://localhost:8030')  'http://127.0.0.1:8030'

print("Safety Status",s.safetyStatus())
print(s.system.listMethods())

# global received_ros_cloud
received_ros_cloud = None

# FIELDS_XYZ = [
#     PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
#     PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
#     PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
# ]
# FIELDS_XYZRGB = FIELDS_XYZ + \
#     [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8
convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)


def callback(ros_cloud):
    global received_ros_cloud
    received_ros_cloud=ros_cloud
    # rospy.loginfo("-- Received ROS PointCloud2 message.")

    # Get cloud data from ros_cloud
    # field_names=[field.name for field in ros_cloud.fields]
    # cloud_data = list(pc2.read_points(ros_cloud, skip_nans=False, field_names = field_names))

    # # Check empty
    # open3d_cloud = open3d.geometry.PointCloud()
    # if len(cloud_data)==0:
    #     print("Converting an empty cloud")
    #     return None

    # # Set open3d_cloud
    # if "rgb" in field_names:
    #     IDX_RGB_IN_FIELD=3 # x, y, z, rgb
        
    #     # Get xyz
    #     xyz = [(x,y,z) for x,y,z,rgb in cloud_data ] # (why cannot put this line below rgb?)

    #     # Get rgb
    #     # Check whether int or float
    #     if type(cloud_data[0][IDX_RGB_IN_FIELD])==float: # if float (from pcl::toROSMsg)
    #         rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
    #     else:
    #         rgb = [convert_rgbUint32_to_tuple(rgb) for x,y,z,rgb in cloud_data ]

    #     # combine
    #     open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))
    #     open3d_cloud.colors = open3d.utility.Vector3dVector(np.array(rgb)/255.0)
    # else:
    #     xyz = [(x,y,z) for x,y,z in cloud_data ] # get xyz
    #     open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))

    # return
    # return open3d_cloud
    # return received_ros_cloud

def Process(event): # Type rospy.TimerEvent
    rospy.loginfo("-- SAFTEY STATUS : %s --",s.safetyStatus())
    s.lightControl(1)
    rospy.loginfo("-- Received ROS PointCloud2 message. --")
    rospy.loginfo("-- Waiting for StartButton Signal --")

    s.buttonStart(0)
    # if s.buttonInterface():
    #     print ("Got it Start signal")
    time.sleep(3)
    if s.buttonInterface() is True:
        s.lightControl(2)
        rospy.loginfo("Got the Start signal")

        t1 = time.time()
        global received_ros_cloud
        field_names=[field.name for field in received_ros_cloud.fields]
        cloud_data = list(pc2.read_points(received_ros_cloud, skip_nans=False, field_names = field_names))

        # Check empty
        open3d_cloud = open3d.geometry.PointCloud()
        if len(cloud_data)==0:
            print("Converting an empty cloud")
            return None

        # Set open3d_cloud
        if "rgb" in field_names:
            IDX_RGB_IN_FIELD=3 # x, y, z, rgb
            
            # Get xyz
            xyz = [(x,y,z) for x,y,z,rgb in cloud_data ] # (why cannot put this line below rgb?)

            # Get rgb
            # Check whether int or float
            if type(cloud_data[0][IDX_RGB_IN_FIELD])==float: # if float (from pcl::toROSMsg)
                rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
            else:
                rgb = [convert_rgbUint32_to_tuple(rgb) for x,y,z,rgb in cloud_data ]

            # combine
            open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))
            open3d_cloud.colors = open3d.utility.Vector3dVector(np.array(rgb)/255.0)
        else:
            xyz = [(x,y,z) for x,y,z in cloud_data ] # get xyz
            open3d_cloud.points = open3d.utility.Vector3dVector(np.array(xyz))

        open3d_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd = open3d_cloud
        # open3d.visualization.draw_geometries([open3d_cloud])
        t2 = time.time()
        print("Convertion completed")
        print("size of pointcloud : ",np.array(open3d_cloud.colors).shape)
        t3 = time.time()
        rgb_crop = np.asarray(open3d_cloud.colors)
        print("Original_pcd_rgb shape : ", rgb_crop.shape)
        rgb_crop = np.reshape(rgb_crop,(1536,2048,3))
        rgb_crop = rgb_crop[550:550+450,670:670+650]
        image = (rgb_crop*255).astype('uint8')
        # cv2.imshow("Image window", image)
        # cv2.imwrite(r"/home/madan/Master_thesis/ros_output.jpeg",cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        act_img = rgb_crop.copy()
        img = np.pad(act_img,((103,103), (3,3), (0,0)),"constant",constant_values=(0))
        pred_img = np.reshape(img,(1,656,656,3))
        pred_img = np.expand_dims(pred_img, axis=-1)
        pred_img = normalize(pred_img, axis=1)
        pred_img = np.reshape(pred_img,(1,656,656,3))

        pred = model.predict(pred_img)
        out_img = np.reshape(pred,(656,656,5))
        class_1 = np.zeros((656,656,3))
        class_2 = np.zeros((656,656,3))
        class_1[:,:,0:2] = out_img[:,:,0:2].copy()
        class_1[:,:,2] = out_img[:,:,3].copy()

        class_2[:,:,0:2] = out_img[:,:,0:2].copy()
        class_2[:,:,2] = out_img[:,:,3].copy()
        class_1[:,:,0] = 0
        class_1[:,:,2] = 0
        class_2[:,:,0] = 0
        class_2[:,:,1] = 0
        class_dict = {}
        class_dict["cube"] = class_2
        class_dict["octogon"] = class_1
        class_keys = class_dict.keys()
        t4 = time.time()
        
        # plt.imshow(class_1)
        # # cv2.imwrite("D:\\01_Thesis\\unet\\train\\masks\\model_68.jpeg",class_1[103:103+450,:]*125)
        ############################################################################################################
        filt_cont = []
        obj_class = []
        for class_type in class_keys:
            data = 255 * class_dict[class_type] # Now scale by 255
            img = data.astype(np.uint8)

            kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
            imgLaplacian = cv2.filter2D(img, cv2.CV_32F, kernel)
            sharp = np.float32(img)
            imgResult = sharp - imgLaplacian
            # convert back to 8bits gray scale
            imgResult = np.clip(imgResult, 0, 255)
            imgResult = imgResult.astype('uint8')
            imgLaplacian = np.clip(imgLaplacian, 0, 255)
            imgLaplacian = np.uint8(imgLaplacian)
            # plt.imshow( imgLaplacian)
            # plt.imshow(imgResult)

            # Create binary image from source image
            bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # plt.imshow(bw)
            # Perform the distance transform algorithm
            dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
            # Normalize the distance image for range = {0.0, 1.0}
            # so we can visualize and threshold it
            cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
            # plt.imshow( dist)
            # Threshold to obtain the peaks
            # This will be the markers for the foreground objects
            _, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)
            # Dilate a bit the dist image
            kernel1 = np.ones((3,3), dtype=np.uint8)
            dist = cv2.dilate(dist, kernel1)
            # plt.imshow( dist)
            # Create the CV_8U version of the distance image
            # It is needed for findContours()
            dist_8u = dist.astype('uint8')
            # Find total markers
            contours, _= cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Create the marker image for the watershed algorithm
            markers = np.zeros(dist.shape, dtype=np.int32)
            # Draw the foreground markers
            for i in range(len(contours)):
                cv2.drawContours(markers, contours, i, (i+1), -1)
            # Draw the background marker
            cv2.circle(markers, (5,5), 3, (255,255,255), -1)
            markers_8u = (markers * 10).astype('uint8')



            # Perform the watershed algorithm
            cv2.watershed(imgResult, markers)
            #mark = np.zeros(markers.shape, dtype=np.uint8)
            mark = markers.astype('uint8')
            mark = cv2.bitwise_not(mark)
            # uncomment this if you want to see how the mark
            # image looks like at that point
            #cv.imshow('Markers_v2', mark)
            # Generate random colors
            colors = []
            for contour in contours:
                colors.append((random.randint(0,256), random.randint(0,256), random.randint(0,256)))
            # Create the result image
            dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
            # Fill labeled objects with random colors
            for i in range(markers.shape[0]):
                for j in range(markers.shape[1]):
                    index = markers[i,j]
                    if index > 0 and index <= len(contours):
                        dst[i,j,:] = colors[index-1]
            # Visualize the final image
            # plt.imshow( dst)
            # cv2.imwrite("D:\\01_Thesis\\unet\\train\\masks\\model_68.jpeg",dst[103:103+450,:]*125)
            re_cont = []
            temp = []
            for cont in contours:
                for pts in cont:
                    temp.append(pts[0])
                re_cont.append(temp)
                temp = []
                
            

            for cnt in re_cont:
                fg = 0
                for pts in cnt:
                    if pts[1] > 500 or pts[1] < 180 :
                        fg = 1
                if fg == 0:
                    filt_cont.append(cnt)
                    obj_class.append(class_type)
                    
            # blank = np.zeros((656,656,3))
            # isClosed = True
            
            # # Blue color in BGR
            # color = (255, 255, 255)
            
            # # Line thickness of 2 px
            # thickness = 2
            # for cnt in filt_cont:
            #     blank = cv2.polylines(blank, [np.array(cnt)], 
            #                     isClosed, color, thickness)
                
            # # plt.imshow(blank)
            # # font
            # font = cv2.FONT_HERSHEY_SIMPLEX
            
            # # org
            # org = (50, 50)
            
            # # fontScale
            # fontScale = 1
            
            # # Blue color in BGR
            # color = (255, 255, 255)
            
            # # Line thickness of 2 px
            # thickness = 2
            
            # # Using cv2.putText() method
            # image = cv2.putText(blank, 'OpenCV', org, font, 
            #                    fontScale, color, thickness, cv2.LINE_AA)

            # c = 1
            # for cnt in filt_cont:
                
            #     arr = np.array(cnt)
            #     x = np.max(arr[...,0])
            #     y = np.max(arr[...,1]) -103
            #     blank = cv2.putText(act_img, str(c), (x,y), font, 
            #                 fontScale, color, thickness, cv2.LINE_AA)
            #     c = c +1
            # # plt.imshow(blank)
            # image = (blank*255).astype('uint8')
            # cv2.imwrite(r"/home/madan/Master_thesis/ros_output.jpeg",cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        t5 = time.time()
        
        points = np.asarray(pcd.points)
        points = np.reshape(points,(1536,2048,3))
        # points_crop = points[500:500+450,680:680+650,:].copy()
        points_crop = points[550:550+450,670:670+650,:].copy()
        # cnt = filt_cont[0]
        # cnt_arr = np.asarray(cnt)
        # x_max = max(cnt_arr[:,0]) #-103
        # x_min = min(cnt_arr[:,0])  #-103
        # y_max = max(cnt_arr[:,1]) -103
        # y_min = min(cnt_arr[:,1]) -103
        # roi_points = points_crop[y_min:y_max,x_min:x_max,:]

        # print('No of objects detected :',len(filt_cont))
        # print('No of objects detected :',len(obj_class))
        

        t6 = time.time()


        points = np.asarray(pcd.points)
        points = np.reshape(points,(1536,2048,3))
        points_crop = points[550:550+450,670:670+650,:].copy()
        rospy.loginfo("No of Objects detected : %d",len(filt_cont))
        print("Objects are : ", obj_class )
        for num_obj in range(len(filt_cont)):
            rospy.loginfo("Object to be picked : %s", obj_class[num_obj])
            cnt = filt_cont[num_obj]
            cnt_arr = np.asarray(cnt)
            x_max = max(cnt_arr[:,0]) #-103
            x_min = min(cnt_arr[:,0])  #-103
            y_max = max(cnt_arr[:,1]) -103
            y_min = min(cnt_arr[:,1]) -103
            roi_points = points_crop[y_min:y_max,x_min:x_max,:]
            # blank_crop = blank[y_min:y_max,x_min:x_max,:]  ## Try using act_img to show color of object
            M = cv2.moments(np.asarray(cnt))
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"]) -103
            roi_pcd = o3d.geometry.PointCloud()
            pk_pcd = o3d.geometry.PointCloud()
            pk_points = np.reshape(points_crop[cY,cX,:],(1,3))
            pk_color = np.reshape(np.array([255,0,0]),(1,3))
            points_crop_rs = np.reshape(roi_points,(roi_points.shape[0]*roi_points.shape[1],3))
            roi_pcd.points = o3d.utility.Vector3dVector(points_crop_rs)

            # blank_crop_rs = np.reshape(blank_crop,(blank_crop.shape[0]*blank_crop.shape[1],3))

            # roi_pcd.colors = o3d.utility.Vector3dVector(blank_crop_rs)
            pk_pcd.colors = o3d.utility.Vector3dVector(pk_color)
            pk_pcd.points = o3d.utility.Vector3dVector(pk_points)
            
            R_pt = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
            # R = np.array([[0,1,0],[1,0,0],[0,0,-1]])

            R_cam = np.array([[0,1,0],[1,0,0],[0,0,-1]])
            R = np.matmul(R_pt,R_cam )
            # print("R : ", R)
            # T_cam = np.array([[0.35],[-0.027],[0.753]])
            T_cam = np.array([0.35,-0.027,0.753])
            # print("T_cam : ", T_cam)
            P_cam = np.array([points_crop[cY,cX,:][0],points_crop[cY,cX,:][1],1*points_crop[cY,cX,:][2]])#.transpose()
            # print("P_cam : ", P_cam)
            P_robot = np.matmul(R,P_cam ) + T_cam
            P_robot

            # obj_sq = 0
            df = 0.0135
            P_robot = P_robot.tolist()
            x = (P_robot[0]) #- 0.03)
            y = (P_robot[1] + 0.01)
            # if obj_sq == 1:
            #     df = 0.075
            z = (P_robot[2] + df)

            # print(x,y,z,type(x),type(y),type(z))
            rospy.loginfo("Object localised at  : %f m along x axis", x)
            rospy.loginfo("Object localised at  : %f m along y axis", y)
            rospy.loginfo("Object localised at  : %f m along z axis", z)

            speed = 0.8
            ##Home
            s.vacuumControl(0)
            s.moveLinear(0.060, 0.0, 0.525, 0.0, 1.0, 0.0, 0.0, speed) #p8
            # move
            #s.moveLinear(0.290, 0.095, 0.438, 0.0, 1.0, 0.0, 0.0, speed) #p1
            s.moveLinear(0.391, 0.011, 0.284, 0.0, 1.0, 0.0, 0.0, speed) #p2
            s.moveLinear(0.396, -0.0177, 0.198, 0.0, 1.0, 0.0, 0.0, speed) #p3
            s.vacuumControl(1)
            s.moveLinear(x, y, z, 0.0, 1.0, 0.0, 0.0, 0.1) ## pick    [0.424     , 0.055     , 0.00899998]

            s.moveLinear(x, y, 0.198, 0.0, 1.0, 0.0, 0.0, speed) #modified p3
            s.moveLinear(0.391, 0.011, 0.284, 0.0, 1.0, 0.0, 0.0, speed) #p2
            # s.moveLinear(0.290, 0.095, 0.438, 0.0, 1.0, 0.0, 0.0, speed) #p1
            if obj_class[num_obj] == "octogon":
                s.moveLinear(0.0528, 0.282, 0.323, 0.0, 1.0, 0.0, 0.0, 0.5) #p5
                s.moveLinear(0.052, 0.292, 0.175, 0.0, 1.0, 0.0, 0.0, 0.5) #p6
                s.vacuumControl(0)
                s.moveLinear(0.055, 0.310, 0.421, 0.0, 1.0, 0.0, 0.0, speed) #p7 drop


            else:
                s.moveLinear(0.218, -0.011, 0.378, 0.0, 1.0, 0.0, 0.0, speed) #p5
                s.moveLinear(0.041, -0.214, 0.378, 0.0, 1.0, 0.0, 0.0, speed) #p6
                s.moveLinear(0.0626, -0.315, 0.251, 0.0, 1.0, 0.0, 0.0, speed) #p7 drop
            s.vacuumControl(0)
            s.moveLinear(0.060, 0.0, 0.525, 0.0, 1.0, 0.0, 0.0, speed) #p8
        filt_cont = []
        obj_class = []
        t7 = time.time()

        print("Time taken for conversion of pointclouds:", t2 - t1)
        print("Time taken for Model prediction :", t4 - t3)
        print("Time taken for Postprocess :", t5 - t4)
        print("Time taken for Pick and Place operation :", t7 - t6)
    
    else:
        rospy.loginfo("-- No Start Signal Received -- ")
        rospy.loginfo("-- Refreshing the Inputs -- ")


def listener():
     
     # In ROS, nodes are uniquely named. If two nodes with the same
     # name are launched, the previous one is kicked off. The
     # anonymous=True flag means that rospy will choose a unique
     # name for our 'listener' node so that multiple listeners can
     # run simultaneously.
     rospy.init_node('pc_listener', anonymous=True)
 
     rospy.Subscriber("/points2", PointCloud2, callback, queue_size=1)
    #  global received_ros_cloud
    #  Process()
     rospy.Timer(rospy.Duration(1), Process)
     
 
     # spin() simply keeps python from exiting until this node is stopped
    #  rospy.spin()


def set_all_default():
    s.lightControl(3)
    time.sleep(2)
    s.lightControl(0)
    print("Shutting down")
    s.buttonStart(0)
if __name__ == '__main__':

    while not rospy.is_shutdown():
        listener()
    # Process()
        rospy.spin()

    rospy.on_shutdown(set_all_default)
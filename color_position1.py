# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

import pyrealsense2 as rs
import cv2, math
from decimal import Decimal, ROUND_HALF_UP
import datetime








# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())
file_name = 'data_c1/'+datetime.datetime.now().strftime("%m%d-%H%M")+'c1'



# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
# greenLower = (29, 86, 6)
# greenUpper = (64, 255, 255)
# blue
# greenLower = (0, 0, 0)
# greenUpper = (360, 200, 50)


greenLower = (100, 43, 46)
greenUpper = (124, 255, 255)
# red
# greenLower = (0, 143, 46)
# greenUpper = (0, 255, 255)
# White
# greenLower = (0, 0, 221)
# greenUpper = (180, 30, 255)
pts = deque(maxlen=args["buffer"])
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])
# allow the camera or video file to warm up
time.sleep(2.0)







def detect_video():

    center_coordinates_array = []
    theta = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    config = rs.config()
    # config.enable_device('148122071850')
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # get camera intrinsics
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:

        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue
        

        






        color_image = np.asanyarray(color_frame.get_data())


        frame = color_image
        result = color_image



        # keep looping
        # while True:
        # grab the current frame
        # frame = vs.read()
        # # handle the frame from VideoCapture or VideoStream
        # frame = frame[1] if args.get("video", False) else frame
        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video


        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = imutils.resize(frame, width=640)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid

            
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            center_coordinates_array = [[x,y],]

            




        # # Get center position from RGB channel
        # color_image = np.asanyarray(color_frame.get_data())
        # #depth_image = np.asanyarray(depth_frame.get_data())
            
        # image = Image.fromarray(color_image)
        # image, center_coordinates_array = yolo.detect_image(image)
        # result = np.asarray(image)
        # print("###########",center_coordinates_array)
        # # End of getting center


        is_detect = False
        if(len(center_coordinates_array) > 0):
            is_detect = True
            for i in range(len(center_coordinates_array)):
                dist = depth_frame.get_distance(int(center_coordinates_array[i][0]), int(center_coordinates_array[i][1]))*1000 #convert to mm

                #calculate real world coordinates
                Xtemp = dist*(center_coordinates_array[i][0] -intr.ppx)/intr.fx
                Ytemp = dist*(center_coordinates_array[i][1] -intr.ppy)/intr.fy
                Ztemp = dist

                Xtarget = Xtemp - 35 #35 is RGB camera module offset from the center of the realsense
                Ytarget = -(Ztemp*math.sin(theta) + Ytemp*math.cos(theta))
                Ztarget = Ztemp*math.cos(theta) + Ytemp*math.sin(theta)

                coordinates_text = "(" + str(Decimal(str(Xtarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + \
                                   ", " + str(Decimal(str(Ytarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + \
                                   ", " + str(Decimal(str(Ztarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + ")"

                # print("###########",coordinates_text)
                print(coordinates_text, "x, y : ", '------', center_coordinates_array[i])
                cv2.putText(result, text=coordinates_text, org=(int(center_coordinates_array[i][0])-160, int(center_coordinates_array[i][1])),
                            fontFace=font, fontScale = 1, color=(255,255,255), thickness = 2, lineType=cv2.LINE_AA)

                
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.rectangle(result, (int(x)-int(radius), int(y)-int(radius)),
                            ((int(x)+int(radius)), (int(y)+int(radius))), (255, 0, 255), 3, 1)
                # cv2.circle(frame, (int(x), int(y)), int(radius),
                #            (0, 255, 255), 2)
                cv2.circle(result, center, 5, (0, 0, 255), -1)
                area = int(radius)*int(radius)
                
                # update the points queue
                pts.appendleft(center)
                # loop over the set of tracked points
                for i in range(1, len(pts)):
                    # if either of the tracked points are None, ignore
                    # them
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    # otherwise, compute the thickness of the line and
                    # draw the connecting lines
                    thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
                    cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        if is_detect:
            with open(file_name+'.csv', "a+") as f:
                f.write(f"{Decimal(str(Xtarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)}, {Decimal(str(Ytarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)}, {Decimal(str(Ztarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)}, {time.time()}\n") 
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0

        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        #cv2.imshow("result_depth", depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



detect_video()
























def findcolor():

    # keep looping
    # while True:
    # grab the current frame
    frame = vs.read()
    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video


    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # # only proceed if the radius meets a minimum size
        # if radius > 10:
        #     # draw the circle and centroid on the frame,
        #     # then update the list of tracked points
        #     cv2.rectangle(frame, (int(x)-int(radius), int(y)-int(radius)),
        #                 ((int(x)+int(radius)), (int(y)+int(radius))), (255, 0, 255), 3, 1)
        #     # cv2.circle(frame, (int(x), int(y)), int(radius),
        #     #            (0, 255, 255), 2)
        #     cv2.circle(frame, center, 5, (0, 0, 255), -1)
        # area = int(radius)*int(radius)
        # cv2.putText(frame, str(area), (50, 100),
        #             cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)
    # update the points queue
    # pts.appendleft(center)
    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
   

    # # if we are not using a video file, stop the camera video stream
    # if not args.get("video", False):
    #     vs.stop()
    # # otherwise, release the camera
    # else:
    #     vs.release()
    # # close all windows
    # # cv2.destroyAllWindows()

    return x, y

# while(True):
#     x,y = findcolor()
#     print(x,y)
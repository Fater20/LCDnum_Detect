# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

import math
# Usual color HSV value
color_dist = {
    'red1': {'lower': np.array([0, 43, 46]), 'upper': np.array([10, 255, 255])},
    'red2': {'lower': np.array([156, 43, 46]), 'upper': np.array([180, 255, 255])},
    'blue': {'lower': np.array([100, 43, 46]), 'upper': np.array([124, 255, 255])},
    'green': {'lower': np.array([35, 43, 46]), 'upper': np.array([77, 255, 255])},
    'yellow': {'lower': np.array([26, 43, 46]), 'upper': np.array([34, 255, 255])},
    'black': {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 46])},
    }

# Empty function
def empty(a):
    pass

# Image stack function
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

#if device_product_line == 'L500':
#    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
#else:
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Streaming loop
try:
    while True: 
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image
        
        # Get frames
        color_frame = frames.get_color_frame()

        # Validate that both frames are valid
        if  not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        imgResult = color_image.copy()

        # Transfer rgb to hsv
        grey_image=cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        #threshold(midImage, midImage, 100, 255, 0);
        gus_image = cv2.GaussianBlur(grey_image, (5, 5), 0)
        edges = cv2.Canny(gus_image, 50, 150, apertureSize=3)

        imgStack = stackImages(0.6,([color_image, imgResult],[gus_image, edges]))

        cv2.namedWindow('Color Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Color Example', imgStack)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
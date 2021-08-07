########################################################################
##                         LCD Number Detect                          ##
########################################################################

# LCD(Digital) Number Detect (use color to do image segmentation)
#
# Usual 7-segment digital number are red (other color is ok, but need to
# be different with the background), so we can find the area of number by
# color. We also need to do eroding and other image preprocessing steps 
# to make every tube of digital numbers (and every digital number) connect 
# with each otherbefore finding the biggest area of red(The area where 
# numbers are most likely to appear.)

import numpy as np
import cv2

import pyrealsense2 as rs

from imutils.perspective import four_point_transform
from imutils import contours
import imutils

from sklearn.neighbors import KNeighborsClassifier as KNN

# Usual color HSV value
color_dist = {
    'red1': {'lower': np.array([0, 43, 46]), 'upper': np.array([10, 255, 255])},
    'red2': {'lower': np.array([156, 43, 46]), 'upper': np.array([180, 255, 255])},
    'blue': {'lower': np.array([100, 80, 46]), 'upper': np.array([124, 255, 255])},
    'green': {'lower': np.array([35, 43, 35]), 'upper': np.array([90, 255, 255])},
    }

# Dictionary of 7-segment digital number 0-9
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 0, 1): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
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

# Measure the distance from the target center to the camera 
def depth_measure(depth_image, center, depth_scale):
    # Whether the center in the edge of the depth image
    if(center[0]>0 & center[0]<639 & center[1]>0 & center[1]<479):
        # Surrounding matrix
        s_array = np.array([[-1, -1],[-1,0],[-1,1],[0, -1],[0,0],[0,1],[1, -1],[1,0],[1,1]])
        
        # Center surrounding matrix
        cs_array = np.array(center)+s_array

        # Measure the distance
        distance = np.sum(depth_image[cs_array[:,1],cs_array[:,0]])/9 * depth_scale
        return distance
    
    # Can not measure
    return 0

# Depth filter
def depth_filter(color_image_src, depth_image_src, depth_min, depth_max):
    # Remove background - Set pixels further than clipping_distance to grey
    white_color = 255
    depth_image_3d = np.dstack((depth_image_src,depth_image_src,depth_image_src)) #depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > depth_max) | (depth_image_3d <= 0) | (depth_image_3d < depth_min), white_color, color_image_src)
    return bg_removed


#数据初始化
s = './num'  #特征图像所在位置
num = 2000   #样本总数  0-9各200个
row = 130  #特征图像的行数
col = 72  #特征图像的列数
a = np.zeros((num,row,col))  #存储所有样本的数值

#存储图像
n = 0
for i in range(0,10):
    for j in range(0,200):
        a[n,:,:] = cv2.imread(s + '/' + str(i) + '_class/' + str(j).zfill(5) + '.jpg',0)
        n = n+1

#提取样本图像特征
feature = np.zeros((num,row,col))  #用来存储所有样本的特征值
for ni in range(0,num):
    for nr in range(0,row):
        for nc in range(0,col):
            if a[ni,nr,nc] == 255:
                feature[ni,nr,nc]=1

f = feature   #简化变量名称
#将feature处理为单行形式
train = feature[:,:].reshape(-1,row*col).astype(np.float32)
#贴标签
trainLabels = [int(i/(num/10)) for i in range(0,num)]
trainLabels = np.asarray(trainLabels)

#构建kNN分类器
neigh =KNN(n_neighbors = 5, algorithm = 'auto')
#拟合模型, trainingMat为训练矩阵,hwLabels为对应的标签
neigh.fit(train, trainLabels)

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
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
# clipping_distance_in_meters meters away
clipping_max_distance_in_meters = 1     #1 meter
clipping_min_distance_in_meters = 0.1   #0.1  meter

clipping_distance_max = clipping_max_distance_in_meters / depth_scale
clipping_distance_min = clipping_min_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Creat new window 'Trackbar'
cv2.namedWindow('Trackbars')
cv2.resizeWindow("Trackbars",640,120)

# Trackbar seetings
cv2.createTrackbar("Dmin","Trackbars", 10,200,empty)
cv2.createTrackbar("Dmax","Trackbars",800,800,empty)

n = 0
# Streaming loop
try:
    while True:
        # Get HSV range & depth range
        d_min = cv2.getTrackbarPos("Dmin","Trackbars")
        d_max = cv2.getTrackbarPos("Dmax","Trackbars")

        # # Get HSV lower and upper array
        # hsv_lower = np.array([h_min,s_min,v_min])
        # hsv_upper = np.array([h_max,s_max,v_max])

        # Depth distance transform to depth color based on depth_scale
        clipping_distance_min = d_min / 100 / depth_scale
        clipping_distance_max = d_max / 100 / depth_scale

        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        # Transform depth_frame and color_frame to numpy array
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Create some images to display
        imgResult = color_image.copy()
        filter_mask = color_image.copy()

        warped = color_image.copy()
        dilate_warped = color_image.copy()
        closing_warped = color_image.copy()
        # Create the empty color list
        color_list = list()

        # Remove background - Set pixels further than clipping_distance to grey
        bg_removed = depth_filter(color_image, depth_image, clipping_distance_min, clipping_distance_max)

        # Transfer rgb to hsv
        image_hsv=cv2.cvtColor(bg_removed, cv2.COLOR_BGR2HSV)

        # Gaussian Blur
        image_gus = cv2.GaussianBlur(image_hsv, (5, 5), 0)

        # Generate mask ( The HSV value of red has two ranges)
        #mask = cv2.inRange(image_hsv,lower,upper)
        mask1 = cv2.inRange(image_hsv,color_dist['red1']['lower'],color_dist['red1']['upper'])
        mask2 = cv2.inRange(image_hsv,color_dist['red2']['lower'],color_dist['red2']['upper'])
        mask = mask1+mask2

        # Set kernel as 3*3
        kernel = np.ones((3,3),np.uint8)

        # Erode image
        erode_mask = cv2.erode(mask, kernel, iterations=1)

        # Dilate image
        opening_mask = cv2.dilate(erode_mask, kernel, iterations=1)

        # Find all the contours in the erode_mask
        contour, hierarchy = cv2.findContours(opening_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        digitCnts = []

        # Try to reduce the effects of highlights and chromatic aberration(色差)
        # After filling small areas, make valid areas stable and connected
        # Whether the contour exist
        if np.size(contour)>0:
            #area = cv2.contourArea(contours)
            contours_area = np.zeros((len(contour),))
            for i in range(len(contour)):
                area = cv2.contourArea(contour[i])
                
                # Fill small areas
                if area < 50:
                    cv2.drawContours(opening_mask,[contour[i]],0,0,-1)

            # Make every number connects with each other and become a complete area
            kernel5 = np.ones((5,5),np.uint8)
            filter_mask = cv2.dilate(opening_mask, kernel5, iterations=20)
            filter_mask = cv2.erode(filter_mask, kernel5, iterations=10)

            contours_filter, hierarchy_filter = cv2.findContours(filter_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the biggest area which is the digital area
            if np.size(contours_filter)>0:
                c = max(contours_filter, key=cv2.contourArea)

                rect = cv2.minAreaRect(c)
                #rect = cv2.boundingRect(c)
                box = cv2.boxPoints(rect)

                # cv2.drawContours(imgResult, [np.int0(box)], -1, (0, 255, 255), 2)
                # cv2.circle(imgResult, tuple(map(int,list(rect[0]))), 2, (255, 0, 0), 2)

                # Extract digital area through four-point perspective transformation
                warped = four_point_transform(opening_mask, np.int0(box).reshape(4, 2))
                imgResult = four_point_transform(color_image, np.int0(box).reshape(4, 2))

                # Make every digital number average and clear
                # Dilate image
                dilate_warped = cv2.dilate(warped, kernel, iterations=1)

                # Erode image
                closing_warped = cv2.erode(dilate_warped, kernel, iterations=2)

                cnts = cv2.findContours(closing_warped.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                for c in cnts:
                    (x, y, w, h) = cv2.boundingRect(c)
                    print(x,y,w,h)
                    
                    area = w*h

                    # Filter out the appropriate size area
                    if w >= 15 and (h >= 20 and h <= 80):
                        digitCnts.append(c)
                    elif (w >=5 and w < 15) and (h >= 20 and h <= 80):
                        digitCnts.append(c)
                
                if digitCnts:
                    # Sort these contours from left to right
                    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
                    digits = []

                    i = 0
                    
                    for c in digitCnts:
                        # Get the ROI area
                        (x, y, w, h) = cv2.boundingRect(c)
                        roi = closing_warped[y:y + h, x:x + w]
                        (roiH, roiW) = roi.shape
                        
                        (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
                        # tilt_dW = int(0.05*dW) # Tilt compensation
                        dHC = int(roiH * 0.1)

                        if w>=15:
                            roi_re = cv2.resize(roi,(72,130))

                        # '1'
                        else:
                            black_block = np.zeros((h,w))
                            roi_stack=np.hstack((black_block,roi,black_block))
                            roi_re = cv2.resize(roi_stack,(72,130))
                            
                        roi_re = cv2.erode(roi_re, kernel, iterations=1)
                        roi_re = cv2.dilate(roi_re, kernel, iterations=3)

                        o = roi_re
                        of = np.zeros((row,col))  #存储待识别图像特征值
                        for nr in range(0,row):
                            for nc in range(0,col):
                                if o[nr,nc] == 255:
                                    of[nr,nc]=1

                        test = of.reshape(-1,row*col).astype(np.float32)

                        #获得预测结果
                        classifierResult = neigh.predict(test)
                        distances, indices = neigh.kneighbors(test)

                        
                        print("距离当前点最近的5个邻居为:",trainLabels[indices])
                        print("距离为:",distances)
                        print("分类返回结果为%d" % (classifierResult))
                        

                        # Query and display the detecting result
                        try:
                            digits.append(int(classifierResult))
                            #cv2.imwrite('./train_class/'+str(int(classifierResult))+ '/' +str(n).zfill(5)+'.jpg',roi_re)
                            n = n + 1
                            cv2.rectangle(imgResult, (x, y), (x + w, y + h), (0, 255, 0), 1)
                            cv2.putText(imgResult, str(int(classifierResult)), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                        except KeyError:
                            print("no detect")
                    # Display the detecting result
                    print(digits)
                    print("######")

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        imgStack = stackImages(0.4,([color_image, filter_mask, imgResult],[mask, erode_mask, opening_mask],[warped,dilate_warped,closing_warped]))

        cv2.namedWindow('Depth Filter and Color locate', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Depth Filter and Color locate', imgStack)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
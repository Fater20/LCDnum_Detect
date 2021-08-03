#####################################################
##         Depth Filter and Color locate           ##
#####################################################

#

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

from imutils.perspective import four_point_transform
from imutils import contours
import imutils

# Color information class
#   type : shape type
#   center: coordinates of the center
#   depth: depth of the center
#   vertices: 
class color_info:
    def __init__(self, center, depth, vertex):
      self.center = center
      self.depth = depth
      self.vertex = vertex

# Shape information class
#   type : shape type
#   center: coordinates of the center
#   depth: depth of the center
#   hull: the hull of the shape (For triangle and rectangle(square), hulls are the vertices; For circle, hull is the first point of contour)
class shape_info:
    def __init__(self, type, center, depth, hull):
      self.type = type
      self.center = center
      self.depth = depth
      self.hull = hull

# Usual color HSV value
color_dist = {
    'red1': {'lower': np.array([0, 43, 46]), 'upper': np.array([10, 255, 255])},
    'red2': {'lower': np.array([156, 43, 46]), 'upper': np.array([180, 255, 255])},
    'blue': {'lower': np.array([100, 80, 46]), 'upper': np.array([124, 255, 255])},
    'green': {'lower': np.array([35, 43, 35]), 'upper': np.array([90, 255, 255])},
    }

DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 0, 1): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 0, 1, 1, 1, 1, 1): 6,
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

# depth filter
def depth_filter(color_image_src, depth_image_src, depth_min, depth_max):
    # Remove background - Set pixels further than clipping_distance to grey
    white_color = 255
    depth_image_3d = np.dstack((depth_image_src,depth_image_src,depth_image_src)) #depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > depth_max) | (depth_image_3d <= 0) | (depth_image_3d < depth_min), white_color, color_image_src)
    return bg_removed

# Color detect and locate (locate all the color targets)
# Also filter the region not in the region of interest
def color_detect(color_image_src, depth_image_src, hsv_lower, hsv_upper, depth_min, depth_max, depth_scale):
    # Copy the color image
    imgResult = color_image_src.copy()

    # Create the empty color list
    color_list = list()

    # Gaussian Blur
    image_gus = cv2.GaussianBlur(color_image_src, (5, 5), 0)

    # Remove background - Set pixels further than clipping_distance to grey
    bg_removed = depth_filter(image_gus, depth_image_src, depth_min, depth_max)

    # white_color = 255
    # depth_image_3d = np.dstack((depth_image_src,depth_image_src,depth_image_src)) #depth image is 1 channel, color is 3 channels
    # bg_removed = np.where((depth_image_3d > clipping_distance_max) | (depth_image_3d <= 0) | (depth_image_3d < clipping_distance_min), white_color, image_gus)

    # Transfer rgb to hsv
    image_hsv=cv2.cvtColor(bg_removed, cv2.COLOR_BGR2HSV)

    # Generate mask ( The HSV value of red has two ranges)
    #mask = cv2.inRange(image_hsv,lower,upper)
    mask1 = cv2.inRange(image_hsv,color_dist['red1']['lower'],color_dist['red1']['upper'])
    mask2 = cv2.inRange(image_hsv,color_dist['red2']['lower'],color_dist['red2']['upper'])
    mask = mask1+mask2

    # Set kernel as 3*3
    kernel = np.ones((3,3),np.uint8)
    # Erode image
    erode_mask = cv2.erode(mask, kernel, iterations=3)
    # Dilate image
    opening_mask = cv2.dilate(erode_mask, kernel, iterations=5)

    # Find all the contours in the erode_mask
    contours, hierarchy = cv2.findContours(opening_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    # Whether the contour exist
    if np.size(contours)>0:
        #area = cv2.contourArea(contours)
        contours_area = np.zeros((len(contours),))
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            
            if area < 200:
                cv2.drawContours(opening_mask,[contours[i]],0,0,-1)

        # Try to reduce the effects of highlights and chromatic aberration(色差)
        # After filling small areas, make valid areas stable and connected
        kernel5 = np.ones((5,5),np.uint8)
        opening_mask = cv2.dilate(opening_mask, kernel5, iterations=5)

        contours_filter, hierarchy_filter = cv2.findContours(opening_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if np.size(contours_filter)>0:
            c = max(contours_filter, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            #rect = cv2.boundingRect(c)
            box = cv2.boxPoints(rect)
            cv2.drawContours(imgResult, [np.int0(box)], -1, (0, 255, 255), 2)
            cv2.circle(imgResult, tuple(map(int,list(rect[0]))), 2, (255, 0, 0), 2)

        contours_info = np.zeros((len(contours_filter),3), dtype = int)
        # for c in contours_filter:
        #     # Find the minimum enclosing rectangle
        #     rect = cv2.minAreaRect(c)

        #     # Get the rectangle's four corner points
        #     box = cv2.boxPoints(rect)

        #     # Draw the contour in red
        #     cv2.drawContours(imgResult, [np.int0(box)], -1, (0, 255, 255), 2)
            
        #     # Draw the center of the rectangle in blue
        #     cv2.circle(imgResult, tuple(map(int,list(rect[0]))), 2, (255, 0, 0), 2)
        
        #     center = list(map(int,list(rect[0])))
        #     distance = depth_measure(depth_image_src, center, depth_scale)
        #     cv2.putText(imgResult,str(int(distance*100))+"cm",tuple(center),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
        #     # Write color information(center coordinates & depth) into color_list
        #     c = color_info([center[0],center[1]],int(distance*100),np.fix(box))
        #     color_list.append(c)

    return color_list, imgResult, opening_mask

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
clipping_min_distance_in_meters = 0.2   #0.2 meter

clipping_distance_max = clipping_max_distance_in_meters / depth_scale
clipping_distance_min = clipping_min_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Creat new window 'Trackbar'
cv2.namedWindow('Trackbars')
cv2.resizeWindow("Trackbars",640,480)

# Trackbar seetings
cv2.createTrackbar("Hmin","Trackbars",  0,180,empty)
cv2.createTrackbar("Hmax","Trackbars",180,180,empty)
cv2.createTrackbar("Smin","Trackbars",  0,255,empty)
cv2.createTrackbar("Smax","Trackbars",255,255,empty)
cv2.createTrackbar("Vmin","Trackbars",  0,255,empty)
cv2.createTrackbar("Vmax","Trackbars",255,255,empty)
cv2.createTrackbar("Dmin","Trackbars", 20,800,empty)
cv2.createTrackbar("Dmax","Trackbars",800,800,empty)

# Streaming loop
try:
    while True:
        # Get HSV range & depth range
        h_min = cv2.getTrackbarPos("Hmin","Trackbars")
        h_max = cv2.getTrackbarPos("Hmax","Trackbars")
        s_min = cv2.getTrackbarPos("Smin","Trackbars")
        s_max = cv2.getTrackbarPos("Smax","Trackbars")
        v_min = cv2.getTrackbarPos("Vmin","Trackbars")
        v_max = cv2.getTrackbarPos("Vmax","Trackbars")
        d_min = cv2.getTrackbarPos("Dmin","Trackbars")
        d_max = cv2.getTrackbarPos("Dmax","Trackbars")

        # Get HSV lower and upper array
        hsv_lower = np.array([h_min,s_min,v_min])
        hsv_upper = np.array([h_max,s_max,v_max])

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

        #center, distance, imgResult, bg_removed, image_hsv, opening_mask = maxColor_locate(color_image, depth_image, hsv_lower, hsv_upper, clipping_distance_min, clipping_distance_max, depth_scale)

        imgResult = color_image.copy()
        filter_mask = color_image.copy()
        opening_warped = color_image.copy()
        warped = color_image.copy()
        # Create the empty color list
        color_list = list()

        # Gaussian Blur
        image_gus = cv2.GaussianBlur(color_image, (5, 5), 0)

        # Remove background - Set pixels further than clipping_distance to grey
        bg_removed = depth_filter(image_gus, depth_image, clipping_distance_min, clipping_distance_max)

        # white_color = 255
        # depth_image_3d = np.dstack((depth_image_src,depth_image_src,depth_image_src)) #depth image is 1 channel, color is 3 channels
        # bg_removed = np.where((depth_image_3d > clipping_distance_max) | (depth_image_3d <= 0) | (depth_image_3d < clipping_distance_min), white_color, image_gus)

        # Transfer rgb to hsv
        image_hsv=cv2.cvtColor(image_gus, cv2.COLOR_BGR2HSV)

        # Generate mask ( The HSV value of red has two ranges)
        #mask = cv2.inRange(image_hsv,lower,upper)
        mask1 = cv2.inRange(image_hsv,color_dist['red1']['lower'],color_dist['red1']['upper'])
        mask2 = cv2.inRange(image_hsv,color_dist['red2']['lower'],color_dist['red2']['upper'])
        mask = mask1+mask2

        # Set kernel as 3*3
        kernel = np.ones((3,3),np.uint8)
        # Dilate image
        dilate_mask = cv2.dilate(mask, kernel, iterations=4)
        # Erode image
        closing_mask = cv2.erode(dilate_mask, kernel, iterations=3)

        # Find all the contours in the erode_mask
        contour, hierarchy = cv2.findContours(closing_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        digitCnts = []
        # Whether the contour exist
        if np.size(contour)>0:
            #area = cv2.contourArea(contours)
            contours_area = np.zeros((len(contour),))
            for i in range(len(contour)):
                area = cv2.contourArea(contour[i])
                
                if area < 200:
                    cv2.drawContours(closing_mask,[contour[i]],0,0,-1)

            # Try to reduce the effects of highlights and chromatic aberration(色差)
            # After filling small areas, make valid areas stable and connected
            kernel5 = np.ones((5,5),np.uint8)
            filter_mask = cv2.dilate(closing_mask, kernel5, iterations=5)

            contours_filter, hierarchy_filter = cv2.findContours(filter_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if np.size(contours_filter)>0:
                c = max(contours_filter, key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                #rect = cv2.boundingRect(c)
                box = cv2.boxPoints(rect)
                cv2.drawContours(imgResult, [np.int0(box)], -1, (0, 255, 255), 2)
                cv2.circle(imgResult, tuple(map(int,list(rect[0]))), 2, (255, 0, 0), 2)

                warped = four_point_transform(mask, np.int0(box).reshape(4, 2))
                imgResult = four_point_transform(color_image, np.int0(box).reshape(4, 2))
                # Set kernel as 3*3
                kernel = np.ones((3,3),np.uint8)

                # Erode image
                erode_warped = cv2.erode(warped, kernel, iterations=3)

                # Dilate image
                opening_warped = cv2.dilate(erode_warped, kernel, iterations=1)

                cnts = cv2.findContours(opening_warped.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                # 循环遍历所有的候选区域
                for c in cnts:
                    # 计算轮廓的边界框
                    (x, y, w, h) = cv2.boundingRect(c)
                    print(w,h)
                    # 如果当前的这个轮廓区域足够大，它一定是一个数字区域
                    # if w >= 15 and (h >= 30 and h <= 40):
                    #     digitCnts.append(c)
                    if w >= 15 and (h >= 30 and h <= 60):
                        digitCnts.append(c)
                
                if digitCnts:
                    # 从左到右对这些轮廓进行排序
                    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
                    digits = []

                    # boundingBoxes = [cv2.boundingRect(c) for c in digitCnts]
                    # print(digitCnts)
                    # print(boundingBoxes)
                    # (cnts, boundingBoxes) = zip(*sorted(zip(digitCnts, boundingBoxes),
                    #                                 key=lambda b: b[1][0], reverse=False))
                    # digitCnts = cnts
                    # digits = []

                    # 循环处理每一个数字
                    i = 0
                    for c in digitCnts:
                        # 获取ROI区域
                        (x, y, w, h) = cv2.boundingRect(c)
                        roi = opening_warped[y:y + h, x:x + w]
                        # 分别计算每一段的宽度和高度
                        (roiH, roiW) = roi.shape
                        (dW, dH) = (int(roiW * 0.3), int(roiH * 0.15))
                        dHC = int(roiH * 0.05)

                        # 定义一个7段数码管的集合
                        segments = [
                            ((0, 0), (w, dH)),	                         # 上
                            ((0, 0), (dW, h // 2)),                      # 左上
                            ((w - dW, 0), (w, h // 2)),	                 # 右上
                            ((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # 中间
                            ((0, h // 2), (dW, h)),	                     # 左下
                            ((w - dW, h // 2), (w, h)),	                 # 右下
                            ((0, h - dH), (w, h))	                     # 下
                        ]
                        on = [0] * len(segments)

                        # 循环遍历数码管中的每一段
                        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):  # 检测分割后的ROI区域，并统计分割图中的阈值像素点
                            segROI = roi[yA:yB, xA:xB]
                            total = cv2.countNonZero(segROI)
                            area = (xB - xA) * (yB - yA)

                            # 如果非零区域的个数大于整个区域的一半，则认为该段是亮的
                            #print(total,area)
                            if total> (0.5 * float(area)) :
                                on[i]= 1

                        print(on)
                        # 进行数字查询并显示结果
                        try:
                            digit = DIGITS_LOOKUP[tuple(on)]
                            digits.append(digit)
                            cv2.rectangle(imgResult, (x, y), (x + w, y + h), (0, 255, 0), 1)
                            cv2.putText(imgResult, str(digit), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                            # 显示最终的输出结果
                        except KeyError:
                            print("no detect")
                    print(digits)
                    print("######")


        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        imgStack = stackImages(0.7,([color_image, imgResult],[mask, opening_warped]))

        cv2.namedWindow('Depth Filter and Color locate', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Depth Filter and Color locate', imgStack)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
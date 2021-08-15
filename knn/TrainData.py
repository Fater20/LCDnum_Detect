import os
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

# Rename file names of train data
def rename(path):
    filelist = os.listdir(path)
    total_num = len(filelist)
    i = 0
    for item in filelist:
        if item.endswith('.jpg'):
            src = os.path.join(os.path.abspath(path), item)
            dst = os.path.join(os.path.abspath(path), str(i).zfill(5) + '.jpg')
            try:
                os.rename(src, dst)
                print ('将 %s 重命名为： %s ...' % (src, dst))
                i = i + 1
            except:
                continue
    print ('总共重命名 %d 个文件' % total_num)
 
if __name__ == '__main__':
    # Initial train data
    s = './train_class'             # The path of feature images
    num = 1000                      # Total number of samples (100 each for 0-9) 
    row = 130                       # Rows of feature image
    col = 72                        # Columns of feature image
    a = np.zeros((num,row,col))     # Matrix storing all samples

    for file_name in range(0,10):
        rename('D:/Program/cv/LCDnum_Detect/knn/num1/new/'+str(file_name))
    rename('D:/Program/cv/LCDnum_Detect/knn/num1/new/other')

    # Store image data
    n = 0
    for i in range(0,10):
        for j in range(0,100):
            a[n,:,:] = cv2.imread(s + '/' + str(i) + '_class/' + str(j).zfill(5) + '.jpg',0)
            n = n+1

    # Get the feature of sample
    feature = np.zeros((num,row,col))
    for ni in range(0,num):
        for nr in range(0,row):
            for nc in range(0,col):
                if a[ni,nr,nc] == 255:
                    feature[ni,nr,nc]=1

    # Process the feature as a single line
    train = feature[:,:].reshape(-1,row*col).astype(np.float32)

    # Get labels
    trainLabels = [int(i/(num/10)) for i in range(0,num)]
    trainLabels = np.asarray(trainLabels)

    # Build KNN classifier
    neigh =KNN(n_neighbors = 3, algorithm = 'auto')
    
    # Fit the model
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

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)
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
    cv2.createTrackbar("Dmin","Trackbars", 10,800,empty)
    cv2.createTrackbar("Dmax","Trackbars",200,800,empty)

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

            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not color_frame:
                continue

            # Transform depth_frame and color_frame to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Create some images to display
            imgResult = color_image.copy()
            filter_mask = color_image.copy()

            warped = color_image.copy()
            dilate_warped = color_image.copy()
            closing_warped = color_image.copy()
            # Create the empty color list
            color_list = list()

            # Transfer rgb to hsv
            image_hsv=cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            # Gaussian Blur
            image_gus = cv2.GaussianBlur(image_hsv, (5, 5), 0)

            # Generate mask ( The HSV value of red has two ranges)
            # mask = cv2.inRange(image_hsv,hsv_lower,hsv_upper)
            # mask = cv2.inRange(image_hsv, np.array([0, 52, 194]), np.array([52, 255, 255]))
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
                    if area < 70:
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
                    dilate_warped = cv2.dilate(warped, kernel, iterations=2)

                    # Erode image
                    closing_warped = cv2.erode(dilate_warped, kernel, iterations=3)

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
                            of = np.zeros((row,col))  # Store the feature of the image to be recognized
                            for nr in range(0,row):
                                for nc in range(0,col):
                                    if o[nr,nc] == 255:
                                        of[nr,nc]=1

                            test = of.reshape(-1,row*col).astype(np.float32)

                            # Get recognition result
                            classifierResult = neigh.predict(test)
                            distances, indices = neigh.kneighbors(test)

                            
                            print("距离当前点最近的3个邻居为:",trainLabels[indices])
                            print("距离为:",distances)
                            print("分类返回结果为%d" % (classifierResult))
                            

                            # Query and display the detecting result
                            try:
                                digits.append(int(classifierResult))
                                filelist = os.listdir('./num1/new/'+str(int(classifierResult)))
                                total_num = len(filelist)     # Get length (number) of file
                                cv2.imwrite('./knn/num1/new/'+str(int(classifierResult))+ '/' +str(total_num).zfill(5)+'.jpg',roi_re)
                                n = n + 1
                                cv2.rectangle(imgResult, (x, y), (x + w, y + h), (0, 255, 0), 1)
                                cv2.putText(imgResult, str(int(classifierResult)), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                            except KeyError:
                                print("no detect")
                        # Display the detecting result
                        print(digits)
                        print("######")


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
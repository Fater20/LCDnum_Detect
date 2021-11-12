# LCDnum_Detect
This project is about detecting LCD numbers(7-segment digital numbers).

Image Segmentation: Finding the contour of number area or Finding the color area of number

Detecting Method: Threading Method or KNN


## Requirements
* python3
* OpenCV4
* module "imutils"
* module "numpy"
* module "sklearn" (using KNN to detect may need this)

## Result Preview
<img src="https://github.com/Fater20/LCDnum_Detect/blob/main/image/KNN/result2.png" width="200" height="150" />

<img src="https://github.com/Fater20/LCDnum_Detect/blob/main/image/KNN/result1.png" width="400" height="300" />


## Introduction
"./ThreadMethod/LCDnumDetect_Color.py": Use Threading Method to detect number and split image by color

"./ThreadMethod/LCDnumDetect_Contour.py": Use Threading Method to detect number and split image by contour

"./KNN/LCDnumDetect_Color1.py": Use KNN from module "sklearn"

"./KNN/LCDnumDetect_Color2.py": Use KNN from module "cv2" (In order to improve the recognition speed, the features of samples are processed as 1/5 of the original)

"./KNN/TrainData.py": A program to collect train data

"./KNN/4-DigitDisplay/ ": A program for stm32 to drive 4-Digit LED Display

"./KNN/train_class.zip": The train data for KNN (Collect from 4-Digit LED Display)

## Attentions
* In terms of accuracy, KNN is better than Threading Method.
* The difficulty of this project lies in image segmentation and the methods which provide in this project may not the best way.
* In KNN method, using KNN functions of OpenCV has higher FPS than that of sklearn, because the feature size id compressed in that program.
* It is recommended to re-collect the training data according to the actual situation (It is easy to do that and the accuracy will improve).

## Reference
* https://www.cxyzjd.com/article/WZZ18191171661/90762434
* https://medium.com/@nikhilanandikam/handwritten-digit-recognition-hdr-using-k-nearest-neighbors-knn-f4c794a0282a
* https://blog.csdn.net/qq_41684249/article/details/104168367

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 14:09:15 2019

@author: Dell
"""

import cv2
import numpy as np
import operator
import os
from PIL import Image
import matplotlib.pyplot as plt

MIN_CONTOUR_AREA = 60

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


#################################################################################
allContoursWithData = []
validContoursWithData = []
class ContourWithData():

    # member variables ############################################################################
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        return True
    
####################################################################################

sign_cord = [[684, 846, 745, 946], [697, 805, 753, 867], [717, 830, 751, 919], [705, 821, 765, 900], [649, 641, 704, 739], [292, 390, 326, 442]]
img_src = 'E:/medtrail/OpenCV_3_KNN_Character_Recognition_Python/image_data/image'
a2 = []
crop_img = np.array([[]])


for n in range(1, 7):
    image = Image.open(img_src+str(n)+'.jpeg')
    #print (type(image))
    #image = np.asarray(image)
    a1 = image.crop((sign_cord[n-1][0], sign_cord[n-1][1], sign_cord[n-1][2], sign_cord[n-1][3]))
    #a1 = image[sign_cord[n-1][1]:sign_cord[n-1][3], sign_cord[n-1][0]:sign_cord[n-1][2]]
    
    a1 = a1.resize((80, 50))
    a1 = np.asarray(a1)
    a1 = cv2.cvtColor(a1, cv2.COLOR_BGR2GRAY)
    a1 = cv2.GaussianBlur(a1, (5,5), 0)
    a1 = cv2.adaptiveThreshold(a1,                           # input image
                               255,                                  # make pixels that pass the threshold full white
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                               cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                               11,                                   # size of a pixel neighborhood used to calculate threshold value
                               2)
    #print (a1[0].shape)
    #plt.imshow(np.array(a1))
    #plt.show()
    #print (a1.shape)
    a2 = a1.reshape(1, -1)
    #print (a2)
    crop_img = np.append(crop_img, a2)
#     plt.imshow(np.array(a1))
#     plt.show()
crop_img = crop_img.reshape(-1, 4000)

    #crop_img.concatenate(a2)


crop_img = np.array(crop_img)


doctors = [151, 151, 151, 151, 151, 151]
doctors = np.array(doctors)

doctors = doctors.astype(float)

doctors = doctors.reshape(-1, 1)

doctors = np.float32(doctors)
crop_img = np.float32(crop_img)

a1Copy = a1.copy()
signContours, signHierarchy = cv2.findContours(a1Copy,        # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                               cv2.RETR_EXTERNAL,                 # retrieve the outermost contours only
                                               cv2.CHAIN_APPROX_SIMPLE)

for signContour in signContours:                          # for each contour
    if cv2.contourArea(signContour) > MIN_CONTOUR_AREA:          # if contour is big enough to consider
        [intX, intY, intW, intH] = cv2.boundingRect(signContour)         # get and break out bounding rect

                                                # draw rectangle around each contour as we ask user for input
        cv2.rectangle(a1,           # draw rectangle on original training image
                      (intX, intY),                 # upper left corner
                      (intX+intW,intY+intH),        # lower right corner
                      (0, 0, 255),                  # red
                      2)                            # thickness
        
#plt.imshow(np.array(a1))
#plt.show()




kNearest = cv2.ml.KNearest_create()

kNearest.train(crop_img, cv2.ml.ROW_SAMPLE, doctors)


##########################################################################
img_src = "E:/medtrail/OpenCV_3_KNN_Character_Recognition_Python/image_data/"

file = 'image1.jpeg'
print (file)

image = Image.open(img_src+file)
image_np = np.asarray(image)
print (image_np.shape)

crop_doc = image.crop((image_np.shape[1]//2, image_np.shape[0]//2, image_np.shape[1], image_np.shape[0]))
#crop_doc = image_np[image_np.shape[0]//2:image_np.shape[0], image_np.shape[1]//2:image_np.shape[1]]
plt.figure(figsize=(40, 20))
plt.imshow(np.array(crop_doc))
plt.show()
crop_doc1 = np.asarray(crop_doc)

crop_doc_np = cv2.cvtColor(crop_doc1, cv2.COLOR_BGR2GRAY)
crop_doc_np = cv2.GaussianBlur(crop_doc_np, (5,5), 0)
crop_doc_np = cv2.adaptiveThreshold(crop_doc_np,                           # input image
                           255,                                  # make pixels that pass the threshold full white
                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                           cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                           11,                                   # size of a pixel neighborhood used to calculate threshold value
                           2)

crop_docCopy = crop_doc_np.copy()
npaContours, npaHierarchy = cv2.findContours(crop_docCopy,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                             cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                             cv2.CHAIN_APPROX_SIMPLE)
for npaContour in npaContours:                             # for each contour
    contourWithData = ContourWithData()                                             # instantiate a contour with data object
    contourWithData.npaContour = npaContour                                         # assign contour to contour with data
    contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
    contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
    contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
    allContoursWithData.append(contourWithData)
    
for contourWithData in allContoursWithData:                 # for all contours
    if contourWithData.checkIfContourIsValid():             # check if valid
        validContoursWithData.append(contourWithData)
        
validContoursWithData.sort(key = operator.attrgetter("intRectX"))

print (len(validContoursWithData))

for contourWithData in validContoursWithData:            # for each contour
                                            # draw a green rect around the current char
    cv2.rectangle(crop_doc1,                                        # draw rectangle on original testing image
                  (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                  (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                  (0, 255, 0),              # green
                  2)                        # thickness

    imgROI = crop_doc_np[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                       contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

    imgROIResized = cv2.resize(imgROI, (80, 50))             # resize image, this will be more consistent for recognition and storage

    npaROIResized = imgROIResized.reshape((1, 80*50))      # flatten image into 1d numpy array

    npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

    res = cv2.matchTemplate(npaROIResized, crop_img, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    
    for pt in zip(*loc[::-1]):
        cv2.rectangle(npaROIResized, pt, (pt[0]+80, pt[0]+50), (0, 255, 255), 2)
        
        
print (imgROIResized.shape)
plt.imshow(np.array(imgROIResized))
plt.show()
    #retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)     # call KNN function find_nearest


crop_doc_np = crop_doc_np.astype(float)
crop_doc_np = np.float32(crop_doc_np)

retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)
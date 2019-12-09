# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:40:56 2019

@author: Dell
"""

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import operator

MIN_CONTOUR_AREA = 60

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###############################################################################
# Step 1: Contour validations

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
    
###############################################################################
# Step 2: Training the Signatures
sign_cord = [[684, 846, 745, 946], [697, 805, 753, 867], [717, 830, 751, 919], [705, 821, 765, 900], [649, 641, 704, 739], [292, 390, 326, 442]]
img_src = 'E:/medtrail/OpenCV_3_KNN_Character_Recognition_Python/image_data/image'
a2 = []
crop_img = np.array([[]])

image = Image.open(img_src+'1.jpeg')
print (np.asarray(image).shape)
#image = np.asarray(image)
a1 = image.crop((684, 846, 745, 946))

z = a1.save('signature.jpeg')

a1 = np.asarray(a1)
plt.imshow(np.array(a1))
plt.show()

a1 = cv2.cvtColor(a1, cv2.COLOR_BGR2GRAY)
a1 = cv2.GaussianBlur(a1, (5,5), 0)
a1 = cv2.adaptiveThreshold(a1,                           # input image
                           255,                                  # make pixels that pass the threshold full white
                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                           cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                           11,                                   # size of a pixel neighborhood used to calculate threshold value
                           2)

###############################################################################
# Step 3: Getting the feature detector
MIN_MATCHES = 8

homography = None

# ORB keypoint detector
orb = cv2.ORB_create()

# create brute force  matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Compute model keypoints and its descriptors
kp_model, des_model = orb.detectAndCompute(a1, None)
print (des_model)

###############################################################################
# Step 4: Working with prescription

img_src = "E:/medtrail/OpenCV_3_KNN_Character_Recognition_Python/image_data/"

file = 'image1.jpeg'
print (file)

image = Image.open(img_src+file)
image_np = np.asarray(image)
print (image_np.shape)

crop_doc = image.crop((image_np.shape[1]//2, image_np.shape[0]//2, image_np.shape[1], image_np.shape[0]))
#crop_doc = image_np[image_np.shape[0]//2:image_np.shape[0], image_np.shape[1]//2:image_np.shape[1]]
z1 = crop_doc.save('cropped_sign.jpeg')
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

    ###############################################################################
    # Step 4: Extracting features    
    
    
    

imgd = Image.fromarray(crop_doc1, 'RGB')
z3 = imgd.save('contours.jpeg')
plt.imshow(np.array(crop_doc1))
plt.show()




# -*- coding: utf-8 -*-
# Credits: Shubham Jaiswal
# This is not my code!

#import necessary packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import operator
#read the query and the template image in gray_sacle
ref_img = cv2.imread("E:/medtrail/OpenCV_3_KNN_Character_Recognition_Python/image_data/image3.jpeg",0)
template_img = cv2.imread("signature_new.jpeg",0)
w, h = template_img.shape[::-1]
#the methods to be used for template matching
meth = 'cv2.TM_CCOEFF_NORMED'
#set a threshold to qualify for a match
threshold = 0.85
a1 = cv2.GaussianBlur(template_img, (5,5), 0)
a1 = cv2.adaptiveThreshold(a1,                           # input image
                           255,                                  # make pixels that pass the threshold full white
                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                           cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                           11,                                   # size of a pixel neighborhood used to calculate threshold value
                           2)
#loop overboth the methods and do template matching
#plot the results if the threshold is qualified

MIN_CONTOUR_AREA = 60
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


img_src = "E:/medtrail/OpenCV_3_KNN_Character_Recognition_Python/image_data/"

file = 'image1.jpeg'
print (file)

#image = Image.open(img_src+file)
#image_np = np.asarray(ref_img)
#print (image_np.shape)
image = Image.fromarray(ref_img)
image_np = np.asarray(image)

crop_doc = image.crop((image_np.shape[1]//2, image_np.shape[0]//2, image_np.shape[1], image_np.shape[0]))
#crop_doc = image_np[image_np.shape[0]//2:image_np.shape[0], image_np.shape[1]//2:image_np.shape[1]]
z1 = crop_doc.save('cropped_sign.jpeg')
plt.figure(figsize=(40, 20))
plt.imshow(np.array(crop_doc))
plt.show()

crop_doc1 = np.asarray(crop_doc)

#crop_doc_np = cv2.cvtColor(crop_doc1, cv2.COLOR_BGR2GRAY)
crop_doc_np = cv2.GaussianBlur(crop_doc1, (5,5), 0)
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
                        # thickness

    imgROI = crop_doc_np[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                       contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]


    img = imgROI.copy()

    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(a1, img,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    if max_val >= threshold:
        cv2.rectangle(crop_doc1,                                        # draw rectangle on original testing image
              (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
              (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
              (0, 255, 0),              # green
              2)
        print(min_val,max_val)
        cv2.rectangle(img,top_left, bottom_right, 0, 2)

        plt.figure(figsize=(40, 20))
        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        plt.show()

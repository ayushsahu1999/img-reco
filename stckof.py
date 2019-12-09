# -*- coding: utf-8 -*-
# Credits: Shubham Jaiswal
# This is not my code!

#import necessary packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
#read the query and the template image in gray_sacle
ref_img = cv2.imread("cropped_sign.jpeg",0)
template_img = cv2.imread("signature.jpeg",0)
w, h = template_img.shape[::-1]
#the methods to be used for template matching
methods = ['cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR_NORMED']
#set a threshold to qualify for a match
threshold = 0.9
#loop overboth the methods and do template matching
#plot the results if the threshold is qualified
for meth in methods:
    img = ref_img.copy()

    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template_img,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    if max_val >= threshold:
        print(min_val,max_val)
        cv2.rectangle(img,top_left, bottom_right, 0, 2)

        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        plt.show()

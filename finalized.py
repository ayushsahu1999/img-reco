import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import operator
import xmltodict,json,os

doctors = []

f = open('demo5_test.txt')
text = f.read()
f.close()

infos = text.split('\n')

template_imgs = []
for info in infos:
    if (info == ''):
        continue
    a1 = info.split(' ')
    path = a1[0]
    docs = a1[1]
    a2 = docs.split(',')
    name = a2[0]
    xmin = int(a2[1])
    ymin = int(a2[2])
    xmax = int(a2[3])
    ymax = int(a2[4])
        
    if (name == 'sp_agarwal'):
        doc = Image.open(path)
        doc = np.asarray(doc)
        template_img = doc[ymin:ymax, xmin:xmax]
        template_img = cv2.fastNlMeansDenoisingColored(template_img,None,10,10,7,21)
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        template_img = cv2.GaussianBlur(template_img, (5,5), 0)
        template_img = cv2.adaptiveThreshold(template_img,                          
                                   255,                                  
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       
                                   cv2.THRESH_BINARY_INV,                
                                   11,
                                   2)
        for i in range(template_img.shape[0]):
            for j in range(template_img.shape[1]):
        #         print(i,j)
                if(template_img[i][j]<100):
                    template_img[i][j] = 0
                else:
                    template_img[i][j] = 255

        template_imgs.append(template_img)
        doctors.append('sp_agarwal')
        
    elif (name == 'gourav_gupta'):
        doc = Image.open(path)
        doc = np.asarray(doc)
        template_img = doc[ymin:ymax, xmin:xmax]
        template_img = cv2.fastNlMeansDenoisingColored(template_img,None,10,10,7,21)
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        template_img = cv2.GaussianBlur(template_img, (5,5), 0)
        template_img = cv2.adaptiveThreshold(template_img,                          
                                   255,                                  
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       
                                   cv2.THRESH_BINARY_INV,                
                                   11,
                                   2)
        for i in range(template_img.shape[0]):
            for j in range(template_img.shape[1]):
        #         print(i,j)
                if(template_img[i][j]<100):
                    template_img[i][j] = 0
                else:
                    template_img[i][j] = 255
        template_imgs.append(template_img)
        doctors.append('gourav_gupta')
        
    elif (name == 'gourav_thakral'):
        doc = Image.open(path)
        doc = np.asarray(doc)
        template_img = doc[ymin:ymax, xmin:xmax]
        template_img = cv2.fastNlMeansDenoisingColored(template_img,None,10,10,7,21)
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        template_img = cv2.GaussianBlur(template_img, (5,5), 0)
        template_img = cv2.adaptiveThreshold(template_img,                          
                                   255,                                  
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       
                                   cv2.THRESH_BINARY_INV,                
                                   11,
                                   2)
        for i in range(template_img.shape[0]):
            for j in range(template_img.shape[1]):
        #         print(i,j)
                if(template_img[i][j]<100):
                    template_img[i][j] = 0
                else:
                    template_img[i][j] = 255
        template_imgs.append(template_img)
        doctors.append('gourav_thakral')

    elif (name == 'kanika_gera'):
        doc = Image.open(path)
        doc = np.asarray(doc)
        template_img = doc[ymin:ymax, xmin:xmax]
        template_img = cv2.fastNlMeansDenoisingColored(template_img,None,10,10,7,21)
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        template_img = cv2.GaussianBlur(template_img, (5,5), 0)
        template_img = cv2.adaptiveThreshold(template_img,                          
                                   255,                                  
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       
                                   cv2.THRESH_BINARY_INV,                
                                   11,
                                   2)
        for i in range(template_img.shape[0]):
            for j in range(template_img.shape[1]):
        #         print(i,j)
                if(template_img[i][j]<100):
                    template_img[i][j] = 0
                else:
                    template_img[i][j] = 255
        template_imgs.append(template_img)
        doctors.append('kanika_gera')
        
# doc["annotation"]["object"]
data_name = "blessings"
src = data_name+"/signature_data/"

for file in os.listdir(src):
    
    if file.endswith(".xml"):
        with open(src+file) as fd:
            doc = json.loads(json.dumps(xmltodict.parse(fd.read())))
#             pprint.pprint(doc)
        
        line = data_name+"/image_data/"+doc["annotation"]["filename"]
        refimg = cv2.imread(line)
        ref_img1 = cv2.cvtColor(refimg, cv2.COLOR_BGR2GRAY)
        temp_img = ref_img1.copy()
        ref_imgs = [ref_img1.shape[0]//2, ref_img1.shape[1]//2, ref_img1.shape[0], ref_img1.shape[1]]

        min_locs_x = []
        min_locs_y = []
        max_locs_x = []
        max_locs_y = []
        max_vals = []
        parts = []
        docs = []
        abc = 0
        number = 0

        for i in range(0, 4):
            if (i == 0):
                ref_imgp = ref_img1[ref_imgs[0]:ref_imgs[2], ref_imgs[1]:ref_imgs[3]]
                refer1 = ref_imgp.copy()
            elif (i == 1):
                ref_imgp = ref_img1[ref_imgs[0]:ref_imgs[2], 0:ref_imgs[1]]
                refer2 = ref_imgp.copy()
            elif (i == 2):
                ref_imgp = ref_img1[0:ref_imgs[0], ref_imgs[1]:ref_imgs[3]]
                refer3 = ref_imgp.copy()
            elif (i == 3):
                ref_imgp = ref_img1[0:ref_imgs[0], 0:ref_imgs[1]]
                refer4 = ref_imgp.copy()
            else:
                break
            ref_img = cv2.GaussianBlur(ref_imgp, (5,5), 0)
            ref_img = cv2.adaptiveThreshold(ref_img,                           # input image
                                       255,                                  # make pixels that pass the threshold full white
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                       cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                       11,                                   # size of a pixel neighborhood used to calculate threshold value
                                       2)

            index = 0
            for template_img in template_imgs:

                w, h = template_img.shape[::-1]
                #the methods to be used for template matching
                # meth = 'cv2.TM_CCOEFF_NORMED'
                #set a threshold to qualify for a match
                threshold = 0.90

            #     plt.imshow(template_img)
            #     plt.show()

                #the methods to be used for template matching
                methods = ['cv2.TM_CCORR_NORMED']
                #set a threshold to qualify for a match

                #loop overboth the methods and do template matching
                #plot the results if the threshold is qualified
                for meth in methods:
                    img = ref_img.copy()
                    img1 = ref_img1.copy()

                    method = eval(meth)

                    # Apply template Matching
                    res = cv2.matchTemplate(img,template_img,method)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                    top_left = max_loc
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    #print (max_val)
                    if max_val >= threshold:
                        number = number + 1
                        abc = 999
                        print (meth)
                        print ('image: '+doc["annotation"]["filename"])
                        print ('Doctor: '+doctors[index])
                        
                        print(min_val,max_val)

                        if (i == 0):
                            min_loc_x = top_left[0] + ref_imgs[1]
                            min_loc_y = top_left[1] + ref_imgs[0]
                            max_loc_x = bottom_right[0] + ref_imgs[1]
                            max_loc_y = bottom_right[1] + ref_imgs[0]

                        elif (i == 1):
                            min_loc_x = top_left[0]
                            min_loc_y = top_left[1] + ref_imgs[0]
                            max_loc_x = bottom_right[0]
                            max_loc_y = bottom_right[1] + ref_imgs[0]

                        elif (i == 2):
                            min_loc_x = top_left[0]
                            min_loc_y = top_left[1] + ref_imgs[0]
                            max_loc_x = bottom_right[0]
                            max_loc_y = bottom_right[1] + ref_imgs[0]

                        elif (i == 3):
                            min_loc_x = top_left[0] + ref_imgs[1]
                            min_loc_y = top_left[1]
                            max_loc_x = bottom_right[0] + ref_imgs[1]
                            max_loc_y = bottom_right[1]
                        cv2.rectangle(refimg,(min_loc_x, min_loc_y), (max_loc_x, max_loc_y), 0, 2)
                        plt.figure(figsize=(40,20))
                        plt.imshow(refimg)
                        plt.show()
                    else:
                        max_vals.append(max_val)
                        min_locs_x.append(top_left[0])
                        min_locs_y.append(top_left[1])
                        max_locs_x.append(bottom_right[0])
                        max_locs_y.append(bottom_right[1])
                        parts.append(i)
                        docs.append(doctors[index])
                index = index + 1


        print ('Number = ' + str(number))                
        if not(abc == 999):
            max1 = max_vals.copy()
            max1.sort(reverse=True)
            for i in range(0, len(max_vals)):
                if (max_vals[i] == max1[0]):
                    z = parts[i]
                    if (z == 0):
                        print ('ARRT: '+str(z))
            #                 min_loc_x = min_locs_x[i]
            #                 min_loc_y = min_locs_y[i]
            #                 max_loc_x = max_locs_x[i]
            #                 max_loc_y = max_locs_y[i]
            #                 print ('Max: '+str(max(max_vals)))
            #                 cv2.rectangle(refer1,(min_loc_x, min_loc_y), (max_loc_x, max_loc_y), 0, 2)
            #     #             plt.figure(figsize=(40,20))
            #                 plt.figure(figsize=(40,20))
            #                 plt.imshow(refer1)
            #                 plt.show()
                        min_loc_x = min_locs_x[i] + ref_imgs[1]
                        min_loc_y = min_locs_y[i] + ref_imgs[0]
                        max_loc_x = max_locs_x[i] + ref_imgs[1]
                        max_loc_y = max_locs_y[i] + ref_imgs[0]
                    elif (z == 1):
                        print ('ARRT: '+str(z))
            #                 min_loc_x = min_locs_x[i]
            #                 min_loc_y = min_locs_y[i]
            #                 max_loc_x = max_locs_x[i]
            #                 max_loc_y = max_locs_y[i]
            #                 print ('Max: '+str(max(max_vals)))
            #                 cv2.rectangle(refer2,(min_loc_x, min_loc_y), (max_loc_x, max_loc_y), 0, 2)
            #     #             plt.figure(figsize=(40,20))
            #                 plt.figure(figsize=(40,20))
            #                 plt.imshow(refer2)
            #                 plt.show()
                        min_loc_x = min_locs_x[i]
                        min_loc_y = min_locs_y[i] + ref_imgs[0]
                        max_loc_x = max_locs_x[i]
                        max_loc_y = max_locs_y[i] + ref_imgs[0]
                    elif (z == 2):
                        print ('ARRT: '+str(z))
                        min_loc_x = min_locs_x[i] + ref_imgs[1]
                        min_loc_y = min_locs_y[i]
                        max_loc_x = max_locs_x[i] + ref_imgs[1]
                        max_loc_y = max_locs_y[i]
            #                 print ('Max: '+str(max(max_vals)))
            #                 cv2.rectangle(refer3,(min_loc_x, min_loc_y), (max_loc_x, max_loc_y), 0, 2)
            #     #             plt.figure(figsize=(40,20))
            #                 plt.figure(figsize=(40,20))
            #                 plt.imshow(refer3)
            #                 plt.show()
            #                 min_locs_x[i] = min_locs_x[i] + ref_imgs[1]//2
            #                 min_locs_y[i] = min_locs_y[i] + ref_imgs[0]
            #                 max_locs_x[i] = max_locs_x[i] + ref_imgs[1]//2
            #                 max_locs_y[i] = max_locs_y[i] + ref_imgs[0]
                    else:
                        print ('ARRT: '+str(z))
                        min_loc_x = min_locs_x[i]
                        min_loc_y = min_locs_y[i]
                        max_loc_x = max_locs_x[i]
                        max_loc_y = max_locs_y[i]
            #                 print ('Max: '+str(max(max_vals)))
            #                 cv2.rectangle(refer4,(min_loc_x, min_loc_y), (max_loc_x, max_loc_y), 0, 2)
            #     #             plt.figure(figsize=(40,20))
            #                 plt.figure(figsize=(40,20))
            #                 plt.imshow(refer4)
            #                 plt.show()
            #             min_loc_x = min_locs_x[i]
            #             min_loc_y = min_locs_y[i]
            #             max_loc_x = max_locs_x[i]
            #             max_loc_y = max_locs_y[i]
                    print ('image: '+doc["annotation"]["filename"])
                    print ('Doctor: '+docs[i])
                    print ('Max: '+str(max_vals[i]))
                    cv2.rectangle(refimg,(min_loc_x, min_loc_y), (max_loc_x, max_loc_y), 0, 2)
        #             plt.figure(figsize=(40,20))
            plt.figure(figsize=(40,20))
            plt.imshow(refimg)
            plt.show()

# Series of helper functions to remove Background from Product Images

import cv2
import numpy as np
import math

def constant_background(img):

    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    h,w = gray.shape
    top_left = gray[0:2,0:2]
    bottom_left = gray[h-2:h,0:2]
    top_right = gray[0:2,w-2:w]
    bottom_right = gray[h-2:h,w-2:w]
    a = np.array([top_left,bottom_left,top_right,bottom_right])

    return np.std(a), np.mean(a)


def RemoveBackground(img, edge_threshold):
    """
    Removes Background from Product Images
    img: color image
    edge_threshold: threshold parameter for Canny edge algorithm
    """
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    threshold_check = 0.20
    cont_back, mean_back = constant_background(img)

    #find edges
    if cont_back <= 1.0:
        #very constant background
        edges = cv2.Canny(gray,40,200,apertureSize=5)
        min_cut = mean_back-1
        max_cut = min(mean_back+1,255)
    else:
        #varying background
        edges = cv2.Canny(gray,edge_threshold,200)
        min_cut = max(mean_back-2,0)
        max_cut = min(mean_back+2,255)

    #Gaussian blur
    blur = cv2.GaussianBlur(edges,(3,3),2)

    #threshold
    _,thresh = cv2.threshold(blur,10,255,cv2.THRESH_BINARY)
    adp_thresh = thresh
    h,w = adp_thresh.shape[:2]

    # masking: Operation mask should be a single-channel 8-bit image,
    # 2 pixels wider and 2 pixels taller than image
    mask = np.zeros((h+2,w+2),np.uint8)
    #color filling. filled pixels are set to 125. NOTE: cv2 reads coordinates in
    #plot system of reference. This is, x runs horizontally to the right and y
    #vertically dow the image. However numpy interprets the image inverted. This
    #is, x runs vertically down the image while y runs horizontally to the
    #right.
    if np.mean(thresh[:5,:5]) == 0:                   #upper left box
        cv2.floodFill(adp_thresh,mask,(0,0),(125))    #upper left point entry

    if np.mean(thresh[h-6:h-1,:5]) == 0:              #lower left box
        cv2.floodFill(adp_thresh,mask,(0,h-1),(125))  #lower left point entry

    if np.mean(thresh[h-6:h-1,w-6:w-1]) == 0:         # lower right box
        cv2.floodFill(adp_thresh,mask,(w-1,h-1),(125))# lower right point entry

    if  np.mean(thresh[:5,w-6:w-1]) == 0:             # upper right box
        cv2.floodFill(adp_thresh,mask,(w-1,0),(125))  # upper right point entry

    mask  = 255 - cv2.inRange(adp_thresh,125,125)    # 255 where adp_thresh == 125 --> 0 where 255
    mask2 = 255 - cv2.inRange(gray, min_cut,max_cut) # 0 where grey in [min_cut, max_cut]
    mm = mask * mask2 # flag background "trapped" within the object and not accessed by floodFill

    # if more than 80% is removed, chances are that floodfill failed
    # in unit8 255*255 = 1
    if (len(np.where(mm==1)[0])/float(h*w)) < threshold_check:
        #print "did not passed the 20% non-background threshold"
        return (255-cv2.inRange(gray, 254,255))/255

    mm = mm*255 #back to 255 scale
    g_mask = np.zeros(mm.shape[:2],np.uint8)
    # mask to be passed to grabCut:
    # 0 - absolutely backgroung
    # 1 - absolutely foreground
    # 2 - probably background
    # 3 - probably foreground
    g_mask = np.where((mm==255),3,2).astype('uint8')

    bgdModel = np.zeros((1,65),np.float64) # default
    fgdModel = np.zeros((1,65),np.float64) # default
    cv2.grabCut(img,g_mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    g_mask = np.where((g_mask==2)|(g_mask==0),0,1).astype('uint8') # if 2 or 0 -> 0 (background)

    if (len(np.where(g_mask==1)[0])/float(h*w)) < threshold_check:
        #print "did not passed the 20% non-background threshold"
        return (255-cv2.inRange(gray, 254,255))/255
    else:
        return g_mask
        # or
        # result = cv2.bitwise_and(images[2675], images[2675], mask = iim)
        # result[result == 0] = 255


def RemoveSole(img):
    """
    Function specifically implemented for shoes. Removes sole from shoe
    images. Useful when finding the dominant color
    """

    back_mask = RemoveBackground(img, 60)
    (cnts, _) = cv2.findContours(back_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in cnts]
    max_index = np.argmax(areas)
    cnt=cnts[max_index]
    (x, y, w, h) = cv2.boundingRect(cnt)
    back_roi = back_mask[y:y + h, x:x + w]
    img_roi = img[y:y + h, x:x + w]

    num = 20
    dy = w/num
    tuple_list = []
    ave_xr_xl = []
    # calculating heights of the image accross (numpy) x axis
    for i in xrange(num-1):
        xl = np.min(np.where(back_roi[:,dy*i:dy*(i+1)]==1)[0])
        xr = np.max(np.where(back_roi[:,dy*i:dy*(i+1)]==1)[0])
        ave_xr_xl.append(xr-xl)
        tuple_list.append((dy*i, xl ,xr))

    # Standard deviation of heights
    std =  np.std(ave_xr_xl)
    # Default cut pixel size. Used within the GrabCut algorithm
    slice_size = int(min(ave_xr_xl)*0.2)

    g_mask = np.zeros(back_roi.shape[:2],np.uint8)
    g_mask = np.where((back_roi==1),3,2).astype('uint8')

    tuple_list.sort(key=lambda x: x[1], reverse=True)
    benchmarkx = tuple_list[0][1]

    for t_list in tuple_list:
        yy,xxl,xxr = t_list

        if xxl == benchmarkx:
            g_mask[xxr-slice_size:xxr, yy:yy+dy] = 0
        else:
            ratio = (benchmarkx-xxl)/float(benchmarkx)
            # increase_by is defined based on whether the shoe is relatively
            # flat. If the shoe is flat, the stdev will be low.
            if std > 28:
                increase_by = int(slice_size * (1.+ratio)**2)
                g_mask[xxr-increase_by:xxr, yy:yy+dy] = 0
            else:
                increase_by = int(slice_size * (1.+ratio))
                g_mask[xxr-increase_by:xxr, yy:yy+dy] = 0

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(img_roi,g_mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    g_mask = np.where((g_mask==2)|(g_mask==0),0,1).astype('uint8') # 1-product
    masked = cv2.bitwise_and(img_roi,img_roi,mask = g_mask)
    return masked




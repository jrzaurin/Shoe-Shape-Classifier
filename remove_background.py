import cv2
import numpy as np
import math

def background_stats(img):
    """
    Simple function to compute the mean and stdev of the background in the four corners
    of the image
    """

    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    h,w = gray.shape
    top_left = gray[0:2,0:2]
    bottom_left = gray[h-2:h,0:2]
    top_right = gray[0:2,w-2:w]
    bottom_right = gray[h-2:h,w-2:w]
    bckg = np.array([top_left,bottom_left,top_right,bottom_right])

    return np.mean(bckg), np.std(bckg)


def RemoveBackground(img, edge_threshold, threshold_check = 0.10):
    """
    Removes Background from Product Images

    Params:
    --------
    img: color image
    edge_threshold : threshold parameter for Canny edge algorithm
    threshold_check: empirically adjusted. How much of your image is allowed to be
                     identified as  background before considering that the process
                     has failed, removing part of the image

    Returns:
    g_mask: mask based on color filling and the grabCut algorithm
    """

    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mean_bckg, std_bckg = background_stats(img)
    #find edges
    if std_bckg <= 1.0:
        #very constant background
        edges = cv2.Canny(gray,40,200,apertureSize=5)
        min_cut = mean_bckg-1
        max_cut = min(mean_bckg+1,255)
    else:
        #varying background
        edges = cv2.Canny(gray,edge_threshold,200)
        min_cut = max(mean_bckg-2,0) # rare ocasions one finds has black background
        max_cut = min(mean_bckg+2,255)

    #Gaussian blur
    blur = cv2.GaussianBlur(edges,(3,3),2)

    #threshold
    _,thresh = cv2.threshold(blur,10,255,cv2.THRESH_BINARY)
    adp_thresh = thresh
    h,w = adp_thresh.shape[:2]

    # masking: Operation mask should be a single-channel 8-bit image,
    # 2 pixels wider and 2 pixels taller than image
    mask = np.zeros((h+2,w+2),np.uint8)

    # color filling: filled pixels are set to 125.
    # NOTE: cv2 reads coordinates in plot system of reference. This is, x runs
    # horizontally to the right and y vertically dow the image. However numpy
    # interprets the image inverted. This is, x runs vertically down the image
    # while y runs horizontally to the right.
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

    mm = mm*255 #back to 255 scale for convenience
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



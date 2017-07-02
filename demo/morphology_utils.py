import numpy as np
import pandas as pd
import cv2
import pylab as pyl
import os
import itertools
from matplotlib.pyplot import ion

# ion()
def plot_image(imname):
    pyl.imshow(imname)
    pyl.show()

def threshold_value(img):
    """
    Returns a threshold value (0.9 or 0.98) based on whether any slice
    of the image within a central box is enterely white (white is a bitch!)
    0.9 or 0.98 come simply from a lot of experimentation.
    """

    is_color = len(img.shape) == 3
    is_grey  = len(img.shape) == 2

    if is_color:
        gray =  cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
    elif is_grey:
        gray = img.copy()

    slices = gray.mean(axis = 1)[20:gray.shape[0]-30]
    is_white = any(x > 0.9*255 for x in slices)
    if is_white:
        return 0.98
    else:
        return 0.9

def threshold_img(img):
    """
    Simple wrap-up function for cv2.threshold()
    """

    is_color = len(img.shape) == 3
    is_grey  = len(img.shape) == 2

    t = threshold_value(img)

    if is_color:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    elif is_grey:
        gray = img.copy()

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    (_, thresh) = cv2.threshold(blurred, t*255, 1, cv2.THRESH_BINARY_INV)

    return thresh


def get_edges(arr, thresh):
    """
    Given an array returns the min/max where that array is less that 255*thresh
    i.e. is not white. If all the slice array is white, returns the middle point.
    """

    if np.any(np.where(arr < thresh*255)[0]):
        e1 = min(np.where(arr < thresh*255)[0])
        e2 = max(np.where(arr < thresh*255)[0])

    else:
        e1 = len(arr)/2
        e2 = len(arr)/2

    return e1,e2


def bounding_box(img):
    """
    Returns right, left, lower and upper limits for the limiting box enclosing
    the item (shoe, dress). Note that given the shapes and colors of some items,
    finding the contours and compute the bounding box is not a viable solution.
    """

    is_color = len(img.shape) == 3
    is_grey  = len(img.shape) == 2

    if is_color:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    elif is_grey:
        gray = img.copy()

    slices = gray.mean(axis = 1)[20:gray.shape[0]-30]
    is_white = any(x > 0.9*255 for x in slices)

    if (is_white):
        h1 = min(np.apply_along_axis(get_edges, axis=0, arr=gray , thresh = 0.98)[0,:])
        h2 = max(np.apply_along_axis(get_edges, axis=0, arr=gray , thresh = 0.98)[1,:])
        w1 = min(np.apply_along_axis(get_edges, axis=1, arr=gray , thresh = 0.98)[:,0])
        w2 = max(np.apply_along_axis(get_edges, axis=1, arr=gray , thresh = 0.98)[:,1])
    else :
        h1 = min(np.apply_along_axis(get_edges, axis=0, arr=gray , thresh = 0.9)[0,:])
        h2 = max(np.apply_along_axis(get_edges, axis=0, arr=gray , thresh = 0.9)[1,:])
        w1 = min(np.apply_along_axis(get_edges, axis=1, arr=gray , thresh = 0.9)[:,0])
        w2 = max(np.apply_along_axis(get_edges, axis=1, arr=gray , thresh = 0.9)[:,1])

    return w1, w2, h1, h2


def shape_df(img, axis, nsteps):
    """
    Returns a data frame with the initial and end points enclosing the product
    in the image, across the x/y axis. Why a dataframe and not tuples? just for
    convenience.
    """

    is_color = len(img.shape) == 3
    is_grey  = len(img.shape) == 2

    if is_color:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    elif is_grey:
        gray = img.copy()

    edges = bounding_box(gray)
    gray_c = gray[edges[2]:edges[3]+1, edges[0]:edges[1]+1]
    thr = threshold_value(gray_c)

    if axis == 'x' :
        cuts = np.rint(np.linspace(5, gray_c.shape[1]-1, nsteps, endpoint=True)).astype(int)

        init = np.apply_along_axis(get_edges, 0, arr = gray_c, thresh = thr)[0,:][cuts]
        end  = np.apply_along_axis(get_edges, 0, arr = gray_c, thresh = thr)[1,:][cuts]

        df = pd.DataFrame(data = {'coord' : cuts, 'init' : init, 'end' : end},
                          columns=['coord', 'init', 'end'])

    elif axis == 'y':
        cuts = np.round(np.linspace(4, gray_c.shape[0]-1, nsteps, endpoint=True)).astype(int)

        init = np.apply_along_axis(get_edges, 1, arr = gray_c, thresh = thr)[:,0][cuts]
        end  = np.apply_along_axis(get_edges, 1, arr = gray_c, thresh = thr)[:,1][cuts]

        df = pd.DataFrame(data = {'coord' : cuts, 'init' : init, 'end' : end},
                          columns=['coord', 'init', 'end'])

    return df


def shape_points(img, nsteps, mirrow=False, only_upper=False):
    """
    Simple formatting the shape_df output to be passed to the ShapeContext class
    """

    if mirrow:
        im =  cv2.flip(img, 2)
    else:
        im = img.copy()

    df_y = shape_df(im, 'y', nsteps)
    df_x = shape_df(im, 'x', nsteps)

    if (not df_y.empty) and (not df_x.empty):
        y_init = [(df_y.init[i], df_y.coord[i]) for i in xrange(df_y.shape[0])]
        y_end  = [(df_y.end[i], df_y.coord[i]) for i in xrange(df_y.shape[0])]
        x_init = [(df_x.coord[i], df_x.init[i]) for i in xrange(df_x.shape[0])]
        x_end  = [(df_x.coord[i], df_x.end[i]) for i in xrange(df_x.shape[0])]

        if only_upper: return x_init

        return y_init+y_end+x_init+x_end
    else:
        return []


def plot_shape(img, axis, df=None, nsteps=None):
    """
    function to overplot the shape points onto the image img
    """

    if df is not None and nsteps:
        print 'Error: provide data frame or nsteps, not both'
        return None

    if df is not None:
        edges = bounding_box(img)
        img_c = img[edges[2]:edges[3]+1, edges[0]:edges[1]+1]
        pyl.figure()
        pyl.gray()
        pyl.imshow(img_c)
        if axis == 'y':
            pyl.plot(df.init,df.coord, 'r*')
            pyl.plot(df.end, df.coord, 'r*')
            pyl.show()
        if axis == 'x':
            pyl.plot(df.coord,df.init, 'r*')
            pyl.plot(df.coord,df.end, 'r*')
            pyl.show()

    elif nsteps:
        pyl.figure()
        pyl.gray()
        pyl.imshow(img)
        if axis == 'y':
            df = shape_df(img, 'y', nsteps)
            pyl.plot(df.init,df.coord, 'r*')
            pyl.plot(df.end, df.coord, 'r*')
            pyl.show()
        if axis == 'x':
            df = shape_df(img, 'x', nsteps)
            pyl.plot(df.coord,df.init, 'r*')
            pyl.plot(df.coord,df.end, 'r*')
            pyl.show()



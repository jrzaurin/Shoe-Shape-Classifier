import numpy as np
import math

def radial_edges(r1, r2, n):
    #return a list of radial edges
    re = [ r1* ( (r2/r1)**(k /(n - 1.) ) ) for k in xrange(0, n)]
    return re


def euclid_distance(p1,p2):
    return math.sqrt( ( p2[0] - p1[0] ) ** 2 + ( p2[1] - p1[1] ) ** 2 )


def get_angle(p1,p2):
    #compute the angle between points between -pi and +pi.
    return math.atan2((p2[1] - p1[1]),(p2[0] - p1[0]))

class ShapeContext(object):
    """
    Given a point in the image, for all other points that are within a given
    radius, computes the relative angles. Radii and angles  are stored in a
    "shape matrix" with dimensions: radial_bins x angle_bins. Each element of
    the matrix contains a counter that corresponds to, for a given point, the
    number of points that fall at that i radius and have angle j.
    """

    def __init__(self,nbins_r=6,nbins_theta=12,r_inner=0.1250,r_outer=2.5):
        self.nbins_r        = nbins_r             # number of bins in a radial direction
        self.nbins_theta    = nbins_theta         # number of bins in an angular direction
        self.r_inner        = r_inner             # inner radius
        self.r_outer        = r_outer             # outer radius
        self.nbins          = nbins_theta*nbins_r # total number of bins

    def distM(self, x):
        """
        Compute the distance matrix

        Params:
        -------
        x: points (x,y) in the image

        Returns:
        --------
        result: a distance matrix with euclidean distance
        """
        result = np.zeros((len(x), len(x)))
        for i in xrange(len(x)):
            for j in xrange(len(x)):
                result[i,j] = euclid_distance(x[i],x[j])
        return result

    def angleM(self, x):
        """
        Compute the angle matrix

        Params:
        -------
        x: points (x,y) in the image

        Returns:
        --------
        result: an angle matrix with angles between points
        """
        #x are points: the output from the function get_points_from_img within
        #the morphology package. Result is an angle matrix for every pair of
        #points get_angle computs the angle between points
        result = np.zeros((len(x), len(x)))
        for i in xrange(len(x)):
            for j in xrange(len(x)):
                result[i,j] = get_angle(x[i],x[j])
        return result

    def compute(self,points,r=None):
        # Compute a distance matrix for every pair of points
        r_array = self.distM(points)

        # Normalize the distance matrix by the mean distance
        mean_dist = r_array.mean()
        r_array_n = r_array / mean_dist

        # Create radial bins:
        r_bin_edges = radial_edges(self.r_inner,self.r_outer,self.nbins_r)

        # create a matrix with labels depending on the location of the points
        # within the radial bins
        r_array_q = np.zeros((len(points),len(points)), dtype=int)
        for m in xrange(self.nbins_r):
            r_array_q +=  (r_array_n < r_bin_edges[m])
        fz = r_array_q > 0 # boolean matrix. True = within radius of interest

        # Angular Bin
        theta_array = self.angleM(points)
        # Ensure all angles are between 0 and 2Pi
        theta_array_2 = theta_array + 2*math.pi * (theta_array < 0)
        # from angle value to angle bin
        theta_array_q = (1 + np.floor(theta_array_2 /(2 * math.pi / self.nbins_theta))).astype('int')

        # Bin histogram: hstack of shape matrices
        BH = np.zeros(len(points)*self.nbins)
        for i in xrange(0,len(points)):
            sm = np.zeros((self.nbins_r, self.nbins_theta))
            for j in xrange(len(points)):
                if (fz[i, j]):
                    # if point is within radius add 1 to the corresponding location in sm
                    sm[r_array_q[i, j] - 1, theta_array_q[i, j] - 1] += 1
            BH[i*self.nbins:i*self.nbins+self.nbins] = sm.reshape(self.nbins)

        return BH


import numpy as np
import matplotlib.pyplot as plt
import bcolz
from random import sample
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from scipy.spatial.distance import cdist,pdist

def save_array(fname, arr):
    carr=bcolz.carray(arr, rootdir=fname, mode='w'); carr.flush()

def load_array(fname):
    return bcolz.open(fname)[:]


def plot_cluster(image_list, labels, c_id, n_images=20, cols=4):

    idx = np.where(labels == c_id)[0]
    idx = sample(idx, n_images)
    imgs = [image_list[i] for i in idx]

    nrow = len(idx) / cols
    ncol = cols

    if ((ncol * nrow) != len(idx)): nrow = nrow + 1

    plt.figure()
    for i,img in enumerate(imgs):
        plt.subplot(nrow,ncol,i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()


def avg_within_ss(X, k):

    model = MiniBatchKMeans(init='k-means++', n_clusters=k, batch_size=50,
                          n_init=3, max_no_improvement=10, verbose=0)
    model.fit(X)

    centroids = model.cluster_centers_
    dist_c = cdist(X, centroids, 'euclidean')
    dist   = np.min(dist_c, axis=1)
    avgwithinss = sum(dist**2)/X.shape[0]

    return avgwithinss


def perc_var_explained(X,k):

    model = MiniBatchKMeans(init='k-means++', n_clusters=k, batch_size=50,
                          n_init=3, max_no_improvement=10, verbose=0)
    model.fit(X)

    centroids = model.cluster_centers_
    dist_c = cdist(X, centroids, 'euclidean')
    dist   = np.min(dist_c, axis=1)
    tot_withinss = sum(dist**2)
    totss = sum(pdist(X)**2)/X.shape[0]
    betweenss = totss - tot_withinss
    pve = (betweenss/totss  *100)

    return pve

def bic(X, k):

    model = MiniBatchKMeans(init='k-means++', n_clusters=k, batch_size=50,
                          n_init=3, max_no_improvement=10, verbose=0)
    model.fit(X)

    centers = model.cluster_centers_
    centers = np.expand_dims(centers, axis=1)
    labels  = model.labels_
    N_C = np.bincount(labels)
    R, M = X.shape

    wcss = sum([sum(cdist(X[np.where(labels == c)], centers[c], 'euclidean')**2) for c in range(k)])
    var = (1.0/(R-k)/M) * wcss
    const_term = 0.5 * k * np.log(R) * (M+1)

    BIC = np.sum([ ( Rn * np.log(Rn) ) -
                   ( Rn * np.log(R) ) -
                   ( ((Rn * M) / 2) * np.log(2*np.pi*var) )  -
                   ( (Rn - 1) * M/ 2 )
                   for Rn in N_C]) - const_term

    return BIC


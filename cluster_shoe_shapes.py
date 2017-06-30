import numpy as np
import cv2
import os
import shapecontext
import morphology_utils
import multiprocessing
import matplotlib.pyplot as plt
import cPickle as pickle
import data_utils
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans

if __name__ == '__main__':

	IM_DIR = "data/shoe_images/"
	DATA_PRODUCTS_DIR = 'data_processed/'
	MODEL_DIR = "models/"

	print 'Reading images'
	images = []
	for img in os.listdir(IM_DIR):
		if 'jpg' in img:
			path = os.path.join(IM_DIR, img)
			images.append(cv2.imread(path))

	sc = shapecontext.ShapeContext()
	def sc_array(img):

		sp = morphology_utils.shape_points(img, 30)
		bh = sc.compute(sp)

		return bh

	print 'Computing the Shape Context arrays in Parallel'
	num_cores = multiprocessing.cpu_count()
	sc_arrays_list  = Parallel(n_jobs=num_cores)(delayed(sc_array)(img) for img in images)
	sc_arrays = np.array(sc_arrays_list)
	data_utils.save_array(os.path.join(DATA_PRODUCTS_DIR, 'shoe_shape_arrays.bc'), sc_arrays)
	# to load
	# sc_arrays = data_utils.load_array(os.path.join(DATA_PRODUCTS_DIR, 'shoe_shape_arrays.bc'))

	shoe_clusters = MiniBatchKMeans(init='k-means++', n_clusters=16, batch_size=50,
			                        n_init=3, max_no_improvement=10, verbose=0)
	shoe_clusters.fit(sc_arrays)
	shoe_clusters_labels = shoe_clusters.labels_
	pickle.dump(shoe_clusters, open(os.path.join(MODEL_DIR, 'shoe_cluster_model.p'), 'wb'))

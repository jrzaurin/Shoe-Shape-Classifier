# Shoe-Shape-Classifier
Using the shape context algorithm to classify shoe shapes.

## The problem
Let's say that you work at a fashion aggregator and you receive hundreds of different shoes every week from different retailers. These retailers send you one identifier and a few images **per shoe-model**, with different points of view (front-view, side-view, sole, etc). Sometimes they send you a "default" identifier, indicating which of the images for a given shoe should be displayed, normally the side view. However, this is not always the case. 

You want your website to look perfect, so ideally, you want to always shoe the same point of view for all shoe images. Therefore, the question is: How do we find the side view as shoe-images come through the feed? 

## The proposed solution
Here I show a version of part of the solution that I implemented back in the days when I faced this problem. The solution here presented uses the [Shape Context](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/sc_digits.html) algorithm to compute the shape descriptors and [MiniBatchKMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html) to cluster the shapes. 

For a series of reasons I cannot share the dataset here. There are a series of plots and diagrams that I hope are sufficient to illustrate the process. 

The steps are:

1-. Use the Shape Context Algorithm to parameterise the shape of the shoes

2-. Use k-means (MiniBatch) to cluster the shapes

You will see that after clustering the shapes there is still some processing needed to isolate the side view. If I have the time I will upload the implementation of such processing. 

## Additional use and further improvements

As it is straightforward to understand, the shape arrays can also be used to build a content-based recommendation algorithm, recommending shoes of similar shape. One could complement this with color histograms or using additional features, such as [haralick textural features](http://haralick.org/journals/TexturalFeatures.pdf), or [GIST descriptors](http://www.quaero.org/media/files/bibliographie/inria_qpr6_douze_gist_evaluation.pdf). 

Of course, ultimately, if you have the budget to pay for someone to classify the shoe images, you can turn your problem into a supervised one and use Deep Learning. A series of convolutional filters will surely capture shapes, color, patterns, etc.

## The code:

I have run this example with 5000 shoe images (jpg) that are in my disk, in a dir: *data/shoe_images/* 
In the "real world" the image feed normally comes in a form of json files with urls pointing towards the images.

The images are of 150 width and varying height (relative to the width). If you have a similar dataset, you could run the code by simply

TO RUN:

`python cluster_shoe_shapes.py --n_clusters k`

Where k is the number of clusters to use. 

Perhaps, the most usefull part of this repo is at the directory demo, where details of the process are provided. There I recommend to have a look to the notebooks in the following order: 

1) morphology_utilities.ipynb

2) shape_context_algo.ipynb

3) clustering_shapes.ipynb

4) remove_background.ipynb

I will emphasize again, I cannot share the dataset. But I am sure once you go through the notebooks you will easily be able to "play" with your own images. 

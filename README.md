# Shoe-Shape-Classifier
Using the shape context algorithm to classify shoe shapes.

## THE PROBLEM
Let's say that you work at a fashion aggregator and you receive hundreds of different shoes every week from different retailers. These retailers send you one identifier and a few images **per shoe-model**, with different points of view (front-view, side-view, sole, etc). Sometimes they send you a "default" identifier, indicating which of the images for a given shoe should be displayed, normally the side view. However, this is not always the case. 

You want your website to look perfect, so ideally, you want to always shoe the same point of view for all shoe images. Therefore, the question is: How do we find the side view as shoe-images come through the feed? 

# THE SOLUTION
Here I show a version of part of the solution that I implemented back in the days when I faced this problem. The solution here presented uses the [Shape Context](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/sc_digits.html) algorithm to compute the shape descriptors and [MiniBatchKMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html) to cluster the shapes. 

For a series of reasons I cannot share the dataset here. There are a series of plots and diagrams that I hope are sufficient to illustrate the process. 

The steps are: 
1-. Use the Shape Context Algorithm to parameterise the shape of the shoes
2-. Use k-means (MiniBatch) to cluster the shapes

You will see that after clustering the shapes there is still some processing needed to isolate the side view. If I have the time I will upload the implementation of such processing. 

As it is straightforward to understand, the shape arrays can also be used to build a content-based recommendation algorithm, recommending shoes of similar shape. 



# Primary dendrite detector for single crystal superalloy

This repository shows an algorithm for the detection of crosses in dendritic structures.
The algorithm has been developed for an image like the one shown below
<p align="center">
  <img src="https://github.com/giovannicampa/cross_detector/blob/master/src/dendrite_crosses.jpg" width="480">
</p>


## Operations

In order to find the location of the crosses in the picture, the following steps are carried out in the order given:
1. **Preprocessing** of the picture
2. **Clustering** of the resulting blobs to analyse them separately
3. **Cross detection** in the blob

### Preprocessing
The preprocessing steps used to prepare the image are the ones shown in the picture below. The result is an ensemble of white blobs on black backgroud. Each of the blobs might represent a cross. Whether this is true, will be checked in the **Cross detection** part.

<p align="center">
  <img src="https://github.com/giovannicampa/cross_detector/blob/master/src/Preprocessing.png" width="1000">
</p>

### Clustering
The clustering operation separates the different blobs from each other so that they can be analysed separetely. The algorithm used for this operation is DBSCAN and uses as features, the row and column location of each white pixel.

### Cross detection
The blobs resuling from the clustering might still contain artifacts. To obtain better results, only the centre of the blob is considered. Once this has been selected, the RANSAC algorithm is used to find the main axis of the cross.

To find the second arm of the cross, if there is one, first the points lying close to the main axis are removed from the cluster. The resulting cluster is then analysed with the RANSAC algorithm.

The centre of the cross is considered to be at the intersection of the two found arms. In case the arms are not perpendicular, the cluster is not considered to be a cross.
<p align="center">
  <img align="center" src="https://github.com/giovannicampa/cross_detector/blob/master/src/cluster_analysis.png" width="1000">
</p>


## Results

The resulting plot shows the detected crosses and tells the mean, max and minimum distance between neighbouring crosses.
<p align="center">
  <img align="center" src="https://github.com/giovannicampa/cross_detector/blob/master/src/result.png" width="1000">
</p>

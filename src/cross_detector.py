from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np

import cv2

from skimage import io
from skimage.color import rgb2gray
from skimage.transform import hough_line, hough_line_peaks
from skimage.morphology import erosion, skeletonize
from skimage.filters import threshold_mean, median
from skimage.exposure import equalize_adapthist
from skimage.draw import rectangle_perimeter

from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor

import matplotlib.pyplot as plt

check_pixel_shift = False
check_preprocessing = False
check_single_cross = False

# ---------------------------------------------------------------------------------------------------------
# - Reading image

image_original = io.imread("./src/dendrite_crosses.jpg")
image_cropped = image_original[20:-20, 20:-20]              # Cropping the border
image= rgb2gray(image_cropped)                              # Converting to gray
image[800:-1, 860:-1] = 0                                   # Removing the measure reference at the bottom right

# ---------------------------------------------------------------------------------------------------------
# - Image preprocessing

image_contrast = equalize_adapthist(image)                  # increase contrasts (preparation for thresholding)
image_median_filter = median(image_contrast)                # median filter to sharped edges

# Removing background with thresholding
threshold = threshold_mean(image_median_filter)             # thresholding
image_thresholded = image_median_filter > threshold         # removing the background
image_thresholded = image_thresholded/255.0                 # adjusting range for cv2

# Erode image to remove smaller areas of unnnecessary details
erode = cv2.erode(src = image_thresholded, kernel = np.ones((2,2), np.uint8), iterations = 1)
opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel = np.ones((6,6)), iterations=3)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel = np.ones((6,6)), iterations=1)
erode = cv2.erode(src = closing, kernel = np.ones((2,2), np.uint8), iterations = 15)

# Reducing the resulting crosses to lines
image_skeleton = skeletonize(erode*255.0)
image_skeleton_dilated = cv2.dilate(src = image_skeleton/255.0, kernel = np.ones((2,2), np.uint8), iterations = 2)

# Correcting the shift that occurs due to the preprocessing
shift_rows = 10
shift_cols = 10
image_skeleton_dilated = image_skeleton_dilated[shift_rows:,shift_cols:]
image_skeleton_dilated = np.append(image_skeleton_dilated, np.zeros([image_skeleton_dilated.shape[0],shift_cols]), axis = 1)
image_skeleton_dilated = np.append(image_skeleton_dilated, np.zeros([shift_rows,image_skeleton_dilated.shape[1]]), axis = 0)


# ------------------------------------------------------------------------------------------------
# - Plots to check the preprocessing results
if check_preprocessing:

    fig, ax = plt.subplots(3,2)
    fig.suptitle('Preprocessing')
    ax[0,0].imshow(image_thresholded, cmap = "gray")
    ax[0,1].imshow(opening, cmap = "gray")
    ax[1,0].imshow(closing, cmap = "gray")
    ax[1,1].imshow(erode, cmap = "gray")
    ax[2,0].imshow(image_skeleton, cmap = "gray")
    ax[2,1].imshow(image_skeleton_dilated, cmap = "gray")

    titles = ["image_thresholded","opening","closing","erode", "skeleton", "skeleton dilated"]

    for i, a in enumerate(ax.ravel()):
        a.set_title(titles[i])
        a.set_axis_off()

if check_pixel_shift:

    f = plt.figure()
    plt.imshow(image, cmap = "gray")
    f2 = plt.figure()
    plt.imshow(image_skeleton_dilated, cmap = "gray")


# ---------------------------------------------------------------------------------------------------------
# - Clustering the cross blobs to analyse them separately

# Converting the image in a list of features for every pixel as a preparation
# for the clustering 
pixel_list = []
rows, cols = image_skeleton_dilated.shape
for r in range(rows):
    for c in range(cols):
        pixel_list.append([image_skeleton_dilated[r, c], r, c])     # [Pixel value, row, col]

pixel_list = np.array(pixel_list)

# Clustering to separate the crosses
dbscan = DBSCAN(eps=1, min_samples=5, n_jobs = -1).fit(pixel_list)
labels = dbscan.labels_.reshape(image_skeleton_dilated.shape)


# References of the cross in the image
top = {"row":0, "col":0}
bottom = top.copy()
left = top.copy()
right = top.copy()
centre = top.copy()
centre_cross = top.copy()



def process_cluster(label):
    """ Processing of the blob corresponding to the input label

    The function takes as input the id label of the current cluster and tries to remove the unnecessary 
    pixels by selecting a square area around the centre of the blob. The resulting blob should have 
    a shape that resembles the one of a cross. At this point the line fitting algorithms (hough transform, PCA)
    can be applied
    
    The "check_single_cross" option allows to plot the result for each cluster separately

    Input values:
    - label: cluster id that corresponds to the current blob

    Return values:
    - centres of the cross: two lists with the [col, row] coordinates of the centre of the cross
    of the current cluster. The first found with the custom method, the second with the centre of mass of the cluster
    """

    nr_elements = sum(sum(labels == label))

    # Skipping too small and too large clusters
    if nr_elements < 200 or nr_elements > 3500 or label == -1:
        # print("Skipped cluster with {}".format(nr_elements))
        return None, None, None

    # New image to show the current cluster
    current_cluster = np.zeros(image_skeleton_dilated.shape)
    current_cluster[labels == label] = 255

    # Filtering out the pixels of the current cluster
    idx_cluster = (labels.ravel() == label).tolist()
    pixel_cluster = pixel_list[idx_cluster,:]


    # Finding the coordinates of the cross limits
    top_id = np.argmax(pixel_cluster[:,1])
    top["row"] = pixel_cluster[top_id, 1]
    top["col"] = pixel_cluster[top_id, 2]

    bottom_id = np.argmin(pixel_cluster[:,1])
    bottom["row"] = pixel_cluster[bottom_id, 1]
    bottom["col"] = pixel_cluster[bottom_id, 2]

    right_id = np.argmax(pixel_cluster[:,2])
    right["row"] = pixel_cluster[right_id, 1]
    right["col"] = pixel_cluster[right_id, 2]

    left_id = np.argmin(pixel_cluster[:,2])
    left["row"] = pixel_cluster[left_id, 1]
    left["col"] = pixel_cluster[left_id, 2]


    # Size of the skeletized cluster
    height = top["row"] - bottom["row"]
    width = right["col"] - left["col"]

    # Centre of the skeletized cluster
    centre["col"] = left["col"] + width /2
    centre["row"] = bottom["row"] + height/2

    # Drawing a rectangle around the centre we want to keep (abs coordinates)
    centre_cross["row"] = (left["row"] + right["row"])/2
    centre_cross["col"] = (top["col"] + bottom["col"])/2

    height_cross = min(top["row"] - centre_cross["row"], centre_cross["row"] - bottom["row"])
    width_cross = min(right["col"] - centre_cross["col"], centre_cross["col"] - left["col"])

    cutoff_fraction = 0.7

    limit_row_top = centre_cross["row"] + height_cross*cutoff_fraction
    limit_row_bottom = centre_cross["row"] - height_cross*cutoff_fraction
    limit_col_left = centre_cross["col"] - width_cross*cutoff_fraction
    limit_col_right = centre_cross["col"] + width_cross*cutoff_fraction


    # Removing the edges of the cross to get a clearer shape
    for r in range(rows):
        for c in range(cols):
            if r > limit_row_top or r < limit_row_bottom or c > limit_col_right or c < limit_col_left:
                current_cluster[r,c] = 0


    pixel_cluster = []
    for r in range(rows):
        for c in range(cols):
            if current_cluster[r, c] != 0:
                pixel_cluster.append([current_cluster[r, c], r, c])     # [Pixel value, row, col]

    pixel_cluster = np.array(pixel_cluster)


    if pixel_cluster.size == 0:
        return None, None, None

    # Fitting RANSAC on cross and finding main cross arm
    X = pixel_cluster[:,2].reshape(-1, 1)
    y = pixel_cluster[:,1].reshape(-1, 1)
    ransac = RANSACRegressor()

    try:
        ransac.fit(X,y)
    except:
        return None, None, None
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)


    # Fitting RANSAC on the points that do not lie on the previously fitted line
    # finds the second cross arm
    X_perpendicular = []
    y_perpendicular = []

    p1 = np.hstack([min(X).reshape(1,-1).ravel(), ransac.predict(min(X).reshape(1,-1)).ravel()])
    p2 = np.hstack([max(X).reshape(1,-1).ravel(), ransac.predict(max(X).reshape(1,-1)).ravel()])

    for point in pixel_cluster:
        p3 = point[3:0:-1]
        dist_2_line = abs(np.cross(p2-p1, p3-p1)/np.linalg.norm(p2-p1))

        # Only considering the points that are not close to the already fitted line
        if dist_2_line > 10:
            X_perpendicular.append(point[2])
            y_perpendicular.append(point[1])


    try:
        X_perpendicular = np.array(X_perpendicular).reshape(-1,1)
        y_perpendicular = np.array(y_perpendicular).reshape(-1,1)
        line_X_perpendicular = np.arange(X_perpendicular.min(), X_perpendicular.max())[:, np.newaxis]
        ransac.fit(X_perpendicular,y_perpendicular)
        line_y_ransac_perpendicular = ransac.predict(line_X_perpendicular)

    except:
        return None, None, None


    if check_single_cross:

        # Drawing a rectangle around the found blob
        rr_outer, cc_outer = rectangle_perimeter((top["row"], left["col"]), (bottom["row"], right["col"]), shape = current_cluster.shape)
        current_cluster[rr_outer, cc_outer] = 255

        # Drawing a rectangle around the centre of the blob
        rr_inner, cc_inner = rectangle_perimeter((centre_cross["row"] + height_cross*cutoff_fraction, centre_cross["col"] + width_cross*cutoff_fraction), (centre_cross["row"] - height_cross*cutoff_fraction, centre_cross["col"] - width_cross*cutoff_fraction), shape = current_cluster.shape)
        current_cluster[rr_inner, cc_inner] = 255


        plt.scatter(X,y, label="Cluster")
        plt.scatter(X_perpendicular, y_perpendicular, label="Cluster secondary axis")
        plt.scatter(line_X, line_y_ransac, label="Main axis")
        plt.scatter(p1[0], p1[1], label="Point 1")
        plt.scatter(p2[0], p2[1], label="Point 2")
        plt.scatter(line_X_perpendicular, line_y_ransac_perpendicular, label="Secondary axis")
        plt.legend()
        plt.show()


        fig, ax = plt.subplots(1,2)
        ax[0].imshow(current_cluster, cmap="gray")
        ax[0].scatter(centre_cross["col"], centre_cross["row"], color = 'tab:orange', label ="Centre custom")

        crop = image_cropped[int(bottom["row"]): int(top["row"]), int(left["col"]): int(right["col"])]
        ax[1].imshow(image_cropped, cmap = "gray")
        ax[1].scatter(centre["col"], centre_cross["row"], color = 'tab:orange', label ="Centre custom")

        plt.plot(line_X, line_y_ransac, label='RANSAC regressor')
        plt.plot(line_X_perpendicular, line_y_ransac_perpendicular, label='RANSAC regressor perpendicular')

        plt.title("Nr elements: {}, centre at row: {}, col: {}, width: {}, height: {}".format(nr_elements, centre["row"], centre["col"], width, height))
        plt.legend()
        plt.show()

    centre_of_cross = [centre_cross["col"], centre_cross["row"]]
    arm_main = np.hstack([line_X, line_y_ransac])
    arm_secondary = np.hstack([line_X_perpendicular, line_y_ransac_perpendicular])
    
    return centre_of_cross,arm_main, arm_secondary





cross_centres = []
axes_main = np.empty([0,2])
axes_secondary = np.empty([0,2])

if check_single_cross == False: # Parallelize cross finding operation over different cluster

    with ProcessPoolExecutor() as executor:
        results = executor.map(process_cluster, np.unique(labels))

    # Store results
    for cross_centre, ax_main, ax_secondary in results:
        if cross_centre != None:
            cross_centres.append(cross_centre)
            axes_main = np.vstack([axes_main, ax_main])
            axes_secondary = np.vstack([axes_secondary, ax_secondary])


else: # Evaluate every cluster separetely and print the picture

    for label in np.unique(labels):
        print("Processing: {}".format(label))
        cross_centre, ax_main, ax_secondary = process_cluster(label)
        
        if cross_centre != None:
            cross_centres.append(cross_centre)
            axes_main = np.vstack([axes_main, ax_main])
            axes_secondary = np.vstack([axes_secondary, ax_secondary])


cross_centres = np.array(cross_centres)
axes_main = np.array(axes_main)
axes_secondary = np.array(axes_secondary)

fig = plt.figure()
plt.scatter(axes_main[:,0], axes_main[:,1], color = 'tab:green', label = "Axes main")
plt.scatter(axes_secondary[:,0], axes_secondary[:,1], color = 'tab:green', label = "Axes secondary")
plt.scatter(cross_centres[:,0], cross_centres[:,1], color = 'tab:blue', label = "Centres")
plt.imshow(image_cropped, cmap = "gray")
plt.legend()
plt.show()


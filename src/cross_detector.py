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
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib import cm


# ---------------------------------------------------------------------------------------------------------
# - Reading image

image_original = io.imread("./src/dendrite_crosses.jpg")
image= rgb2gray(image_original)


# ---------------------------------------------------------------------------------------------------------
# - Image preprocessing

image_contrast = equalize_adapthist(image)   # increase contrasts (preparation for thresholding)
image_median_filter = median(image_contrast)      # median filter to sharped edges

# Removing background with thresholding
threshold = threshold_mean(image_median_filter)
image_thresholded = image_median_filter > threshold
image_thresholded = image_thresholded/255.0

# Erode image to remove smaller areas of unnnecessary details
image_eroded = cv2.erode(src = image_thresholded, kernel = np.ones((2,2), np.uint8), iterations = 15)
image_dilated = cv2.dilate(src = image_eroded, kernel = np.ones((2,2), np.uint8), iterations = 2)
image_dilated = image_eroded

# Reducing the resulting crosses to lines
image_skeleton = skeletonize(image_dilated*255.0)

image_skeleton_dilated = cv2.dilate(image_skeleton/255.0, kernel = np.ones((2,2), np.uint8), iterations = 2)


# ---------------------------------------------------------------------------------------------------------
# - Clustering

# Converting the image in a list of features for every pixel as a preparation
# for the clustering 
pixel_list = []
rows, cols = image_skeleton_dilated.shape
for r in range(rows):
    for c in range(cols):
        pixel_list.append([image_skeleton_dilated[r, c], r, c])

pixel_list = np.array(pixel_list)

# Clustering to separate the different crosses
dbscan = DBSCAN(eps=1, min_samples=5).fit(pixel_list)
labels = dbscan.labels_.reshape(image_skeleton_dilated.shape)

# Iterating over labels = over separate crosses

top = {"row":0, "col":0}
bottom = top.copy()
left = top.copy()
right = top.copy()
centre = top.copy()
centre_cross = top.copy()



def process_cluster(label):

    nr_elements = sum(sum(labels == label))

    # Skipping too small and too large clusters
    if nr_elements < 200 or nr_elements > 3500:
        # print("Skipped cluster with {}".format(nr_elements))
        return

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

    # Drawing a rectangle around the centre we want to keep
    centre_cross["row"] = (left["row"] + right["row"])/2
    centre_cross["col"] = (top["col"] + bottom["col"])/2

    height_cross = min(top["row"] - centre_cross["row"], centre_cross["row"] - bottom["row"])
    width_cross = min(right["col"] - centre_cross["col"], centre_cross["col"] - left["col"])

    limit_row_top = centre_cross["row"] + height_cross*0.7
    limit_row_bottom = centre_cross["row"] - height_cross*0.7
    limit_col_left = centre_cross["col"] - width_cross*0.7
    limit_col_right = centre_cross["col"] + width_cross*0.7


    # Removing the edges of the cross to get a clearer shape
    for r in range(rows):
        for c in range(cols):
            if r > limit_row_top or r < limit_row_bottom or c > limit_col_right or c < limit_col_left:
                current_cluster[r,c] = 0
                

    current_cluster = cv2.erode(current_cluster/255.0, kernel = (2,2), iterations=3)*255.0

    # # Applying the hough transform to find the edges
    # origin = np.array((0, image_skeleton_dilated.shape[1]))
    # tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    # h, theta, d = hough_line(current_cluster, theta=tested_angles)

    # for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=2)):
    #     y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    #     plt.plot(origin, (y0, y1), '-r')

    # Finding the crosses with PCA
    # pixel_list_current_cluster = []
    # for r in range(rows):
    #     for c in range(cols):
    #         if current_cluster[r,c] != 0:
    #             pixel_list_current_cluster.append([r, c])

    # pixel_list_current_cluster = np.array(pixel_list_current_cluster)

    # pca = PCA(n_components=2)
    # pca.fit(pixel_list_current_cluster)

    # m_1, m_2 = pca.components_[0]

    # dx_1 = np.cos(np.arctan(m_1))
    # dy_1 = np.sin(np.arctan(m_1))

    # dx_2 = np.cos(np.arctan(m_2))
    # dy_2 = np.sin(np.arctan(m_2))


    # Drawing a rectangle around the found blob
    rr_outer, cc_outer = rectangle_perimeter((top["row"], left["col"]), (bottom["row"], right["col"]), shape = current_cluster.shape)
    # current_cluster[rr_outer, cc_outer] = 255
    image_thresholded[rr_outer, cc_outer] = 255

    rr_inner, cc_inner = rectangle_perimeter((centre_cross["row"] + height_cross*0.7, centre_cross["col"] + width_cross*0.7), (centre_cross["row"] - height_cross*0.7, centre_cross["col"] - width_cross*0.7), shape = current_cluster.shape)
    # current_cluster[rr_inner, cc_inner] = 255
    image_thresholded[rr_inner, cc_inner] = 255

    # plt.imshow(image, cmap=cm.gray)

    # plt.imshow(current_cluster, cmap=cm.gray)
    # plt.imshow(image_thresholded, cmap=cm.gray)
    # plt.arrow(centre["col"], centre["row"], centre["col"] + dx_1*10, centre["row"] + dy_1*10, width = 0.5)
    # plt.arrow(centre["col"], centre["row"], centre["col"] + dx_2*10, centre["row"] + dy_2*10, width = 0.5)

    # plt.scatter(centre_cross["col"], centre_cross["row"])

    # plt.title("Nr elements: {}, centre at row: {}, col: {}, width: {}, height: {}".format(nr_elements, centre["row"], centre["col"], width, height))
    # plt.show()
    return centre_cross["col"], centre_cross["row"]



# Parallelize cross finding operation over different cluster
centres = []
with ProcessPoolExecutor() as executor:     
    results = executor.map(process_cluster, np.unique(labels))

# Store results
for cross_centre in results:
    if cross_centre != None:
        centres.append(cross_centre)

# for label in cross_cluster_labels:
#     cross_centre = process_cluster(label)
#     if cross_centre != None:
#         centres.append(cross_centre)



centres = np.array(centres)
plt.scatter(centres[:,0], centres[:,1], color = "r")
plt.imshow(image_thresholded, cmap = cm.gray)
plt.show()

# # ---------------------------------------------------------------------------------------------------------
# # - Plots



# tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
# h, theta, d = hough_line(image_dilated, theta=tested_angles)


# fig, axes = plt.subplots(1, 4, figsize=(15, 6))
# ax = axes.ravel()

# # Hough transform
# ax[2].imshow(np.log(1 + h), extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]], cmap=cm.gray, aspect=1/1.5)
# ax[2].set_title('Hough transform')
# ax[2].set_xlabel('Angles (degrees)')
# ax[2].set_ylabel('Distance (pixels)')
# ax[2].axis('image')


# # Identified lines
# ax[3].imshow(image, cmap=cm.gray)
# origin = np.array((0, image.shape[1]))

# for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
#     y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
#     ax[3].plot(origin, (y0, y1), '-r')


# ax[3].set_xlim(origin)
# ax[3].set_ylim((image.shape[0], 0))
# ax[3].set_axis_off()
# ax[3].set_title('Detected lines')

# plt.tight_layout()
# plt.show()
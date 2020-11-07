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
image_eroded_3 = cv2.erode(src = image_thresholded, kernel = np.ones((2,2), np.uint8), iterations = 15)
image_dilated = cv2.dilate(image_eroded_3, kernel = np.ones((2,2), np.uint8), iterations = 5)

# Reducing the resulting crosses to lines
image_skeleton = skeletonize(image_dilated*255.0)

image_skeleton_dilated = cv2.dilate(image_skeleton/255.0, kernel = np.ones((2,2), np.uint8), iterations = 4)


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

for label in np.unique(labels):

    nr_elements = sum(sum(labels == label))

    # Skipping too small and too large clusters
    if nr_elements < 200 or nr_elements > 3500:
        print("Skipped cluster with {}".format(nr_elements))
        continue

    # New image to show the current cluster
    current_cluster = np.zeros(image_skeleton_dilated.shape)
    current_cluster[labels == label] = 255


    idx_cluster = (labels.ravel() == label).tolist()
    pixel_cluster = pixel_list[idx_cluster,:]

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


    height = top["row"] - bottom["row"]
    width = right["col"] - left["col"]

    # centre = np.array([left["x"] + width /2, bottom["row"] + height/2])
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
                

    # # Applying the hough transform to find the edges
    # origin = np.array((0, image_skeleton_dilated.shape[1]))
    # tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    # h, theta, d = hough_line(current_cluster, theta=tested_angles)

    # for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    #     y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    #     plt.plot(origin, (y0, y1), '-r')



    # Drawing a rectangle around the found blob
    rr_outer, cc_outer = rectangle_perimeter((top["row"], left["col"]), (bottom["row"], right["col"]), shape = current_cluster.shape)
    current_cluster[rr_outer, cc_outer] = 255


    rr_inner, cc_inner = rectangle_perimeter((centre_cross["row"] + height_cross*0.7, centre_cross["col"] + width_cross*0.7), (centre_cross["row"] - height_cross*0.7, centre_cross["col"] - width_cross*0.7), shape = current_cluster.shape)
    current_cluster[rr_inner, cc_inner] = 255

    plt.imshow(current_cluster, cmap=cm.gray)

    plt.title("Nr elements: {}, centre at row: {}, col: {}, width: {}, height: {}".format(nr_elements, centre["row"], centre["col"], width, height))
    plt.show()



# # ---------------------------------------------------------------------------------------------------------
# # - Plots


# fig, ax = plt.subplots(1,3)

# ax[0].imshow(image, cmap=cm.gray)
# ax[1].imshow(image_skeleton_dilated, cmap=cm.gray)

# ax[0].axis("off")
# ax[1].axis("off")

# ax[0].set_title("Median filter")
# ax[1].set_title("Thresholded")

# plt.show()



# tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
# h, theta, d = hough_line(image_dilated, theta=tested_angles)


# fig, axes = plt.subplots(1, 4, figsize=(15, 6))
# ax = axes.ravel()

# # Original image
# ax[0].imshow(image_original, cmap=cm.gray)
# ax[0].set_title('Input image')
# ax[0].set_axis_off()

# # Processed image
# ax[1].imshow(image_skeleton, cmap=cm.gray)
# ax[1].set_title('Processed image')
# ax[1].set_axis_off()

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
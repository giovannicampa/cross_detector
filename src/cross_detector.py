import numpy as np

import cv2

from skimage import io
from skimage.color import rgb2gray
from skimage.transform import hough_line, hough_line_peaks
from skimage.morphology import erosion, skeletonize
from skimage.filters import threshold_mean, median
from skimage.exposure import equalize_adapthist


from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
from matplotlib import cm

# ---------------------------------------------------------------------------------------------------------
# - Reading image

image_original = io.imread("./src/dendrite_crosses.jpg")
image= rgb2gray(image_original)

image_contrast = equalize_adapthist(image)   # increase contrasts
image_median_f = median(image_contrast)      # median filter to sharped edges

threshold = threshold_mean(image_median_f)
image_thresholded = image_median_f > threshold

image_thresholded = image_thresholded/255.0

# Erode image to remove smaller areas
image_eroded_3 = cv2.erode(src = image_thresholded, kernel = np.ones((2,2), np.uint8), iterations = 15)
image_dilated = cv2.dilate(image_eroded_3, kernel = np.ones((2,2), np.uint8), iterations = 5)

image_skeleton = skeletonize(image_dilated*255.0)

image_skeleton_dilated = cv2.dilate(image_skeleton/255.0, kernel = np.ones((2,2), np.uint8), iterations = 4)

# image_skeleton_dilated[image_skeleton_dilated > 0.001] = 0

# plt.imshow(image_skeleton_dilated)
# plt.show()

pixel_list = []
rows, cols = image_skeleton_dilated.shape
for r in range(rows):
    for c in range(cols):
        pixel_list.append([image_skeleton_dilated[r, c], r, c])


# Clustering the different crosses
dbscan = DBSCAN(eps=1, min_samples=5).fit(pixel_list)
labels = dbscan.labels_.reshape(image_skeleton_dilated.shape)

for label in np.unique(labels):

    nr_elements = sum(sum(labels == label))
    # If cluster too small, skip it
    if nr_elements < 200 or nr_elements > 5000:
        print("Skipped cluster with {}".format(nr_elements))
        continue
    current_cluster = np.zeros(image_skeleton_dilated.shape)
    current_cluster[labels == label] = 255

    origin = np.array((0, image_skeleton_dilated.shape[1]))
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(current_cluster, theta=tested_angles)

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        plt.plot(origin, (y0, y1), '-r')

    plt.imshow(current_cluster)

    plt.title("Nr elements: {}".format(nr_elements))
    plt.show()



# fig, ax = plt.subplots(1,3)

# ax[0].imshow(image, cmap=cm.gray)
# ax[1].imshow(image_skeleton_dilated, cmap=cm.gray)

# ax[0].axis("off")
# ax[1].axis("off")

# ax[0].set_title("Median filter")
# ax[1].set_title("Thresholded")

# plt.show()
# ---------------------------------------------------------------------------------------------------------
# - Preprocessing





# tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
# h, theta, d = hough_line(image_dilated, theta=tested_angles)

# # ---------------------------------------------------------------------------------------------------------
# # - Plots
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
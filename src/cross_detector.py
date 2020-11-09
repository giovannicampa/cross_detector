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

from scipy.ndimage.measurements import center_of_mass

check_pixel_shift = True
check_preprocessing = True
check_single_cross = False

# ---------------------------------------------------------------------------------------------------------
# - Reading image

image_original = io.imread("./src/dendrite_crosses.jpg")
image_cropped = image_original[20:-20, 20:-20]
image= rgb2gray(image_cropped)

image[800:-1, 860:-1] = 0

# ---------------------------------------------------------------------------------------------------------
# - Image preprocessing

image_contrast = equalize_adapthist(image)   # increase contrasts (preparation for thresholding)
image_median_filter = median(image_contrast)      # median filter to sharped edges

# Removing background with thresholding
threshold = threshold_mean(image_median_filter)
image_thresholded = image_median_filter > threshold
image_thresholded = image_thresholded/255.0

# Erode image to remove smaller areas of unnnecessary details
opening = cv2.morphologyEx(image_thresholded, cv2.MORPH_OPEN, kernel = np.ones((6,6)), iterations=1)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel = np.ones((6,6)), iterations=1)
erode = cv2.erode(src = closing, kernel = np.ones((2,2), np.uint8), iterations = 15)

# Reducing the resulting crosses to lines
image_skeleton = skeletonize(erode*255.0)
image_skeleton_dilated = cv2.dilate(src = image_skeleton/255.0, kernel = np.ones((2,2), np.uint8), iterations = 2)

# Correcting the shift due to the preprocessing
shift_rows = 10
shift_cols = 10
image_skeleton_dilated = image_skeleton_dilated[shift_rows:,shift_cols:]
image_skeleton_dilated = np.append(image_skeleton_dilated, np.zeros([image_skeleton_dilated.shape[0],shift_cols]), axis = 1)
image_skeleton_dilated = np.append(image_skeleton_dilated, np.zeros([shift_rows,image_skeleton_dilated.shape[1]]), axis = 0)


# ------------------------------------------------------------------------------------------------
# - Check plots
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
    plt.imshow(image)
    f2 = plt.figure()
    plt.imshow(image_skeleton_dilated)

plt.show()


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

# Clustering to separate the crosses and analyse them separately
dbscan = DBSCAN(eps=1, min_samples=5).fit(pixel_list)
labels = dbscan.labels_.reshape(image_skeleton_dilated.shape)


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

    # Drawing a rectangle around the centre we want to keep (abs coordinates)
    centre_cross["row"] = (left["row"] + right["row"])/2
    centre_cross["col"] = (top["col"] + bottom["col"])/2

    height_cross = min(top["row"] - centre_cross["row"], centre_cross["row"] - bottom["row"])
    width_cross = min(right["col"] - centre_cross["col"], centre_cross["col"] - left["col"])

    limit_row_top = centre_cross["row"] + height_cross*0.8
    limit_row_bottom = centre_cross["row"] - height_cross*0.8
    limit_col_left = centre_cross["col"] - width_cross*0.8
    limit_col_right = centre_cross["col"] + width_cross*0.8


    # Removing the edges of the cross to get a clearer shape
    for r in range(rows):
        for c in range(cols):
            if r > limit_row_top or r < limit_row_bottom or c > limit_col_right or c < limit_col_left:
                current_cluster[r,c] = 0
                

    # Centre of the blob as centre of mass
    centre_mass_row, centre_mass_col = center_of_mass(current_cluster)


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
    current_cluster[rr_outer, cc_outer] = 255

    rr_inner, cc_inner = rectangle_perimeter((centre_cross["row"] + height_cross*0.7, centre_cross["col"] + width_cross*0.7), (centre_cross["row"] - height_cross*0.7, centre_cross["col"] - width_cross*0.7), shape = current_cluster.shape)
    current_cluster[rr_inner, cc_inner] = 255

    if check_single_cross:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(current_cluster, cmap="gray")
        ax[0].scatter(centre_cross["col"], centre_cross["row"], color = 'tab:orange', label ="Centre custom")
        ax[0].scatter(centre_mass_col, centre_mass_row, color = 'tab:blue', label = "Centre of mass")

        crop = image_cropped[int(bottom["row"]): int(top["row"]), int(left["col"]): int(right["col"])]
        ax[1].imshow(image_cropped, cmap = "gray")
        ax[1].scatter(centre["col"], centre_cross["row"], color = 'tab:orange', label ="Centre custom")
        ax[1].scatter(centre_mass_col, centre_mass_row, color = 'tab:blue', label = "Centre of mass")

        plt.legend()
        plt.show()
    # plt.arrow(centre["col"], centre["row"], centre["col"] + dx_1*10, centre["row"] + dy_1*10, width = 0.5)
    # plt.arrow(centre["col"], centre["row"], centre["col"] + dx_2*10, centre["row"] + dy_2*10, width = 0.5)

    # plt.scatter(centre_cross["col"], centre_cross["row"])

    # plt.title("Nr elements: {}, centre at row: {}, col: {}, width: {}, height: {}".format(nr_elements, centre["row"], centre["col"], width, height))
    # plt.show()
    return [centre_cross["col"], centre_cross["row"]], [centre_mass_col, centre_mass_row]



# Parallelize cross finding operation over different cluster
centres_custom = []
centres_mass = []
with ProcessPoolExecutor() as executor:     
    results = executor.map(process_cluster, np.unique(labels))

# Store results
for centres in results:
    if centres != None:
        cross_centre_custom, cross_centre_mass = centres
        centres_custom.append(cross_centre_custom)
        centres_mass.append(cross_centre_mass)

# for label in np.unique(labels):
#     cross_centre_custom, cross_centre_mass = process_cluster(label)
#     if cross_centre_custom != None:
#         centres_custom.append(cross_centre_custom)
#         centres_mass.append(cross_centre_mass)



centres_custom = np.array(centres_custom)
centres_mass = np.array(centres_mass)
plt.scatter(centres_custom[:,0], centres_custom[:,1], color = 'tab:orange', label ="Centre custom")
plt.scatter(centres_mass[:,0], centres_mass[:,1], color = 'tab:blue', label = "Centre of mass")
plt.imshow(image_cropped, cmap = "gray")
plt.legend()
plt.show()


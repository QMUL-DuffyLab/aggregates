import cv2
import sys
import numpy as np
from trimer import Aggregate

# code adapted from https://www.pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/

input_file = '004_new.bmp'
pixel_size = 1 # in nanometres
trimer_radius = 2.25 # also nm
image_size = 1024 # in pixels
rho = (trimer_radius / pixel_size) / image_size

np.set_printoptions(threshold=sys.maxsize)
img = cv2.imread(input_file, 0)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
# img is now a big array that's either 0 or 255
output = cv2.connectedComponentsWithStats(img)
(num_labels, labels_im, stats, centroids) = output
print("Number of aggregates = {}".format(num_labels))
aggregates = []

for i in range(0, num_labels):
        if i == 0: # background
            continue
        # otherwise, we are examining an actual connected component
        else:
        	text = "examining component {}/{}".format(i + 1, num_labels)
        # print a status message update for the current connected
        # component
        # extract the connected component statistics and centroid for
        # the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        print("[INFO] {}: area = {}".format(text, area))
        (cX, cY) = centroids[i]
        # clone our original image (so we can draw on it) and then draw
        # a bounding box surrounding the connected component along with
        # a circle corresponding to the centroid
        output = img.copy()
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
        # construct a mask for the current connected component by
        # finding a pixels in the labels array that have the current
        # connected component ID
        componentMask = (labels_im == i).astype("uint8") * 255
        # compare the area of a single trimer to the area of the
        # grain to get a rough estimate of how many circles to try and place
        # also note that 0.9 might still be too high - impossible to pack
        # 100% of the area with circles
        n = int(0.9 * area / (np.pi * (trimer_radius / pixel_size)**2))
        if (n > 0):
            ag = Aggregate(componentMask, x, y, w, h, area, n, rho)
            aggregates.append(ag)
            ag.shapefill.make_svg('components/{:03d}.svg'.format(i))



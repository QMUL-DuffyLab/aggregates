import cv2
import sys
import numpy as np

img = cv2.imread('004_new.bmp', 0)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
output = cv2.connectedComponentsWithStats(img)
(num_labels, labels_im, stats, centroids) = output
print("Number of labels = {}".format(num_labels))
np.set_printoptions(threshold=sys.maxsize)
print("labels[0] = {}".format(labels_im[0]))
print(np.max(labels_im))

# def imshow_components(labels):
#     # Map component labels to hue val
#     for i in range(num_labels):
#         label_hue = np.uint8(179*labels[i]/np.max(labels))
#         blank_ch = 255*np.ones_like(label_hue)
#         labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

#         # cvt to BGR for display
#         labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

#         # set bg label to black
#         labeled_img[label_hue==0] = 0

#         cv2.imwrite('labeled_{:3d}.png'.format(i), labeled_img)
#         # cv2.waitKey()

for i in range(0, num_labels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
        	text = "examining component {}/{} (background)".format(
        		i + 1, num_labels)
        # otherwise, we are examining an actual connected component
        else:
        	text = "examining component {}/{}".format( i + 1, num_labels)
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
        # show our output image and connected component mask
        cv2.imwrite("components/{:3d}.png".format(i), output)
        cv2.imwrite("components/connected_component_{:3d}.png".format(i), componentMask)

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    print(label_hue[0])
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imwrite('labeled.png', labeled_img)
    # cv2.waitKey()

imshow_components(labels_im)

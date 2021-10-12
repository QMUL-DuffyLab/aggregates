#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import sys
import argparse
import numpy as np
from trimer import Aggregate, real_aggregate

# code adapted from https://www.pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/

parser = argparse.ArgumentParser(description="")
parser.add_argument("-i", "--input_file", default="004.bmp", help="Input filename")
parser.add_argument("-s", "--image_size", type=int, default=1000, help="Size of the imaged area - default is 1μm. We assume a square area!")
parser.add_argument("-p", "--pixel_size", default=None, help="Size of one pixel in nanometres")
parser.add_argument("-t", "--trimer_radius", type=float, default=5.0, help="Radius of one trimer in nanometres")
parser.add_argument("-mp", "--max_pulls", type=int, default=10, help="Maximum number of times to pull circles around when packing")

args = parser.parse_args()

img = cv2.imread(args.input_file, 0)

if args.pixel_size is not None:
    assert (float(np.shape(img)[0]) / args.image_size == args.pixel_size),\
    '''Image dimensions not compatible with parameters -
     image size in pixels = {}, image size in nm = {},
     pixel size given = {}. Fix it idiot.'''.format(
            np.shape(img)[0], args.image_size, args.pixel_size)
else:
    args.pixel_size = float(np.shape(img)[0]) / args.image_size

# the size as a proportion of the image for each circle (trimer):
# we can ignore the image dimensions here because we give both of
# these quantities in nanometres
rho = (args.trimer_radius / args.image_size)
r = args.trimer_radius * args.pixel_size
# 2.5 bc we just compare the hypnotenuse to the nn_cutoff
# so this is effectively doing (2 * (1.25 r))
nn_cutoff = 2.5 * r

np.set_printoptions(threshold=sys.maxsize)
# make img into a binary array - either 0 or 255
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
output = cv2.connectedComponentsWithStats(img)
(num_labels, labels_im, stats, centroids) = output
print("Number of aggregates found = {}".format(num_labels))
aggregates = []

for i in range(0, num_labels):
        if i == 0: # background
            continue
        # otherwise, we are examining an actual connected component
        else:
            print("examining component {}/{}".format(i, num_labels))
        # extract connected component statistics for the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        # construct a mask for the current connected component by
        # finding pixels in the labels array that have the current
        # connected component ID
        componentMask = (labels_im == i).astype("uint8") * 255
        # compare the area of a single trimer to the area of the
        # grain to get a rough estimate of how many circles to try and place
        n = int(area / (np.pi * (args.trimer_radius / args.pixel_size)**2))
        if (n > 0):
            ag = real_aggregate(r, nn_cutoff, componentMask, n, args.max_pulls, i)
            ag.make_neighbours("components/{:03d}_neighbours.jpg".format(i))
            aggregates.append(ag)


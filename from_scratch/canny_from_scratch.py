#!/bin/env python3

import sys
import numpy as np
import cv2
from scipy import ndimage
import utils as utl

def _sobel_filters(img):
    Kx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], np.float32)
    Ky = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    return (np.hypot(Ix, Iy), np.arctan2(Iy, Ix))

def _non_max_suppression(mag, direction):
    direction = direction * 180.0 / np.pi
    direction[direction < 0] += 180.0

    for y in range(1, mag.shape[0] - 1):
        for x in range(1, mag.shape[1] - 1):
            q = 255.0 
            r = 255.0 

            if (0.0 <= direction[y, x] < 22.5) or (157.5 <= direction[y, x] <= 180.0):
                q = mag[y, x + 1]
                r = mag[y, x - 1]
            elif (22.5 <= direction[y, x] < 67.5):
                q = mag[y + 1, x - 1]
                r = mag[y - 1, x + 1]
            elif (67.5 <= direction[y, x] < 112.5):
                q = mag[y + 1, x]
                r = mag[y - 1, x]
            elif (112.5 <= direction[y, x] < 157.5):
                q = mag[y - 1, x - 1]
                r = mag[y + 1, x + 1]

            if (mag[y, x] < q) or (mag[y, x] < r):
                mag[y, x] = 0

    return mag


def _threshold(img, lowThreshold, highThreshold):
    img[np.where(img > highThreshold)] = 255.0
    img[np.where(img < highThreshold)] = 0.0
    img[np.where((img <= highThreshold) & (img >= lowThreshold))] = 127.0

    return img 


def _hysteresis(img):
    for y in range(1, img.shape[0] - 1):
        for x in range(1, img.shape[1] - 1):
            if (img[y, x] == 127.0):
                if (
                    (img[y + 1, x - 1] == 255.0) or
                    (img[y + 1, x] == 255.0) or
                    (img[y + 1, x + 1] == 255.0) or
                    (img[y, x - 1] == 255.0) or
                    (img[y, x + 1] == 255.0) or
                    (img[y - 1, x - 1] == 255.0) or
                    (img[y - 1, x] == 255.0) or
                    (img[y - 1, x + 1] == 255.0)
                ):
                    img[y, x] = 255.0 
                else:
                    img[y, x] = 0.0
    return np.uint8(img)


def Canny(image, threshold1, threshold2):
    img = np.float32(image)
    mag, direction = _sobel_filters(img)
    img = _non_max_suppression(mag, direction)
    img = _threshold(img, threshold1, threshold2)
    return _hysteresis(img)

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    my_img = img.copy() 

    blur = cv2.GaussianBlur(img, (5, 5), 1)
    edge = cv2.Canny(blur, 50, 100)

    my_blur = utl.GaussianBlur(my_img, (5, 5), 1)
    my_edge = Canny(my_blur, 30, 70)

    utl.Plot(
        imgs = [
            edge,
            my_edge,
        ],
        titles = [
            'OpenCv Version', 
            'My Version',
        ],
        cmap='Greys_r',
    )
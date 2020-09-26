import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy import ndimage

def GaussianBlur(src, ksize, sigma):
    x_size = ksize[1] // 2
    y_size = ksize[0] // 2
    y, x = np.mgrid[-y_size:y_size + 1, -x_size:x_size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    kernel =  np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    
    return ndimage.filters.convolve(src, kernel)

def DrawLines(img, lines):
    if lines is None:
        return img

    for line in lines:
        for rho, theta in line:
            cosT = np.cos(theta)
            sinT = np.sin(theta)
            x0 = cosT * rho
            y0 = sinT * rho
            x1 = int(x0 + 2000 * (-sinT))
            y1 = int(y0 + 2000 * (cosT))
            x2 = int(x0 - 2000 * (-sinT))
            y2 = int(y0 - 2000 * (cosT))
            
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img


def Plot(imgs, titles, figsize=(12, 12), cmap=None):
    figure = plt.figure(figsize=figsize)
    
    for i in range(len(imgs)):
        subplot = figure.add_subplot(1, len(imgs), i + 1)
        subplot.title.set_text(titles[i])
        subplot.imshow(imgs[i], cmap=cmap)

    plt.show()
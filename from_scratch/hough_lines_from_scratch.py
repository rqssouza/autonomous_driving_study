#!/bin/env python3

import sys
import numpy as np
import cv2
import utils as utl
import canny_from_scratch as canny

#Painfully slow :(
def HoughLines(img, rho_resolution, theta_resulution, threshold): 
    max_dist = np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
    theta_resulution = round(np.rad2deg(theta_resulution))
    acc = np.zeros(
        (round(2 * max_dist / rho_resolution), round(180 / theta_resulution)),
        np.int,
    )
    elected_lines = []
    
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if img[y, x]:
                for t in range(0, 180, theta_resulution):
                    t_rad = np.deg2rad(t)
                    d = x * np.cos(t_rad) + y * np.sin(t_rad)
                    d_indx = int(round(d + max_dist) / rho_resolution)
                    t_indx = int(t / theta_resulution)
                    acc[d_indx, t_indx] += 1
                    if acc[d_indx, t_indx] == threshold:
                        elected_lines.append([d_indx, t_indx])

    lines = np.zeros((len(elected_lines), 1, 2), 'float32')
    for i in range(len(elected_lines)):
        lines[i, 0, 0] = (elected_lines[i][0] * rho_resolution) - round(max_dist)
        lines[i, 0, 1] = np.deg2rad(elected_lines[i][1] * theta_resulution)
    return lines


if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    my_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    my_gray = gray.copy() 

    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(gray, 50, 100)

    my_gray = utl.GaussianBlur(my_gray, (5, 5), 1)
    my_edges = canny.Canny(my_gray, 30, 70)

    img = utl.DrawLines(
        img = img,
        lines = cv2.HoughLines(edges, 1, (np.pi / 180.0) * 1, 300),
    )
    my_img = utl.DrawLines(
        img = my_img,
        lines = HoughLines(my_edges, 1, (np.pi / 180.0) * 1, 300),
    )

    utl.Plot(
        imgs = [
            img,
            my_img
        ],
        titles = [
            'OpenCv Version', 
            'My Version',
        ],
    )
#!/usr/bin/python

import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from __builtin__ import filter

def get_neighbours(thresh, indices_list, i, j):
    dimentions = thresh.shape
    
    def filter_func(index):
        if index in indices_list or index[0] < 0 or index[0] >= dimentions[0] or index[1] < 0 or index[1] >= dimentions[1] or  \
            thresh[index[0], index[1]] == 0 or thresh[index[0], index[1]] == 0 or (index[0] == i and index[1] == j) :
            return False
        return True
    
    return filter(filter_func, np.concatenate(np.transpose(np.meshgrid(range(i-1, i+2), range(j-1, j+2)))).tolist())

def mark_neighbors(thresh, markers, i, j):
    indices_list_index = 0
    indices_list = [[i, j]]
    
    while indices_list_index < len(indices_list):
        index = indices_list[indices_list_index]

#         Add items to the indices list
        neighbors = get_neighbours(thresh, indices_list, index[0], index[1])
        indices_list.extend(neighbors)
        markers[index[0], index[1]] = 255
        indices_list_index += 1


def main():
    
    img = cv2.imread('euler5.jpg', cv2.cv.CV_LOAD_IMAGE_COLOR)    
#     thresh_img = cv2.threshold(img, 150, 255, cv2.cv.CV_THRESH_BINARY)
    
#     img_gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY))
    h,s,v = cv2.split(cv2.cvtColor(img, cv2.cv.CV_BGR2HSV))
#     r,g,b = cv2.split(img)
    ch = s
    ch = cv2.medianBlur(ch,5)
    ch = cv2.GaussianBlur(ch, (5, 5), 0)
#     thresh = cv2.adaptiveThreshold(ch,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,0)
    thresh = cv2.threshold(ch, 255/2 + 5, 255, cv2.THRESH_BINARY)[1]
    
    markers = np.zeros(s.shape, dtype=np.uint8)
#     markers_index = 1
#     for i in xrange(s.shape[0]):
#         for j in xrange(s.shape[1]):
#

    mark_neighbors(thresh, markers, 76, 220)
    
    plt.imshow(markers, 'gray')
    plt.show()
    



if __name__ == '__main__':
    sys.exit(main())

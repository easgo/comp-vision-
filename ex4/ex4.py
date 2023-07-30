#!/usr/bin/python

import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_neighbours(thresh, markers, i, j):
    dimentions = thresh.shape
    
    def filter_func(index):
        if markers[index[0], index[1]] != 0 or index[0] < 0 or index[0] >= dimentions[0] or index[1] < 0 or index[1] >= dimentions[1] or  \
            thresh[index[0], index[1]] == 0 or (index[0] == i and index[1] == j) :
            return False
        return True
    
    return filter(filter_func, np.concatenate(np.transpose(np.meshgrid(range(i-1, i+2), range(j-1, j+2)))).tolist())

def mark_neighbors(thresh, markers, i, j, marking_value):
    indices_list_index = 0
    indices_list = [[i, j]]
    markers[i, j] = marking_value
    
    while indices_list_index < len(indices_list):
        index = indices_list[indices_list_index]

#        Add items to the indices list
        neighbors = get_neighbours(thresh, markers, index[0], index[1])        
        indices_list.extend(neighbors)
        for neighbor in neighbors:
            markers[neighbor[0], neighbor[1]] = marking_value
        
        indices_list_index += 1


def main():
    
    img = cv2.imread('euler5.jpg', cv2.cv.CV_LOAD_IMAGE_COLOR)    
    h,s,v = cv2.split(cv2.cvtColor(img, cv2.cv.CV_BGR2HSV))
    ch = s
    ch = cv2.medianBlur(ch,5)
    ch = cv2.GaussianBlur(ch, (5, 5), 0)

    thresh = cv2.threshold(ch, 255/2 - 10 + 5, 255, cv2.THRESH_BINARY)[1]
    thresh_img_3c = cv2.merge([thresh, thresh, thresh])
    
    markers = np.zeros(s.shape, dtype=np.int32)
    markers_index = 1
    for i in xrange(s.shape[0]):
        for j in xrange(s.shape[1]):
            if thresh[i, j] != 0 and markers[i, j] == 0:
                mark_neighbors(thresh, markers, i, j, markers_index)
                markers_index += 1
    mark_neighbors(thresh, markers, 212, 227, markers_index)
    plt.imshow(cv2.equalizeHist(np.uint8(markers)), 'gray')
    #plt.imshow(thresh, 'gray')
    
    plt.show()
    



if __name__ == '__main__':
    sys.exit(main())

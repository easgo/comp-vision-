#!/usr/bin/python

import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt



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
    
    
    
    
    plt.imshow(thresh, 'gray')
    plt.show()
    



if __name__ == '__main__':
    sys.exit(main())

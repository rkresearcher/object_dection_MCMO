import cv2
import numpy as np
from epigeo import * 

vidcap = cv2.VideoCapture('1.avi')
success,image1 = vidcap.read()
fps = vidcap.get(cv2.CAP_PROP_FPS)

operated_img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
operated_img = np.float32(operated_img)
dest = cv2.cornerHarris(operated_img,2,5,0.07)
dest_old = cv2.dilate(dest,None)

mask = np.zeros_like(image1)
mask[...,1] = 255
while success:
#  cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
     
    # Calculation of Sobelx
 # sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
     
    # Calculation of Sobely
#  sobelxy = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
     
    # Calculation of Laplacian
  laplacian = cv2.Laplacian(image,cv2.CV_64F)

  
  image = cv2.medianBlur(image, 5)
  operated_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  operated_img = np.float32(operated_img)
  dest = cv2.cornerHarris(operated_img,2,5,0.07)
  dest = cv2.dilate(dest,None) 
  image[dest > 0.01 * dest.max()] = [0,0,255]



  flow = cv2.calcOpticalFlowFarneback(dest_old,dest, None, 0.5, 3, 15, 3, 5, 1.2, 0)


  magnitude,angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

  mask[..., 0] = angle * 180 / np.pi / 2
  mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

  rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
  
  #epipolar geometry
  epi(image1,image)  

  cv2.imshow("Dense Optical Flow",rgb)
  
  dest_old = dest
  image1 = image

 # cv2.imshow('sobelx',sobelx)
 # cv2.imshow('sobelxy',sobelxy)
# cv2.imshow('laplacian',laplacian)
  cv2.imshow('with MH',image)
  k = cv2.waitKey(5) & 0xFF
  if k == 27:
     break
 


print ('fames conversion done')
#elastime = count/fps


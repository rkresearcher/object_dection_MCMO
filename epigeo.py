import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def drawlines(img1, img2, lines, pts1, pts2):
    """img1 - image on witch we draw the epilines for the points in img2
       lines - corresponding epilines"""
    r1, c,_ = img1.shape
    print ("r1",r1)
  #  img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
   # img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
       # x0, y0 = map(int, [0, abs(r[0])/abs(r[1]]))
        x0 = 0
        y0 = 0
       # print ('r is', r)
        p1 = abs(r[2]) + (abs(r[0]))*c/(abs(r[1]))
        print (p1)
        if p1>=15000:
             p1 = r1
        x1, y1 = map(int, [c, (p1)])
       # print ('x1',x1)
        print ('y1',y1)
        print ('Done')
        img1 = cv.line(img1, (x0, y0), (x1, y1), color)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


#if __name__ == '__main__':
def epi(img1, img2):
   # img1 = cv.imread('left.jpg', 0)         # queryingImage # left image
    #img2 = cv.imread('right.jpg', 0)        # trainImage # right image
#    img1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
 #   img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2 ,F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    print ('lines', lines2)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    '''plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
  
    plt.show()
   
    plt.close()
    '''
#    print ('line 3',img3)
 #   print ('line 4',img4)

import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((10*7,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
try:
    images = glob.glob('*.jpg')
    print("imgs imported")
except:
    exit(1)

chessboard_dim = (10, 7)

for fname in images:
    
    im = cv.imread(fname, cv.IMREAD_GRAYSCALE)
    im3 = cv.imread(fname, cv.IMREAD_COLOR)
    im = cv.resize(im, (720, 480))
    im3 = cv.resize(im3, (720, 480))
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners( im, chessboard_dim )
    # If found, add object points, image points (after refining them)
    print(str(fname))
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(im,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners( im3, chessboard_dim, corners2, ret )
        cv.imshow('img', im3)
        print(str(fname)+"ret")

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, im.shape[::-1], None, None)

img = cv.imread('16.jpg')
h,  w = [480,720] 
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

file = open("cameraVariables.txt", "w")
file.write(repr(mtx) + "\n" + repr(dist) + "\n" + repr(newcameramtx) + "\n" + repr(roi) + "\n")
file.close()

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi

dst = dst[y:y+h, x:x+w]

cv.imwrite('calibresult.png', dst)

cv.waitKey()
cv.destroyAllWindows()
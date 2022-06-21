# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np


#fps
# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0
    
# contour size
contour_min_area = 0.05  # percent of frame area
contour_max_area = 1 # percent of frame area
h,  w = [480,720] 
area = h*w

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()


camera.resolution = (w, h)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(w, h))

sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# allow the camera to warmup
time.sleep(0.1)


#undisortion parameters
mtx = np.array([[789.01561846,   0.        , 352.99936601],
       [  0.        , 783.42954704, 272.16686614],
       [  0.        ,   0.        ,   1.        ]], dtype=np.uint8)
dist = np.array([[ 0.09735091, -0.21149343, -0.01425879, -0.00228256, -0.07600104]], dtype=np.uint8)
#newcameramtx = np.array([[705.53936768,   0.        , 345.26773159],
 #      [  0.        , 725.87365723, 321.56364447],
  #     [  0.        ,   0.        ,   1.        ]], dtype=np.uint8)
#roi = (23, 36, 656, 441)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
font = cv2.FONT_HERSHEY_SIMPLEX
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    
    
    # time when we finish processing for this frame
    new_frame_time = time.time()
    
    #image undisortion
    # undistort
    dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    image = dst[y:y+h, x:x+w]
    
    #yellow detection
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
   
    lower_yellow = np.array([20,100,100],np.uint8)
    upper_yellow = np.array([40,255,255],np.uint8)    
    mask = cv2.inRange(hsv,lower_yellow,upper_yellow)
    yellowcnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(yellowcnts)>0:
        yellow_area = max(yellowcnts, key=cv2.contourArea)
        (xg,yg,wg,hg)=cv2.boundingRect(yellow_area)
        cv2.rectangle(image, (xg,yg),(xg+wg,yg+hg),(100, 255, 0), 2)

        targets = []
        for c in yellowcnts:
                    
            # basic contour data
            ca = cv2.contourArea(c)
            bx,by,bw,bh = cv2.boundingRect(c)
            ba = bw*bh

            
            p = 100*ca/area
            if (p >=contour_min_area) and (p <= contour_max_area):
                M = cv2.moments(c)#;print( M )
                tx = int(M['m10']/M['m00'])
                ty = int(M['m01']/M['m00'])
                targets.append((p,tx,ty,bx,by,bw,bh,c))

        
        targets.sort()
        targets.reverse()
        targets = targets[:1]
        for size,x,y,bx,by,bw,bh,c in targets:
                cv2.drawContours(image,[c],0,(100, 255, 0), 3)
   
    # Calculating the fps
 
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
 
    # converting the fps into integer
    fps = int(fps)
 
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
 
    # putting the FPS count on the frame
    cv2.putText(image, fps, (7, 30), font, 1, (100, 255, 0), 2, cv2.LINE_AA)
    
    # crosshair
    r = 100
    lc = (0,255,0)
    lw = 1
    y = 480 / 2
    x = 720 / 2
    x = int(round(x,0))
    y = int(round(y,0))

    cv2.line(image,(x,y-r*2),(x,y+r*2),lc,lw)
    cv2.line(image,(x-r*2,y),(x+r*2,y),lc,lw)
    
    #image = cv2.filter2D(image, -1, sharpen_kernel)
    cv2.imshow("rpiCamera", image)
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        #cv2.imwrite('/home/pi/Desktop/imageRPI.jpg', image) 
        exit(0)
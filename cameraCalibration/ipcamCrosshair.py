#libs
import urllib.request
import cv2
import numpy as np
import time

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
#input stream config + execution
url='http://192.168.137.172:8081/shot.jpg'

#undisortion parameters
mtx = np.array([[548.19212071,   0.        , 357.3683516 ],
       [  0.        , 548.33381053, 238.03923489],
       [  0.        ,   0.        ,   1.        ]], dtype=np.uint8)
dist = np.array([[-1.04663944e-02,  3.21502590e-01,  4.16207669e-04,
        -2.21301610e-03, -6.71281368e-01]], dtype=np.uint8)

# font which we will be using to display FPS
font = cv2.FONT_HERSHEY_SIMPLEX

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

while True:
    #capturing frames
    
    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
    
    img = cv2.imdecode(imgNp, -1)
    img = cv2.rotate(img, cv2.ROTATE_180)
    
    # resizing the frame size according to our need
    #img = cv2.resize(img, (720, 480))
    
    # time when we finish processing for this frame
    new_frame_time = time.time()
    
    #image undisortion
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    img = dst[y:y+h, x:x+w]
    
    #yellow detection
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
   
    lower_yellow = np.array([20,100,100],np.uint8)
    upper_yellow = np.array([40,255,255],np.uint8)    
    mask = cv2.inRange(hsv,lower_yellow,upper_yellow)
    yellowcnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(yellowcnts)>0:
        yellow_area = max(yellowcnts, key=cv2.contourArea)
        (xg,yg,wg,hg)=cv2.boundingRect(yellow_area)
        cv2.rectangle(img, (xg,yg),(xg+wg,yg+hg),(100, 255, 0), 2)

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
                cv2.drawContours(img,[c],0,(100, 255, 0), 3)
   
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
    cv2.putText(img, fps, (7, 30), font, 1, (100, 255, 0), 2, cv2.LINE_AA)
    # crosshair
    r = 100
    lc = (0,255,0)
    lw = 1
    y = 480 / 2
    x = 720 / 2
    x = int(round(x,0))
    y = int(round(y,0))

    cv2.line(img,(x,y-r*2),(x,y+r*2),lc,lw)
    cv2.line(img,(x-r*2,y),(x+r*2,y),lc,lw)
    
    cv2.imshow('phoneCamera', img)
    if ord('q') == cv2.waitKey(1):
        exit(0)
#libs
import urllib.request
import cv2
import numpy as np
import time
import threading

  
#input stream config + execution
url1='http://192.168.137.172:8081/shot.jpg'
url2='http://192.168.137.40:8080/shot.jpg'
cam1='redmi1'
cam2='redmi2'

def camera(url, cam):
    # used to record the time when we processed last frame
    #prev_frame_time = 0
    # used to record the time at which we processed current frame
    #new_frame_time = 0
    while True:
        #capturing frames
        imgPath = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)  
        
        img = cv2.imdecode(imgNp, -1)
    
        #img = imutils.resize (img, width = 450)
    
        img = cv2.rotate(img, cv2.ROTATE_180)
        # Our operations on the frame come here
        gray = img
    
    
    # resizing the frame size according to our need
        gray = cv2.resize(gray, (720, 480))
    # font which we will be using to display FPS
        font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
        #new_frame_time = time.time()
    
    #red detection
        hsv = cv2.cvtColor(gray,cv2.COLOR_BGR2HSV)
   
        lowerRed = np.array([88,50,50],np.uint8)
        upperRed = np.array([179,255,255],np.uint8)
        
        mask = cv2.inRange(hsv,lowerRed,upperRed)
        bluecnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
        if len(bluecnts)>0:
            blue_area = max(bluecnts, key=cv2.contourArea)
            (xg,yg,wg,hg)=cv2.boundingRect(blue_area)
            cv2.rectangle(gray, (xg,yg),(xg+wg,yg+hg),(100, 255, 0), 2)
        
    # Calculating the fps
 
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
        #fps = 1/(new_frame_time-prev_frame_time)
        #prev_frame_time = new_frame_time
 
    # converting the fps into integer
        #fps = int(fps)
 
    # converting the fps to string so that we can display it on frame
    # by using putText function
        #fps = str(fps)
 
    # putting the FPS count on the frame
        #cv2.putText(gray, fps, (7, 30), font, 1, (100, 255, 0), 2, cv2.LINE_AA)
        #windowName = 'phoneCamera' + cam
        cv2.imshow(cam, gray)

if __name__ == "__main__":
    t1 = threading.Thread(target=camera, args=(url1,cam1,))
    #t1.setDaemon(True)
    
    #t2.setDaemon(True)
    t1.start()
    #t2.start()
    
    #t2 = threading.Thread(target=camera, args=(url2,cam2,))
    #t2.start()
    while True:    
        if ord('q') == cv2.waitKey(1):
            t1.join()
         #   t2.join()
            exit(0)
        pass
    

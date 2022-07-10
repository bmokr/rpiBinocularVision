#libs
import urllib.request
import cv2
import numpy as np
import threading
import queue
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
import math

class Camera_THR:
    
    #buffer setup
    buffer_length = 5
    buffer_all = False
    buffer = None
    
    urlCam = ''
    camName = ''
    camera_frame_rate = 30
    cameraType = ''
    h,  w = [480,720]
    
        
    def cameraStart(self):
        
        self.stopped = False
        #buffer
        if self.buffer_all:
            self.buffer = queue.Queue(self.buffer_length)
        else:
        #last frame only
            self.buffer = queue.Queue(1)

        #black frame 
        self.black_frame = np.zeros((self.h,self.w,3),np.uint8)

        #start thread
        if self.cameraType == 'redmi':
            self.thread = threading.Thread(target=self.loopREDMI)
            self.thread.start()
        elif self.cameraType == 'rpi':
            self.thread = threading.Thread(target=self.loopRPI)
            self.thread.start()
        
    def cameraStop(self):
        
        self.stopped = True
        time.sleep(0.5)
        self.buffer = None
        self.thread.join()

    def loopREDMI(self):
        
        #disortion parameters
        mtx = np.array([[548.19212071, 0., 357.3683516], [0., 548.33381053, 238.03923489],
        [0., 0., 1.]], dtype=np.uint8)
        dist = np.array([[-1.04663944e-02, 3.21502590e-01, 4.16207669e-04,
        -2.21301610e-03, -6.71281368e-01]], dtype=np.uint8)

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (self.w,self.h), 1, (self.w,self.h))
        
        #load start frame
        frame = self.black_frame
        if not self.buffer.full():
            self.buffer.put(frame,False)
            
        while not self.stopped:
            #capturing frames
            
            imgPath = urllib.request.urlopen(self.urlCam)
            imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)  
            img = cv2.imdecode(imgNp, -1)    
            img = cv2.rotate(img, cv2.ROTATE_180)
            img = cv2.resize(img, (self.w, self.h))
            
            #undistort
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            #crop the image
            x, y, w, h = roi
            img = dst[y:y+h, x:x+w]
                
            #true buffered mode (for files, no loss)
            if self.buffer_all:

                #buffer is full, pause and loop
                if self.buffer.full():
                    time.sleep(1/self.camera_frame_rate)
                #or load buffer with next frame
                else:
                    frame = img
                    self.buffer.put(frame,False)
                    
            #false buffered mode (for camera, loss allowed)
            else:
                frame = img
                #open a spot in the buffer
                if self.buffer.full():
                    self.buffer.get()

                self.buffer.put(frame,False)
        
    def loopRPI(self):
        
        camera = PiCamera()
        camera.resolution = (self.w, self.h)
        camera.framerate = 30
        rawCapture = PiRGBArray(camera, size=(self.w, self.h))
        
        
        #disortion parameters
        mtx = np.array([[789.01561846, 0., 352.99936601], [0., 783.42954704, 272.16686614],
        [0., 0., 1.]], dtype=np.uint8)
        dist = np.array([[0.09735091, -0.21149343, -0.01425879, -0.00228256, -0.07600104]], dtype=np.uint8)

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (self.w,self.h), 1, (self.w,self.h))
        
        
        # load start frame
        frame = self.black_frame
        if not self.buffer.full():
            self.buffer.put(frame,False)
            
        #capturing frames
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            if self.stopped == True:
                break
            
            grabbed = False  
            #grab the raw NumPy array representing the image, then initialize the timestamp
            #and occupied/unoccupied text
            img = frame.array
            #undistort
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            #crop the image
            x, y, w, h = roi
            img = dst[y:y+h, x:x+w]
                
            if self.buffer_all:

                #buffer is full, pause and loop
                if self.buffer.full():
                    time.sleep(1/self.camera_frame_rate)
                    
                #or load buffer with next frame
                else:
                    frame = img
                    self.buffer.put(frame,False)
                    
            #false buffered mode (for camera, loss allowed)
            else:
                frame = img
                #open a spot in the buffer
                if self.buffer.full():
                    self.buffer.get()

                self.buffer.put(frame,False)
                
            #clear the stream in preparation for the next frame
            rawCapture.truncate(0)
                    
    def grabber(self,black=True,wait=0):

        #black frame default
        if black:
            frame = self.black_frame

        #no frame default
        else:
            frame = None

        #get from buffer (fail if empty)
        try:
            frame = self.buffer.get(timeout=wait)   
        except:
            pass
          
        return frame


class Detection_Measurment:
    
    #contour size
    contour_min_area = 0.05  #percent of frame area
    contour_max_area = 1     #percent of frame area
    
    lower_yellow = np.array([20,100,100],np.uint8)
    upper_yellow = np.array([40,255,255],np.uint8)
    
    def detect(self,frame):
        #frame dimensions
        width,height,depth = np.shape(frame)
        area = width*height

        #yellow mask
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,self.lower_yellow,self.upper_yellow)
        yellowcnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        targets = []
        if len(yellowcnts)>0:
            
            for c in yellowcnts:
                        
                #basic contour data
                ca = cv2.contourArea(c)
                bx,by,bw,bh = cv2.boundingRect(c)
                ba = bw*bh

                p = 100*ca/area
                if (p >=self.contour_min_area) and (p <= self.contour_max_area):
                    M = cv2.moments(c)
                    tx = int(M['m10']/M['m00'])
                    ty = int(M['m01']/M['m00'])
                    targets.append((p,tx,ty,bx,by,bw,bh,c))
   
            targets.sort()
            targets.reverse()
            targets = targets[:1]
            for size,x,y,bx,by,bw,bh,c in targets:
                    cv2.drawContours(frame,[c],0,(100, 255, 0), 3)
        
        return [(x,y,size) for (size,x,y,bx,by,bw,bh,c) in targets]

class Frame_Angles:
 
    #variables
    pixel_width = 720
    pixel_height = 480

    angle_width = 60
    angle_height = None

    x_origin = None
    y_origin = None

    x_adjacent = None
    x_adjacent = None

    #init
    def __init__(self,pixel_width=None,pixel_height=None,angle_width=None,angle_height=None):

        #full frame dimensions in pixels
        if type(pixel_width) in (int,float):
            self.pixel_width = int(pixel_width)
        if type(pixel_height) in (int,float):
            self.pixel_height = int(pixel_height)

        #full frame dimensions in degrees
        if type(angle_width) in (int,float):
            self.angle_width = float(angle_width)
        if type(angle_height) in (int,float):
            self.angle_height = float(angle_height)

        #do initial setup
        self.build_frame()

    def build_frame(self):

        #center point (also max pixel distance from origin)
        self.x_origin = int(self.pixel_width/2)
        self.y_origin = int(self.pixel_height/2)

        #theoretical distance in pixels from camera to frame
        #this is the adjacent-side length in tangent calculations
        #the pixel x,y inputs is the opposite-side lengths
        self.x_adjacent = self.x_origin / math.tan(math.radians(self.angle_width/2))
        self.y_adjacent = self.y_origin / math.tan(math.radians(self.angle_height/2))

    def angles(self,x,y):

        return self.angles_from_center(x,y)

    def angles_from_center(self,x,y):

        xtan = x/self.x_adjacent
        ytan = y/self.y_adjacent

        xrad = math.atan(xtan)
        yrad = math.atan(ytan)

        return math.degrees(xrad),math.degrees(yrad)

    def pixels_from_center(self,x,y,degrees=True):

        if degrees:
            x = math.radians(x)
            y = math.radians(y)

        return int(self.x_adjacent*math.tan(x)),int(self.y_adjacent*math.tan(y))

    def distance(self,*coordinates):
        return self.distance_from_origin(*coordinates)

    def distance_from_origin(self,*coordinates):
        return math.sqrt(sum([x**2 for x in coordinates]))
    
    def intersection(self,pdistance,langle,rangle,degrees=False):

        #fix degrees
        if degrees:
            langle = math.radians(langle)
            rangle = math.radians(rangle)

        #fix angle orientation (from center frame)
        #here langle is measured from right baseline
        #here rangle is measured from left  baseline
        langle = math.pi/2 - langle
        rangle = math.pi/2 + rangle

        #all calculations using tangent
        ltan = math.tan(langle)
        rtan = math.tan(rangle)

        #get Y value
        #use the idea that pdistance = ( Y/ltan + Y/rtan )
        Y = pdistance / ( 1/ltan + 1/rtan )
       
        #get X measure from left-camera-center using Y
        X = Y/ltan

        #done
        return X,Y

    def location(self,pdistance,lcamera,rcamera,center=False,degrees=True):

        #separate values
        lxangle,lyangle = lcamera
        rxangle,ryangle = rcamera

        #yangle should be the same for both cameras (if aligned correctly)
        yangle = (lyangle+ryangle)/2

        #fix degrees
        if degrees:
            lxangle = math.radians(lxangle)
            rxangle = math.radians(rxangle)
            yangle  = math.radians( yangle)

        #get X,Z (remember Y for the intersection is Z frame)
        X,Z = self.intersection(pdistance,lxangle,rxangle,degrees=False)

        #get Y
        #using yangle and 2D distance to target
        Y = math.tan(yangle) * self.distance_from_origin(X,Z)

        #baseline-center instead of left-camera-center
        if center:
            X -= pdistance/2

        #get 3D distance
        D = self.distance_from_origin(X,Y,Z)
        
        #done
        return X,Y,Z,D

#libs
import urllib.request
import cv2
import numpy as np
import time


#input stream config + execution
url='http://192.168.137.172:8081/shot.jpg'

frameNr = 0

while True:
    #capturing frames
    
    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
    
    img = cv2.imdecode(imgNp, -1)
    img = cv2.rotate(img, cv2.ROTATE_180)

    # resizing the frame size according to our need
    img = cv2.resize(img, (720, 480))
    cv2.imshow('phoneCamera', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("z"):
        print(str(key))
        cv2.imwrite(str(frameNr)+'.jpg', img)
        frameNr += 1
        print(frameNr)
    elif key == ord("q"):
        print(str(key))
        exit(0)
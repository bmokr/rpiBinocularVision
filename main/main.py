#libs
import cv2
import numpy as np
import time
import traceback
import pseudoHeader as pH 

def run():

    try:
        
        #fps
        prev_frame_time = 0
        new_frame_time = 0
        font = cv2.FONT_HERSHEY_SIMPLEX

        #output file
        f = open("measurment.txt", "a")
        f.write("Distance" + "\tMeasurment\n")
        f.close()
        
        #cameras variables
        pixel_width = 720
        pixel_height = 480
        camera_separation = float(input("Enter cameras separation <x.x>[m]: "))
        
        #input stream config
        url1 = 'http://192.168.137.172:8081/shot.jpg'
        url2 = 'http://192.168.137.40:8080/shot.jpg'
        cam1 = 'redmiL'
        cam2 = str(input("First left camera: 8081\nEnter second right camera <redmiR/rpi>: "))
        
        #setup
        cL = pH.Camera_THR()
        cR = pH.Camera_THR()
        
        #left parameters
        cL.urlCam = url1
        cL.camName = cam1
        cL.cameraType = 'redmi'
        targeter1 = pH.Detection_Measurment()
        angle_width_1 = 68
        angle_height_1 = 46
        
        #right parameters
        cR.camName = cam2
        targeter2 = pH.Detection_Measurment()
        
        if cam2 == "redmiR":
            cR.cameraType = 'redmi'
            angle_width_2 = 68
            angle_height_2 = 46
            cR.urlCam = url2
        elif cam2 == "rpi":
            cR.cameraType = 'rpi'
            angle_width_2 = 58
            angle_height_2 = 44

        #set up angles, one for each
        #can be changed to only one if cameras are the same
        angler_1 = pH.Frame_Angles(pixel_width,pixel_height,angle_width_1,angle_height_1)
        angler_1.build_frame()
        
        angler_2 = pH.Frame_Angles(pixel_width,pixel_height,angle_width_2,angle_height_2)
        angler_2.build_frame()

        #start threads
        cL.cameraStart()
        cR.cameraStart()
        
        #pause to stabilize
        time.sleep(0.5)
       
        # variables
        maxsd = 2 #maximum size difference of targets, percent of frame
        klen  = 3 #length of target queues

        #last positive target
        X,Y,Z,D = 0,0,0,0

        #target queues
        x1k,y1k,x2k,y2k = [],[],[],[]
        x1m,y1m,x2m,y2m = 0,0,0,0

        #main loop
        while True:
            
            #get frames
            frame1 = cL.grabber(black=True,wait=1)
            frame2 = cR.grabber(black=True,wait=1)
            
            #get target
            target1 = targeter1.detect(frame1)
            target2 = targeter2.detect(frame2)

            #check queues
            if not (target1 and target2):
                x1k,y1k,x2k,y2k = [],[],[],[] #reset
            else:

                #split
                x1,y1,s1 = target1[0]
                x2,y2,s2 = target2[0]             
                
                #chec similar size
                if abs(s1-s2) > maxsd:
                    x1k,y1k,x2k,y2k = [],[],[],[] #reset
                else:

                    # update queues
                    x1k.append(x1)
                    y1k.append(y1)
                    x2k.append(x2)
                    y2k.append(y2)

                    #check if queues are full
                    if len(x1k) >= klen:

                        #trim
                        x1k = x1k[-klen:]
                        y1k = y1k[-klen:]
                        x2k = x2k[-klen:]
                        y2k = y2k[-klen:]

                        #mean values
                        x1m = sum(x1k)/klen
                        y1m = sum(y1k)/klen
                        x2m = sum(x2k)/klen
                        y2m = sum(y2k)/klen
                                
                        #get angles from camera centers
                        xlangle,ylangle = angler_1.angles_from_center(x1m,y1m)
                        xrangle,yrangle = angler_2.angles_from_center(x2m,y2m)
                        
                        #triangulate
                        X,Y,Z,D = angler_1.location(camera_separation,(xlangle,ylangle),(xrangle,yrangle),center=True,degrees=True)
        
            
            #display coordinate data & fps data
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            
            text = 'FPS: {:2.0f}\nX: {:3.1f}\nY: {:3.1f}\nZ: {:3.1f}\nD: {:3.1f}'.format(fps,X,Y,Z,D)
            lineloc = 0
            lineheight = 30
            for t in text.split('\n'):
                lineloc += lineheight
                cv2.putText(frame1, t, (10,lineloc), font, 1, (100,255,0), 1, cv2.LINE_AA, False)
            
            #display windows
            cv2.imshow(cam1, frame1)
            cv2.imshow(cam2, frame2)
            
            #keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
            elif key == ord('x'):
                f = open("measurment.txt", "a")
                f.write(str(D) + "\t" + str(input("Enter measurment: ") + "\n"))
                f.close()
    
    except:
        print(traceback.format_exc())

    #close cameras & windows
    try:
        cL.cameraStop()
    except:
        pass

    try:
        cR.cameraStop()
    except:
        pass
                
    cv2.destroyAllWindows()
    exit(0)   
        
if __name__ == "__main__":
    run()
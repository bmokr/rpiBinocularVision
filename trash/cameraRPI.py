from picamera import PiCamera
import cv2

camera = PiCamera()
camera.start_preview()
if ord('q') == cv2.waitKey():
    camera.capture('/home/pi/Desktop/imagePycharm.jpg')
    camera.stop_preview()
    exit(0)
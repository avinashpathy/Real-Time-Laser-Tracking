#CODE FOR OBJECT DETECTION AND TRACKING USING OPENCV.

from memory_profiler import profile
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
import cv2
import numpy as np
import time

@profile
def detect():
    capture=cv2.VideoCapture(r'C:\Users\Avinash\Downloads\laser.mkv') #we use 0 for webcam and file name of the pre-recorded video for tracking objects in pre-recorded videoes.
    start = time.time()
    #r'C:\Users\Arjun Rao N\Downloads\lasser.mkv'
    countt=0
    totaltime=0
    printt=True #used as a key to print the execution time once 
    while True:
        e1 = cv2.getTickCount()
        countt+=1
        ret, bgr_frame = capture.read()
        if ret:
            hsv_frame = cv2.cvtColor(bgr_frame,cv2.COLOR_BGR2HSV) #converting BGR color grading to HSV(hue saturation value)

            #creating a numpy array for boarder colors. The colors between these two colors are only tracked.
            lower_limit = np.array([161,155,84]) #HSV values of light red.
            upper_limit = np.array([179,255,255]) ##HSV values of dark red.

            color_mask = cv2.inRange(hsv_frame, lower_limit, upper_limit) #crating mask.
            contourparts, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #used to lists of x and y axis of any red color present in the frame.
            contourparts = sorted(contourparts, key=lambda x:cv2.contourArea(x), reverse=True) #selects the largest red spot by considering the area.

            a=[] #used as a key to enter the if statement.
            xaxis_medium=0 #this variable is used as output to draw a vertical line passing through the center of the red object detected.
            yaxis_medium=0

            if contourparts!=a:  
                for cnt in contourparts:
                    (leftmost_end, bottommost_end, width, height) = cv2.boundingRect(cnt) #to get the dimentions of the red object.
                    xaxis_medium = int((leftmost_end +(leftmost_end+width)) /2) #to get the x_axis value of center of the red spot.
                    yaxis_medium = int((bottommost_end + (bottommost_end+height)) /2)
                    break

            e2 = cv2.getTickCount()
            t = (e2 - e1)/cv2.getTickFrequency()
            while printt==True:
                print("The time of execution of this function:{}".format(t))
                end=time.time()
                print('Time consumed in seconds is {}'.format(end-start))
                printt=False
            totaltime=totaltime+t

            cv2.line(bgr_frame,(xaxis_medium,0),(xaxis_medium,1000),(0,255,0),2) #used to draw a vertical line passing through the center of the red object.
            cv2.line(bgr_frame,(0,yaxis_medium),(1700,yaxis_medium),(0,255,0),2)
            cv2.imshow("Frame",bgr_frame)


            key = cv2.waitKey(1) #used to terminate the program by pressing 'esc' key
            if key==27:
                break
    print(totaltime)
    print(countt)
    print(totaltime/countt)

    print(cv2.useOptimized()) #checks if OpenCV is running optimized code or not
    capture.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    detect()

#%lprun -f detect detect()

    #                                                       MEMBERS OF THE GROUP:
    #                                                           ARJUN RAO N.
    #                                                           AVINASH PATHY.
    #                                                           BASAVASAGAR K P.
    #                                                           CHINMAY N.

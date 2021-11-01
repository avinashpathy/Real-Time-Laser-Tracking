#CODE FOR BACKGROUND SUBTRACTION
#import cProfile
#from memory_profiler import profile
import cv2
import time
@profile
def detect():
    capture=cv2.VideoCapture(r'C:\Users\Avinash\Downloads\laser.mkv') #we use 0 for webcam and file name of the pre-recorded video for tracking objects in pre-recorded videoes.'G:\movies and series\movies\airlift.mkv'
    start = time.time()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG() 
    countt=0
    totaltime=0
    printt=True #used as a key to print the execution time once 
    while True:
        e1 = cv2.getTickCount()
        countt+=1
        ret, frame = capture.read()
        
        if ret:
            
            fgmask = fgbg.apply(frame)
            color_mask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            
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

            cv2.line(frame,(xaxis_medium,0),(xaxis_medium,1000),(0,255,0),2) #used to draw a vertical line passing through the center of the red object.
            cv2.line(frame,(0,yaxis_medium),(1700,yaxis_medium),(0,255,0),2)
            cv2.imshow("Frame",frame)
            cv2.imshow("Frame2",color_mask)
            

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


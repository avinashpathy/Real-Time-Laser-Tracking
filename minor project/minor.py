import cv2
#inport cv

cap = cv2.VideoCapture(r'C:\Users\Avinash\Downloads\laser.mkv')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG() 

while True:
    ret, frame = cap.read()
    if ret == True:
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        cv2.imshow('frame',fgmask)


        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    else:
        break


cap.release()
cv2.destroyAllWindows()

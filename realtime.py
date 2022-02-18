import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

# reading image
#img = cv2.imread('C:\Gokul\mtech\mini_project\gandhis.png')
cap=cv2.VideoCapture(1)
fourcc= cv2.VideoWriter_fourcc(*'XVID')
#ArduinoSerial=serial.Serial('com7',9600,timeout=0.1)
out= cv2.VideoWriter('face detection4.avi',fourcc,20.0,(640,480))
time.sleep(1)
while cap.isOpened():
    ret, frame= cap.read()
    frame=cv2.flip(frame,1)  #mirror the imag
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    for contour in contours:
        if i == 0:
            i = 1
            continue

        approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
        #cv2.drawContours(frame, [contour], 0, (0, 0, 255), 1)
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
        if len(approx) == 4 and cv2.contourArea(contour) > 500:
             cv2.putText(frame, ' STAMP', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
             cv2.circle(frame, (x, y), 2, (0, 255, 0), 2)
             print("x,y",x,y)
    

    cv2.imshow('img',frame)
    if cv2.waitKey(10)&0xFF== ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

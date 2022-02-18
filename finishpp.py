import cv2
import numpy as np
from matplotlib import pyplot as plt

# reading image
img = cv2.imread('C:\Gokul\mtech\mini_project\gandhis.png')

# converting image into grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# setting threshold of gray image
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# using a findContours() function
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
i = 0
for contour in contours:

	# here we are ignoring first counter because
	# findcontour function detects whole image as shape
	if i == 0:
		i = 1
		continue


	approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
    
    
	
	# using drawContours() function
	cv2.drawContours(img, [contour], 0, (0, 0, 255), 1)
   
	# finding center point of shape
	M = cv2.moments(contour)
	if M['m00'] != 0.0:
		x = int(M['m10']/M['m00'])
		y = int(M['m01']/M['m00'])
	# putting shape name at center of each shape
	#if len(approx) == 3:
	#	cv2.putText(img, 'Triangle', (x, y),
	#				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    
	if len(approx) == 4 and cv2.contourArea(contour) > 60:
            cv2.putText(img, ' STAMP', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.circle(img, (x, y), 2, (0, 255, 0), 2)
            print("x,y",x,y)
            
            
      
     
           
area = cv2.contourArea(contour)               
print("area", area)           
cv2.imshow('shapes', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
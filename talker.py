#!/usr/bin/env python


import rospy
import cv2

import numpy as np
from std_msgs.msg import String 
#from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from turtlesim.srv import TeleportAbsolute

def talker():
    
    rospy.init_node('color', anonymous=True)
   
   
    while not rospy.is_shutdown():
	x=5
	y=5
	w = 0
	h = 0
	lowerBound_blue = np.array([40,70,50])
	upperBound_blue =np.array([140,255,245])

	cap = cv2.VideoCapture(0)
	font = cv2.FONT_HERSHEY_SIMPLEX
	while(cap.isOpened()):
		ret,frame = cap.read()
		img = cv2.resize(frame,(340,220))
		if ret == True:	
			imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
			mask=cv2.inRange(imgHSV,lowerBound_blue,upperBound_blue)
			gaus = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 12)
			#cv2.imshow("Gaussian", gaus)
			#cv2.imshow("mask",mask)
			#cv2.imshow("cam",img)
			kernelOpen=np.ones((5,5))
			kernelClose=np.ones((20,20))
	
			maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
			maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
		
			#cv2.imshow("maskClose",maskClose)
			#cv2.imshow("maskOpen",maskOpen)
			maskFinal=maskClose
			_,conts,h=cv2.findContours(maskFinal,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
			#cv2.drawContours(img,conts,-1,(255,0,0),3)
			rospy.wait_for_service('turtle1/teleport_absolute')
			turtle1_teleport = rospy.ServiceProxy('turtle1/teleport_absolute', TeleportAbsolute)
			for i in range(len(conts)):
			    x,y,w,h=cv2.boundingRect(conts[i])
			    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0), 2)
			    cv2.putText(img, 'Blue!', (x,y+h), font, 0.8, (0, 255, 0), 2)
			   
			
			    x1 = x + w/2
			    y1 = y + h/2
			   
			    turtle1_teleport(x1/30.909,11-y1/18,0)
			    rospy.loginfo("checking for cmd X:" + str(x1/30.909)+" Y: " + str(y1/17)+" Theta: "+ str(0) )
			cv2.imshow("cm",img)
			
			
        		

		#res = cv2.bitwise_and(img,img, mask= maskClose)
		#cv2.imshow("r",res)

			#cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break




        #rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
	cap.release()

	cv2.destroyAllWindows()
        pass


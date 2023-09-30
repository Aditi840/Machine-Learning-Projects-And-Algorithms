import cv2
import numpy as np

#Initiating the camera
cap = cv2.VideoCapture(0)

#Infinite loop for capturing the frames
while(True):
	ret,img = cap.read()

	#Reshaping the image to be fed into kmeans data
	Z = img.reshape((-1,3))
	#Converting it to float data type
	Z = np.float32(Z)

	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 3, 0.9)
	#Number of clusters to be defined
	K = 9
	ret,label,center=cv2.kmeans(Z,K,None,criteria,1,cv2.KMEANS_RANDOM_CENTERS)

	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))
	#increasing the intensity for brighter image
	res2 = res2+15

	cv2.imshow("Cartoonized", res2)
	#Exiting the application on press of q
	if(cv2.waitKey(10)==ord('q')):
		break

cap.release()
cv2.destroyAllWindows()

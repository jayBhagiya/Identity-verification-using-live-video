import numpy as np
import os
import cv2 
import time

faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(3, 640)
cap.set(4, 640)

_none = input('\n [Info] Press Enter for face-image capturing..')

img_folder_path = '/home/jay-bhagiya/Codes/mini-project/final-codes/face_images/'
test_folder_path = img_folder_path + 'testing/'
train_folder_path = img_folder_path + 'training/'

face_id = len(list(os.walk(train_folder_path)))

new_folder_train_path = train_folder_path + 'face{}'.format(face_id)
if not os.path.exists(new_folder_train_path):
	os.makedirs(new_folder_train_path)

new_folder_test_path = test_folder_path + 'face{}'.format(face_id)
if not os.path.exists(new_folder_test_path):
	os.makedirs(new_folder_test_path)

print('\n [Info] Initializing face capture. Look the camera and wait ...')

image_count = 0

while True:
	ret, frame = cap.read()
	img = cv2.flip(frame, 1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.2,
		minNeighbors=5,
		minSize=(20, 20)
	)

	for (x,y,w,h) in faces:
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

		image_count += 1

		if image_count <= 5:
			cv2.imwrite(new_folder_test_path + '/{}face{}.jpg'.format(image_count, face_id), roi_color)
			time.sleep(2)
		else:
			cv2.imwrite(new_folder_train_path + '/{}face{}.jpg'.format(image_count - 5, face_id), roi_color)

		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

	cv2.imshow('Video', img)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
	elif image_count >= 20:
		break

print('\n [Info] Image extraction completed..')
cap.release()
cv2.destroyAllWindows()
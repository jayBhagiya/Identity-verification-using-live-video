from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import pickle
import os
import cv2 

font = cv2.FONT_HERSHEY_SIMPLEX
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(3, 640)
cap.set(4, 640)

_none = input('\n [Info] Press Enter for face-image capturing..')

with open('ResultMap.pkl', 'rb') as fileReadStream:
	ResultMap = pickle.load(fileReadStream)

model = load_model('model.h5')

def predict_face(frame):
	test_image = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_NEAREST)
	test_image = np.expand_dims(test_image, axis=0)

	result = model.predict(test_image, verbose=0)

	return ResultMap[np.argmax(result)]

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

	if len(faces):
		for (x,y,w,h) in faces:
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]

			face_id = predict_face(roi_color)

			cv2.putText(img, str(face_id), (x+5, y-5), font, 1, (255, 255, 255), 2)
			cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

	cv2.imshow('Video', img)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()


# ImagePath = '/home/jay-bhagiya/Codes/mini-project/final-codes/face_images/testing/face8/2face8.jpg'
# test_image = image.load_img(ImagePath,target_size=(64, 64))
# test_image = image.img_to_array(test_image)

# test_image = np.expand_dims(test_image,axis=0)

# model = load_model('model.h5')

# result = model.predict(test_image,verbose=0)

# print('####'*10)
# print('Prediction is: ',ResultMap[np.argmax(result)])
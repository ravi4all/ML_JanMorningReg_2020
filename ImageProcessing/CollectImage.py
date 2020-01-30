import cv2
import numpy as np

dataset = cv2.CascadeClassifier('data.xml')
capture = cv2.VideoCapture(0)

faceList = []

while True:
    ret,img = capture.read()
    if ret:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray, 1.2)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face,(50,50))
            if len(faceList) < 200:
                faceList.append(face)
                print(len(faceList))

        cv2.imshow('result',img)
        if cv2.waitKey(1) == 27 or len(faceList) >= 200:
            break
    else:
        print("Camera not working")

faceList = np.asarray(faceList)
np.save('face_1.npy',faceList)

cv2.destroyAllWindows()
capture.release()

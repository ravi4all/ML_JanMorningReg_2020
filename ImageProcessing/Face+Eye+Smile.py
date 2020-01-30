import cv2

face_data = cv2.CascadeClassifier('data.xml')
smile_data = cv2.CascadeClassifier('haarcascade_smile.xml')
eyes_data = cv2.CascadeClassifier('haarcascade_eye.xml')

capture = cv2.VideoCapture(0)

while True:
    ret,img = capture.read()
    if ret:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_data.detectMultiScale(gray, 1.2)
        eyes = eyes_data.detectMultiScale(img)
        smile = smile_data.detectMultiScale(img)
        # print(eyes)
        for x,y,w,h in eyes:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        # for x,y,w,h in smile:
        #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 125, 55), 2)
        cv2.imshow('result',img)
        if cv2.waitKey(1) == 27:
            break
    else:
        print("Camera not working")

cv2.destroyAllWindows()
capture.release()

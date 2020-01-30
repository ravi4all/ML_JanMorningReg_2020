import cv2

face_data = cv2.CascadeClassifier('data.xml')
smile_data = cv2.CascadeClassifier('haarcascade_smile.xml')
eyes_data = cv2.CascadeClassifier('haarcascade_eye.xml')

capture = cv2.VideoCapture('video_1.mp4')

while True:
    ret,img = capture.read()
    img = cv2.resize(img,None,fx=0.6,fy=0.6)
    if ret:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_data.detectMultiScale(gray, 1.2)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

            myface = img[y:y+h, x:x+w, :]
            eyes = eyes_data.detectMultiScale(myface)
            smile = smile_data.detectMultiScale(myface,1.8)
            for ex, ey, ew, eh in eyes:
                cv2.rectangle(myface, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

            for sx, sy, sw, sh in smile:
                cv2.rectangle(myface, (sx, sy), (sx + sw, sy + sh), (100, 0, 255), 2)

        cv2.imshow('result',img)
        if cv2.waitKey(1) == 27:
            break
    else:
        print("Camera not working")

cv2.destroyAllWindows()
capture.release()

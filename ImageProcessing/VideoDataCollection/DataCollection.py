import cv2

dataset = cv2.CascadeClassifier('data.xml')
capture = cv2.VideoCapture('video_1.mp4')
i = 0
while True:
    i += 1
    ret,img = capture.read()
    img = cv2.resize(img,None,fx=0.5,fy=0.5)
    if ret:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray, 1.4)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            face = img[y:y+h,x:x+w,:]
            face = cv2.resize(face,(100,100))

            cv2.imwrite(f'img_{i}.png',face)

        cv2.imshow('result',img)
        if cv2.waitKey(1) == 27:
            break
    else:
        print("Camera not working")
cv2.destroyAllWindows()
capture.release()

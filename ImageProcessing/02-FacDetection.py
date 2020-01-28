import cv2

data = cv2.CascadeClassifier('data.xml')
img = cv2.imread('img_1.jpg')

while True:
    # img = cv2.resize(img,None,fx=0.5,fy=0.5)
    faces = data.detectMultiScale(img, 1.28)
    cv2.imshow('result', img)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)

    if cv2.waitKey(10) == 27:
        break
import cv2
import numpy as np

face_1 = np.load('face_1.npy')
face_1 = face_1.reshape((200,-1))

face_2 = np.load('face_2.npy')
face_2 = face_2.reshape((200,-1))

faceData = np.concatenate([face_1,face_2])

labels = np.zeros((faceData.shape[0],1))
labels[200:,:] = 1.0

users = {
    0 : "Ravi_1",
    1 : "Ravi_2"
}

faceData = faceData / 255

def distance(x1,x2):
    return np.sqrt(sum(x2 - x1)**2)

def knn(data,target,k=5):
    n = data.shape[0]
    dis = []
    for i in range(n):
        d = distance(data[i],target)
        dis.append(d)

    dis = np.asarray(dis)
    sorted_index = np.argsort(dis)
    lab = labels[sorted_index][:k]
    count = np.unique(lab,return_counts=True)
    outcome = count[0][np.argmax(count[1])]
    return outcome

dataset = cv2.CascadeClassifier('data.xml')
capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    ret,img = capture.read()
    if ret:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray, 1.2)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

            target = img[y:y+h,x:x+w,:]
            target = cv2.resize(target,(50,50))
            index = knn(faceData,target.flatten())
            text = users[int(index)]
            cv2.putText(img,text,(x,y),font,1,(255,255,0),2)

        cv2.imshow('result',img)
        if cv2.waitKey(1) == 27:
            break
    else:
        print("Camera not working")
cv2.destroyAllWindows()
capture.release()
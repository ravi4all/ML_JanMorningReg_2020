import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

one = np.load('one.npy')
one = one.reshape((400,-1))

five = np.load('five.npy')
five = five.reshape((400,-1))

faceData = np.concatenate([one,five])

labels = np.zeros((faceData.shape[0],1))
labels[400:,:] = 1.0

users = {
    0 : "One",
    1 : "Five"
}

svm = SVC()
svm.fit(faceData,labels)

predictions = svm.predict(faceData)
acc = accuracy_score(labels,predictions)
print("Score is",acc)

#faceData = faceData / 255

# def distance(x1,x2):
#     return np.sqrt(sum(x2 - x1)**2)
#
# def knn(data,target,k=5):
#     n = data.shape[0]
#     dis = []
#     for i in range(n):
#         d = distance(data[i],target)
#         dis.append(d)
#
#     dis = np.asarray(dis)
#     sorted_index = np.argsort(dis)
#     lab = labels[sorted_index][:k]
#     count = np.unique(lab,return_counts=True)
#     outcome = count[0][np.argmax(count[1])]
#     print(lab)
#     return outcome

# img = cv2.imread('images/five/img (1).png')
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img = cv2.resize(img, (50,50))
# img = img.reshape(1,-1)
# print(svm.predict(img))

capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
x = 40
y = 40
w = 200
h = 200
while True:
    ret,img = capture.read()
    if ret:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

        rect = img[y:y+h,x:x+w,:]
        gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
        target = cv2.resize(gray,(50,50))
        # index = knn(faceData,target.flatten())
        index = svm.predict(target.reshape(1,-1))
        print(index)
        text = users[int(index)]
        cv2.putText(img,text,(x,y),font,1,(255,255,0),2)

        cv2.imshow('result',img)
        if cv2.waitKey(1) == 27:
            break
    else:
        print("Camera not working")

cv2.destroyAllWindows()

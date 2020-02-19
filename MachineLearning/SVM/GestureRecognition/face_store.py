import cv2
import os, random
import numpy as np

dataList = []
one_img = os.listdir('images/five')

while True:
    imgName = random.choice(one_img)
    img = cv2.imread('images/five/'+imgName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_img = cv2.resize(gray, (50, 50))
    if len(dataList) < 400:
        dataList.append(gray_img)
        print(len(dataList))
    if len(dataList) >= 400:
        break

dataList = np.asarray(dataList)
np.save('five.npy', dataList)

cv2.destroyAllWindows()

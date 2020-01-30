import cv2
import numpy as np
import matplotlib.pyplot as plt

faceList = np.load('face_1.npy')
faceList = faceList.reshape((200,-1))
img = faceList[100]
# print(faceList.shape)
# print(img.shape)

plt.imshow(img.reshape((50,50,3)))
plt.show()
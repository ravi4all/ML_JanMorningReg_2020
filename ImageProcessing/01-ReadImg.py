import cv2
# import matplotlib.pyplot as plt

img = cv2.imread('img_1.jpg')
# print(img)
# plt.imshow(img)
# plt.show()

while True:
    cv2.imshow('result',img)
    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()
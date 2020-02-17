Python 3.7.3 (v3.7.3:ef4ec6ed12, Mar 25 2019, 22:22:05) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> from sklearn.datasets import make_moons
>>> data, target = make_moons(400)
>>> data.shape
(400, 2)
>>> plt.scatter(data[:,0],data[:,1],c=target)
<matplotlib.collections.PathCollection object at 0x000002B99D5770F0>
>>> plt.show()
>>> data, target = make_moons(400,noise=0.2)
>>> plt.scatter(data[:,0],data[:,1],c=target)
<matplotlib.collections.PathCollection object at 0x000002B99D74F9E8>
>>> plt.show()
>>> from sklearn.datasets import make_blobs
>>> data, target = make_blobs(400)
>>> plt.scatter(data[:,0],data[:,1],c=target)
<matplotlib.collections.PathCollection object at 0x000002B99E0F5AC8>
>>> plt.show()
>>> data, target = make_moons(400,noise=0.4)
>>> plt.scatter(data[:,0],data[:,1],c=target)
<matplotlib.collections.PathCollection object at 0x000002B99E0BC2B0>
>>> plt.show()
>>> x1 = np.array([3,4,6,7,2])
>>> x2 = np.array([5,8,1,3,4,7])
>>> xx,yy = np.meshgrid(x1,x2)
>>> xx
array([[3, 4, 6, 7, 2],
       [3, 4, 6, 7, 2],
       [3, 4, 6, 7, 2],
       [3, 4, 6, 7, 2],
       [3, 4, 6, 7, 2],
       [3, 4, 6, 7, 2]])
>>> yy
array([[5, 5, 5, 5, 5],
       [8, 8, 8, 8, 8],
       [1, 1, 1, 1, 1],
       [3, 3, 3, 3, 3],
       [4, 4, 4, 4, 4],
       [7, 7, 7, 7, 7]])
>>> z = xx ** 2 + yy ** 2
>>> plt.contour(xx,yy,z)
<matplotlib.contour.QuadContourSet object at 0x000002B99C9B9A20>
>>> plt.show()
>>> plt.contourf(xx,yy,z)
<matplotlib.contour.QuadContourSet object at 0x000002B99E2AA6A0>
>>> plt.show()
>>> xx1 = np.arange(min(x1) - 1, max(x1) + 1, 0.01)
>>> xx2 = np.arange(min(x2) - 1, max(x2) + 1, 0.01)
>>> xx1.shape
(700,)
>>> xx2.shape
(900,)
>>> xx,yy = np.meshgrid(xx1,xx2)
>>> xx.shape
(900, 700)
>>> yy.shape
(900, 700)
>>> z = xx ** 2 + yy ** 2
>>> plt.contourf(xx,yy,z)
<matplotlib.contour.QuadContourSet object at 0x000002B99E450278>
>>> plt.show()
>>> plt.contour(xx,yy,z)
<matplotlib.contour.QuadContourSet object at 0x000002B99F709F60>
>>> plt.show()
>>> z.shape
(900, 700)
>>> xx.shape
(900, 700)
>>> 

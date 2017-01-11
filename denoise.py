import numpy as np
import matplotlib.pyplot as plt
import cv2
from OD_setup import *
from denoise_setup import *

D = np.load('D_cifar_1.npy')

M,K = D.shape
lamda = 1
sigma = 30
mu = 30/sigma
e = M*(1.15*sigma)**2
x_list = []
x = cv2.imread('cat.jpg', 0)
x = np.array(x,float)

N = int(x.shape[0]**2)

x =  x.reshape(N, 1)
x_0 = np.array(x)
x_list.append(x_0)
np.random.seed(0)
x += np.random.normal(0,sigma,(N,1))
x_noisy = np.array(x)
x_list.append(x_noisy)

R_list = R(N,M)
RTR = np.zeros((N,N))
for i in range(len(R_list)):
    RTR += R_list[i].T @ R_list[i]

inv = np.linalg.inv(mu*np.identity(N) + RTR)

for i in range(10):
    print(i)
    alpha_list = alpha_patch(lamda,x,D,100,e,R_list)
    x = det_X(mu,x,D,alpha_list,R_list,inv)

x_list.append(x)
denoise_show(x_list)
plt.show()

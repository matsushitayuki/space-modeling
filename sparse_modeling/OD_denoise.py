import numpy as np
import matplotlib.pyplot as plt
import cv2
from OD_setup import *
from denoise_setup import *

D = np.load('D_cifar_1.npy')
stride = 16

M,K = D.shape
n = int(M**0.5)

lamda = 0.05
sigma = 0.05
mu = 0.005*30/sigma
e = M*(1.15*sigma)**2
X_list = []
X = cv2.imread('barbara.jpg', 0)
X = np.array(X,float)/255

N = int(X.shape[0])


X_0 = np.array(X)
X_list.append(X_0)
np.random.seed(0)
X += np.random.normal(0,sigma,(N,N))
X_noisy = np.array(X)
X_list.append(X_noisy)

Y = X_noisy.reshape(N**2,1)

RTR = R_T_R(N,n,stride)
RTR += mu*np.ones((N**2,1))
inv = 1/RTR

for i in range(10):
    R_T_Dalpha = np.zeros((N,N))
    A = np.zeros((K,K))
    B = np.zeros((M,K))
    for i in np.arange(0,(N-n+1),stride):
        for j in np.arange(0,(N-n+1),stride):
            print(i,j)
            RX = R(X,i,j,n)
            RX = RX.reshape(n**2,1)
            alpha = alpha_line(lamda,RX,D,50,e)
            Dalpha = (D@alpha).reshape(n,n)
            R_T_Dalpha += R_T(Dalpha,i,j,N)
            A += alpha@alpha.T
            B += RX@alpha.T
            D = Dict_update(D,A,B)

X = X.reshape(N**2,1)
R_T_Dalpha = R_T_Dalpha.reshape(N**2,1)
X = inv * (mu*Y + R_T_Dalpha)
X = X.reshape(N,N)
X_list.append(X)

for i in range(len(X_list)):
    plt.subplot(1,3,i+1)
    plt.imshow(X_list[i],vmin=0,vmax=1)
    plt.gray()
    plt.axis('off')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from OD_setup import *

def R(N,n):
    R_0 = np.zeros((n,N))
    p = int(N**0.5 - n**0.5 + 1)
    R_init = [int(N**0.5*i + j) for i in range(p) for j in range(p)]
    R_list = []
    for r in R_init:
        R = np.array(R_0)
        R_element = [int(r + N**0.5*i + j) for i in range(int(n**0.5)) for j in range(int((n**0.5)))]
        for i in range(n):
            R[i,R_element[i]] = 1
        R_list.append(R)
    return R_list

def alpha_line(lamda,x,D,T,e):
    alpha = alpha_ADMM(lamda,x,D,T)
    while np.linalg.norm(x - D@alpha)**2 > e:
        lamda = 0.8*lamda
        alpha = alpha_ADMM(lamda,x,D,T)
    return alpha

def alpha_patch(lamda,X,D,T,e,R_list):
    alpha_list = []
    i = 1
    for r in R_list:
        x = r@X
        alpha_r = alpha_line(lamda,x,D,T,e)
        alpha_list.append(alpha_r)
        i += 1

    return np.array(alpha_list)

def det_X(mu,X,D,alpha_list,R_list,inv):
    RDalpha = np.zeros((R_list[1].shape[1],1))
    for i in range(len(R_list)):
        RDalpha += R_list[i].T @ D @ alpha_list[i]
    X = inv @ (mu*X + RDalpha)
    return X

def denoise_show(x_list):
    for i in range(len(x_list)):
        x_t = x_list[i].reshape(int(x_list[i].shape[0]**0.5),int(x_list[i].shape[0]**0.5))
        plt.subplot(1,3,i+1)
        plt.imshow(x_t)
        plt.gray()

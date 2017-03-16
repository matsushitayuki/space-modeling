import numpy as np
import matplotlib.pyplot as plt
import cv2

def cost(lamda,x,D,alpha):
    cost_F = 0.5*np.linalg.norm(x - D@alpha)**2
    cost_L = sum(abs(alpha))[0]
    cost =  cost_F + lamda*cost_L
    return cost,cost_F,cost_L

def S(lamda,y):
    y_copy = np.array(y)
    for i in range(len(y_copy)):
        if y_copy[i] > lamda:
            y_copy[i] -= lamda
        elif y_copy[i] < -lamda:
            y_copy[i] += lamda
        else:
            y_copy[i] = 0
    return y_copy

def alpha_ADMM(lamda,x,D,T):
    K = D.shape[1]
    alpha1 = np.zeros((K,1))
    alpha2 = np.zeros((K,1))
    u = np.zeros((K,1))
    inv = np.linalg.inv(D.T@D + np.identity(K))
    for i in range(T):
        alpha1 = inv@(D.T@x +(alpha2 - u))
        alpha2 = S(lamda,alpha1 + u)
        u = u + alpha1 - alpha2
    return alpha2

def Dict_update(D,A,B):
    D_copy = np.array(D)
    K = D_copy.shape[1]
    for i in range(K):
        if A[i,i] != 0:
            D_copy[:,i] += (B[:,i] - D_copy@A[:,i])/A[i,i]
            D_copy[:,i] = D_copy[:,i] / max(np.linalg.norm(D_copy[:,i]),1)
    return D_copy

def ODL(lamda,x,D,A,B,T=100):
    alpha = alpha_ADMM(lamda,x,D,T)
    A += alpha@alpha.T
    B += x@alpha.T
    D = Dict_update(D,A,B)
    return D,alpha,A,B

def cost_D(lamda,x,D):
    i = len(x)
    temp_D = 0
    temp_D_F = 0
    temp_D_L = 0
    temp_list = []
    for xi in x:
        alpha_D = alpha_ADMM(lamda,xi,D,100)
        cost_D,cost_D_F,cost_D_L = cost(lamda,xi,D,alpha_D)
        temp_D += cost_D
        temp_D_F += cost_D_F
        temp_D_L += cost_D_L
    return temp_D/i,temp_D_F/i,temp_D_/i

def gene_sample(d):
    d = np.array(d).reshape(3,32,32).transpose(1,2,0)
    d = cv2.cvtColor(d,cv2.COLOR_RGB2GRAY)
    d = d/255
    d = d.reshape(1024,1)
    return d

def pic_show(D,n):
    from math import ceil
    import random
    size = ceil(n**0.5)
    for i in range(n):
        a = random.randint(0,D.shape[1]-1)
        plt.subplot(size,size,i+1)
        d = D[:,a].reshape(32,32)
        plt.imshow(d)
        plt.gray()
        plt.axis('off')

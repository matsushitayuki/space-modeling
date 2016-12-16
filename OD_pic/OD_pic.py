import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
from OD_setup import *

def gene_sample(d):
    d = np.array(d).reshape(3,32,32).transpose(1,2,0)
    d = cv2.cvtColor(d,cv2.COLOR_RGB2GRAY)
    d = d/255
    d = d.reshape(1024,1)
    return d

def pic_show(D):
    n = 20
    for i in range(n):
        plt.subplot(5,4,i+1)
        d = D[:,i].reshape(32,32)
        plt.imshow(d)
        plt.gray()
        plt.axis('off')
M = 1024
K = 250

lamda = 1.2/M**0.5

with open('data_batch_1','rb') as fo:
    d = pickle.load(fo,encoding='latin-1')

T = 5000

D = np.random.rand(M,K)
with open('data_batch_4','rb') as fo_D:
    d_D = pickle.load(fo_D,encoding='latin-1')

for i in range(K):
    d_D_i = gene_sample(d_D['data'][i]).reshape(1024,)
    D[:,i] = d_D_i

D_0 = np.array(D)

A = np.zeros((K,K))
B = np.zeros((M,K))

for i in range(T):
    print(i)
    x = gene_sample(d['data'][i])
    D,alpha,A,B = ODL(lamda,x,D,A,B)

np.save('D_0_pic.npy',D_0)
np.save('D_pic.npy',D)

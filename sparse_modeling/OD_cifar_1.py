import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
from OD_setup import *

M = 1024
K = 2048

lamda = 1.2/M**0.5

with open('data_batch_1','rb') as fo:
    d1 = pickle.load(fo,encoding='latin-1')

T = len(d1['data'])

D = np.random.rand(M, K)

D_0 = np.array(D)

A = np.zeros((K,K))
B = np.zeros((M,K))


for i in range(T):
    print("1",i)
    x = gene_sample(d1['data'][i])    
    D,alpha,A,B = ODL(lamda,x,D,A,B)

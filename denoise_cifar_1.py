import numpy as np
import matplotlib.pyplot as plt
import pickle
from OD_setup import *

def det_x(lamda,y,D,alpha,mu):
    return (mu*y + D@alpha)/(1+mu)

def cost_denoise(lamda,x,D,alpha,x_0,mu):
    return 0.5*np.linalg.norm(x_0 - D@alpha)**2 + lamda*sum(abs(alpha)) + mu/2*np.linalg.norm(x_0 - x)**2

D = np.load('D_cifar_1.npy')

M,K = D.shape
lamda = 1.2/M**0.5
rho = 0.1
mu = 30/rho

T = 200

with open('data_batch_2','rb') as fo:
    d = pickle.load(fo,encoding='latin-1')

x_list = []

x = gene_sample(d['data'][0])
x_0 = np.array(x)
x_list.append(x_0)
x += np.random.normal(0,rho,(M,1))
x_noisy = np.array(x)
x_list.append(x_noisy)

alpha = alpha_ADMM(lamda,x,D,100)
ktemp = cost_denoise(lamda,x_0,D,alpha,x_0,mu)
print(ktemp)
temp = 10000
i =1

while abs(temp - ktemp) > 0.1:
    ktemp = temp
    print(i)
    i += 1
    alpha = alpha_ADMM(lamda,x,D,100)
    x = det_x(lamda,x,D,alpha,mu)
    temp = cost_denoise(lamda,x,D,alpha,x_0,mu)
    print(temp)

x_list.append(x)

for i in range(len(x_list)):
    x_t = x_list[i].reshape(32,32)
    plt.subplot(1,3,i+1)
    plt.imshow(x_t)
    plt.gray()

plt.show()

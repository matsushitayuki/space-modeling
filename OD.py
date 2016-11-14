import numpy as np
import matplotlib.pyplot as plt

M = 256
#a number of dimension of sample

K = 200
#a number of basis of dictionary
#if M < K,it's overcomplete Dictionary

D = np.random.randn(M,K)
#ititial Ditionary
A = np.zeros((K,K))
B = np.zeros((M,K))
#used for Dictionary update

T = 1000000
#time to iterate

def S(y,lamda):
    y_copy = np.array(y)
    for i in range(len(y_copy)):
        if y_copy[i] > lamda:
            y_copy[i] = y_copy[i] - lamda
        elif y_copy[i] < -lamda:
            y_copy[i] = y_copy[i] + lamda
        else:
            y_copy[i] = 0
    return y_copy
#define soft-thresholding function

def alpha_ADMM(x,D,M):
 alpha_1 = np.zeros((K,1))
 alpha_2 = np.zeros((K,1))
 lamda = 1.2/M**0.5
 inv = np.linalg.inv(np.identity(K)+D.T@D/lamda)
 u = np.zeros((K,1))
 for i in range(100):
  kalpha_1 = alpha_1
  kalpha_2 = alpha_2
  alpha_1 = inv @ (D.T@x/lamda + alpha_2 -u)
  alpha_2 = S(alpha_1+u,1)
  u = u + kalpha_1 - kalpha_2

 return alpha_2
#define basis pursuit function(using ADMM)
def Algorithm2(D,A,B):
 for i in range(K):
  if A[i,i] != 0:
   D[:,i] += (B[:,i] - D@A[:,i])/A[i,i]
   D[:,i] = D[:,i] / max(np.linalg.norm(D[:,i]),1)
  else:
   D[:,i] = B[:,i] - D@A[:,i]
   D[:,i] = D[:,i] / max(np.linalg.norm(D[:,i]),1)
 return D
#used for Dictionary update

temp = 0
f = []
f_t = 0

for i in range(T):
 kf_t = f_t
 i_time = i+1
 x = np.random.normal(0,1,(M,1))
 
 #generate sample
 alpha_t = alpha_ADMM(x,D,M)
 #basis pursuit (alpha is coefficient vector)

 A += alpha_t @ alpha_t.T
 B += x @ alpha_t.T

 temp_t = 0.5 * np.linalg.norm(x-D@alpha_t)**2
 temp += temp_t
 f_t = temp/i_time
 f_t_dif = kf_t - f_t
 print(i_time,f_t,f_t_dif)
 f.append(f_t)
 #calculate loss function

 D = Algorithm2(D,A,B)
 # Dictionary update

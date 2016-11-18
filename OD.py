import numpy as np
import matplotlib.pyplot as plt
import random

M = 2

K = 2

lamda = 1.2 / M**(1/2)


D = np.random.randn(M,K)
for i in range(K):
 D[:,i] = D[:,i]/max(np.linalg.norm(D[:,i]),1.0)
D_0 = np.array(D)

A = np.zeros((K,K))
B = np.zeros((M,K))

T = 1

def calc_cost(x,D,alpha):
 y_ = 0.5*np.linalg.norm(x-D@alpha)**2 + lamda*sum(abs(alpha))
 return y_

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

def alpha_ADMM(x,D):
 alpha_1 = np.zeros((K,1))
 alpha_2 = np.zeros((K,1))
 inv = np.linalg.inv(D.T@D + np.identity(K))
 u = np.zeros((K,1))
 for i in range(100):
  kalpha_1 = alpha_1
  kalpha_2 = alpha_2
  alpha_1 = inv @ (D.T@x + alpha_2 -u)
  alpha_2 = S(alpha_1+u,lamda)
  u = u + alpha_1 - alpha_2
  print(alpha_2)
 return alpha_2

def Algorithm2(D,A,B):
 for ite in range(100):
  for i in range(K):
   if A[i,i] != 0:
    D[:,i] += (B[:,i] - D@A[:,i])/A[i,i]
    D[:,i] = D[:,i] / max(np.linalg.norm(D[:,i]),1)
   else:
    D[:,i] = D[:,i] / max(np.linalg.norm(D[:,i]),1)
 return D

temp_opt = 0
temp_0 = 0
f_opt = []
f_0 = []
x1 = np.array([[1],[0]])
x2 = np.array([[0],[1]])

x_set = [x1,x2]

for i in range(1,T+1):
 x = random.choice(x_set)
 a = np.random.normal(0,0.5,(M,1))
 x = x + a

 alpha_opt_t = alpha_ADMM(x,D)
 alpha_0_t = alpha_ADMM(x,D_0)

 A += alpha_opt_t @ alpha_opt_t.T
 B += x @ alpha_opt_t.T

 D = Algorithm2(D,A,B)

 temp_opt_t = 0.5 * np.linalg.norm(x-D@alpha_opt_t)**2 + lamda*sum(abs(alpha_opt_t))
 temp_opt_t_1 = 0.5 * np.linalg.norm(x-D@alpha_opt_t)**2
 temp_opt_t_2 =  lamda * sum(abs(alpha_opt_t))
 temp_0_t = 0.5 * np.linalg.norm(x-D_0@alpha_0_t)**2 + lamda*sum(abs(alpha_0_t))
 temp_0_t_1 = 0.5 * np.linalg.norm(x-D_0@alpha_0_t)**2
 temp_0_t_2 = lamda * sum(abs(alpha_0_t))
 temp_opt += temp_opt_t
 temp_0 += temp_0_t
 f_opt_t = temp_opt/i
 f_opt.append(f_opt_t)
 f_0_t = temp_0/i
 f_0.append(f_0_t)
 print(f_opt_t <= f_0_t)
 print(temp_opt_t_1,temp_opt_t_2,temp_0_t_1,temp_0_t_2)
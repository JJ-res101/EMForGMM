# -*- coding: utf-8 -*-
"""
Created on Sun May 24 21:31:26 2020

@author: jc-chen97
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
 data:input data 
   N*C, N is sample number, C is sample's channel
 mu:The means of K distributions
   C*K, C is sample's channel,K is number of Gaussian distributions
 delta:The covariances of K distributions
   C*C*K,
 prior:
   1*K,
 responds:
   N*K,
'''
# 1.load data 
# Take pictures, for example
data = cv2.imread('test.bmp',0)
data = cv2.GaussianBlur(data,(21,21),1)
hist = cv2.calcHist([data],[0],None,[256],[0,256])
data = data/255
h,w = data.shape[:2]
data = data.reshape(h*w,-1)
N,C = data.shape                    #Just make sure that the size of the data is N*C

# 2.Initialize parameter
K = 3                              # The number of Gaussian distributions
mu = np.zeros((C,K))                # The means of K distributions
delta = np.zeros((C,C,K))           # The covariances of K distributions
iter_num = 500                      # Maximum number of iterations
np.random.seed(0)
for k in range(K):
    # initializes the parameters of the K group distribution
    mu[:,k] = np.random.normal(loc=0, scale= np.sqrt(2. / K), size=(C,1)).reshape(-1,)
    delta[:,:,k] = np.identity(C)
prior = np.ones((1,K))/K            # Prior probability
responds = np.zeros((N,K))          # responsibilities
# A special form of the gaussian. The implement of parallel computing.
Gaussian_kernel= lambda x,mu,delta_inv: np.exp(-0.5*np.sum((x-mu).T*np.dot(delta_inv,(x-mu)).T,1))
likelihood = 0
likelihood_last = 0
# main loop
for i in range(iter_num):
    # E step
    print('Now we are in E step')
    for k in range(K):
        # For kth distribution
        delta_det = np.linalg.det(delta[:,:,k])+1e-6
        delta_inv = np.linalg.inv(delta[:,:,k]+1e-6*np.identity(C))
        u = np.expand_dims(mu[:,k],axis=1)
        responds[:,k] = Gaussian_kernel(data.T,u,delta_inv)/ delta_det**0.5
    responds = responds * prior 
    
    likelihood_last = likelihood
    likelihood = np.log(np.sum(responds,1)).sum()
    if abs(likelihood-likelihood_last)<0.01:
        print('It is enough!')
        break       
    responds = responds/responds.sum(1,keepdims = True)     
    # M step
    print('Now we are in M step')
    print('likelihood is %.3f'%likelihood)
    N_k = responds.sum(0,keepdims=True)          #(1,K)
    mu = np.dot(responds.T,data).T/N_k           #(K*N x N*C).T = C*K
    prior = N_k/N                                #re-estimae prior
    for k in range(K):
        u = np.expand_dims(mu[:,k],axis=1)       #re-estimae means
        #re-estimae covariances
        delta[:,:,k] = np.dot((data.T-u),((data.T-u).T*responds[:,k].reshape(-1,1)))/N_k[0,k] #C*N x (N*C·N*1) = C*C


plt.close('all')
plt.plot(hist)
plt.axvline(mu[0][0]*255,color='red')
plt.axvline(mu[0][1]*255,color='red')
plt.axvline(mu[0][2]*255,color='red')



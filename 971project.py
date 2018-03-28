from numpy import *
import numpy as np
from scipy.stats import norm
from __future__ import division
from sklearn import linear_model

S0=36
r=0.06
K=40
sigma=0.2
T=1
M=1000
N=51
deltat=T/(N-1)
discount=np.exp(-r*deltat)

def samplepath(S0,r,K,sigma,T,M,N):
    deltat=T/N
    S=np.zeros((M,N))
    S=mat(S)
    S[:,0]=S0
    m=M//2

    for i in range(1,N):
        bm=np.random.randn(m,1)
        S[0:m,i]=multiply(S[0:m,i-1],np.exp((r-0.5*sigma**2)*deltat+sigma*np.sqrt(deltat)*bm))
        S[m:,i]=multiply(S[m:,i-1],np.exp((r-0.5*sigma**2)*deltat+sigma*np.sqrt(deltat)*(-bm)))
        
    return S
# how to use other variance reduction method?

def leastsquare(x,y):
   #x and y are both M*1 column vector
    dimension=x.shape
    M=dimension[0]
    x_nonzero=x[x>0]
    y_nonzero=y[x>0]
    x1=np.exp(-x_nonzero/2)
    x2=np.exp(-x_nonzero/2)*(1-x_nonzero)
    x3=np.exp(-x_nonzero/2)*(1-2*x_nonzero+x_nonzero**2/2)
    x_mat=np.vstack((x1,x2,x3)).T
    model=linear_model.LinearRegression()
    model.fit(x_mat,y_nonzero)
    model_prediction=np.zeros((M,1))
    for i in range(0,M):
        if x[i,0]==0:
            continue
        else:
            a=x[i,0]
            a1=np.exp(-a/2)
            a2=np.exp(-a/2)*(1-a)
            a3=np.exp(-a/2)*(1-2*a+a**2/2)
            a_mat=np.hstack((a1,a2,a3))
            model_prediction[i,0]=model.predict(a_mat.reshape(1, -1))
   
    return model_prediction
#Here we can use other basis function

#pricing American put with no dividends
S=samplepath(S0,r,K,sigma,T,M,N)

value=mat(np.zeros((M,N)))
value[:,N-1]=K-S[:,N-1]
value[:,N-1][value[:,N-1]<0]=0
choice=mat(np.zeros((M,N)))
choice[:,N-1][value[:,N-1]>0]=1

for i in range(N-1,1,-1):
    value[:,i-1]=K-S[:,i-1]
    value[:,i-1][value[:,i-1]<0]=0
    x=np.where(value[:,i-1]>0,S[:,i-1],0)
    y=np.where(value[:,i-1]>0,value[:,i],0)
    y=y*discount
    continuation=leastsquare(x,y)
    #continuation=testsquare(x,y)
    exercise=np.where(value[:,i-1]>0,value[:,i-1],0)
    choice[:,i-1]=np.where(exercise>continuation,1,0)

cashflow=np.zeros((M,1))    
choice=choice[:,1:]
value=value[:,1:]
for i in range(0,M):
    for j in range(0,N-1):
        if choice[i,j]==1:
            cashflow[i,0]=value[i,j]*np.exp(-(j+1)*r*deltat)
            break
price=cashflow.mean()  
se=cashflow.std()
price  
se
  


















   
    
    
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
M=10000
N=51
alpha=0.5

def samplepath(S0,r,K,sigma,T,M,N,alpha):
    deltat=T/N
    S=np.zeros((M,N))
    S=mat(S)
    S[:,0]=S0*np.exp(alpha*sigma*np.sqrt(T)*np.random.rand(M,1))
    m=M//2

    for i in range(1,N):
        bm=np.random.randn(m,1)
        S[0:m,i]=multiply(S[0:m,i-1],np.exp((r-0.5*sigma**2)*deltat+sigma*np.sqrt(deltat)*bm))
        S[m:,i]=multiply(S[m:,i-1],np.exp((r-0.5*sigma**2)*deltat+sigma*np.sqrt(deltat)*(-bm)))
        
    return S
# how to use other variance reduction method?
'''
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
'''
def leastsquare(x,y):
   #x and y are both M*1 column vector
    dimension=x.shape
    M=dimension[0]
    x_nonzero=x[x>0]
    y_nonzero=y[x>0]
    x1=x_nonzero
    x2=x_nonzero**2
    x3=x_nonzero**3
    x4=x_nonzero**4
    x_mat=np.vstack((x1,x2,x3,x4)).T
    model=linear_model.LinearRegression()
    model.fit(x_mat,y_nonzero)
    x_pre=np.hstack((x,x**2,x**3,x**4))
    y_pre=model.predict(x_pre).reshape(M,1)
    model_prediction=np.where(x>0,y_pre,0)
            
    return model_prediction

#Here we can use other basis function

#pricing American put with no dividends
def put_calculation(S0,r,K,sigma,T,M,N,alpha):
    deltat=T/(N-1)
    discount=np.exp(-r*deltat)
    S=samplepath(S0,r,K,sigma,T,M,N,alpha)
    value=mat(np.zeros((M,N)))
    value[:,-1]=K-S[:,-1]
    value[:,-1][value[:,-1]<0]=0
    cashflow=mat(np.zeros((M,N)))
    cashflow[:,-1]=value[:,-1]

    for i in range(N-1,1,-1):
        value[:,i-1]=K-S[:,i-1]
        value[:,i-1][value[:,i-1]<0]=0
        x=np.where(value[:,i-1]>0,S[:,i-1],0)
        y=np.where(value[:,i-1]>0,cashflow[:,i]*discount,0)
        continuation=leastsquare(x,y)
        exercise=value[:,i-1]
        cashflow[:,i-1]=np.where(exercise>continuation,value[:,i-1],cashflow[:,i]*discount )
    y=cashflow[:,1]*discount
    x=S[:,0]
    x=x.getA()
    x1=x
    x2=x**2
    x3=x**3
    x4=x**4
    x_mat=np.hstack((x1,x2,x3,x4))
    model=linear_model.LinearRegression()
    model.fit(x_mat,y)
    s1=S0
    s2=S0**2
    s3=S0**3
    s4=S0**4
    s_mat=np.hstack((s1,s2,s3,s4))
    price=model.predict(s_mat.reshape(1, -1))
    para=model.coef_
    delta=para[0,0]+2*para[0,1]*S0+3*para[0,2]*S0**2+4*para[0,3]*S0**3
    gamma=2*para[0,1]+6*para[0,2]*S0+12*para[0,3]*S0**2
    
    return price,delta,gamma,para

def call_calculation(S0,r,K,sigma,T,M,N,alpha):
    deltat=T/(N-1)
    discount=np.exp(-r*deltat)
    S=samplepath(S0,r,K,sigma,T,M,N,alpha)
    value=mat(np.zeros((M,N)))
    value[:,-1]=S[:,-1]-K
    value[:,-1][value[:,-1]<0]=0
    cashflow=mat(np.zeros((M,N)))
    cashflow[:,-1]=value[:,-1]

    for i in range(N-1,1,-1):
        value[:,i-1]=S[:,i-1]-K
        value[:,i-1][value[:,i-1]<0]=0
        x=np.where(value[:,i-1]>0,S[:,i-1],0)
        y=np.where(value[:,i-1]>0,cashflow[:,i]*discount,0)
        continuation=leastsquare(x,y)
        exercise=value[:,i-1]
        cashflow[:,i-1]=np.where(exercise>continuation,value[:,i-1],cashflow[:,i]*discount )
    y=cashflow[:,1]*discount
    x=S[:,0]
    x=x.getA()
    x1=x
    x2=x**2
    x3=x**3
    x4=x**4
    x_mat=np.hstack((x1,x2,x3,x4))
    model=linear_model.LinearRegression()
    model.fit(x_mat,y)
    s1=S0
    s2=S0**2
    s3=S0**3
    s4=S0**4
    s_mat=np.hstack((s1,s2,s3,s4))
    price=model.predict(s_mat.reshape(1, -1))
    para=model.coef_
    delta=para[0,0]+2*para[0,1]*S0+3*para[0,2]*S0**2+4*para[0,3]*S0**3
    gamma=2*para[0,1]+6*para[0,2]*S0+12*para[0,3]*S0**2
    
    return price,delta,gamma,para








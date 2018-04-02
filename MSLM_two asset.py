from numpy import *
import numpy as np
from scipy.stats import norm
from __future__ import division
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

S10=40
S20=40
r=0.06
K=40
sigma1=0.2
sigma2=0.2
rho=0.2
T=1
M=10000
N=51
alpha1=0.5
alpha2=0.5

def samplepath(S10,S20,sigma1,sigma2,r,T,M,N,rho,alpha1,alpha2):
    deltat=T/(N-1)
    S1=np.zeros((M,N))
    S1=mat(S1)
    S2=np.zeros((M,N))
    S2=mat(S2)
    S1[:,0]=S10*np.exp(alpha1*sigma1*np.sqrt(T)*np.random.rand(M,1))
    S2[:,0]=S20*np.exp(alpha2*sigma2*np.sqrt(T)*np.random.rand(M,1))
    m=M//2

    for i in range(1,N):
        bm1=np.random.randn(m,1)
        S1[0:m,i]=multiply(S1[0:m,i-1],np.exp((r-0.5*sigma1**2)*deltat+sigma1*np.sqrt(deltat)*bm1))
        S1[m:,i]=multiply(S1[m:,i-1],np.exp((r-0.5*sigma1**2)*deltat+sigma1*np.sqrt(deltat)*(-bm1)))
        bm2=np.random.randn(m,1)
        bm3=rho*bm1+np.sqrt(1-rho**2)*bm2
        S2[0:m,i]=multiply(S2[0:m,i-1],np.exp((r-0.5*sigma2**2)*deltat+sigma2*np.sqrt(deltat)*bm3))
        S2[m:,i]=multiply(S2[m:,i-1],np.exp((r-0.5*sigma2**2)*deltat+sigma2*np.sqrt(deltat)*(-bm3)))
        
    return S1,S2

def leastsquare(x1,x2,y,value):
    dimension=x1.shape
    M=dimension[0]
    x1_nonzero=x1[x1>0]
    x1_nonzero=mat(x1_nonzero)
    x2_nonzero=x2[x2>0]
    x2_nonzero=mat(x2_nonzero)
    dataset=np.hstack((x1_nonzero.T,x2_nonzero.T))
    poly_reg=PolynomialFeatures(degree = 3)
    x_reg=poly_reg.fit_transform(dataset)
    y=y[value>0]
    y=y.T
    model=linear_model.LinearRegression()
    model.fit(x_reg,y)
    x1=mat(x1)
    x2=mat(x2)
    dataset1=np.hstack((x1,x2))    
    x_pre=poly_reg.fit_transform(dataset1)
    y_pre=model.predict(x_pre).reshape(M,1)
    model_prediction=np.where(value>0,y_pre,0)   
   
    return model_prediction 

def put_calculation(S10,S20,K,sigma1,sigma2,r,T,M,N,rho,alpha1,alpha2):
    deltat=T/(N-1)
    discount=np.exp(-r*deltat)
    [S1,S2]=samplepath(S10,S20,sigma1,sigma2,r,T,M,N,rho,alpha1,alpha2)
    value=mat(np.zeros((M,N)))
    maxS=np.where(S1[:,-1]>S2[:,-1],S1[:,-1],S2[:,-1])
    value[:,-1]=K-maxS
    value[:,-1][value[:,-1]<0]=0
    cashflow=mat(np.zeros((M,N)))
    cashflow[:,-1]=value[:,-1]

    for i in range(N-1,1,-1):
        maxS=np.where(S1[:,i-1]>S2[:,i-1],S1[:,i-1],S2[:,i-1])
        value[:,i-1]=K-maxS
        value[:,i-1][value[:,i-1]<0]=0
        x1=np.where(value[:,i-1]>0,S1[:,i-1],0)
        x2=np.where(value[:,i-1]>0,S2[:,i-1],0)
        y=cashflow[:,i]*discount       
        continuation=leastsquare(x1,x2,y,value[:,i-1])
        exercise=value[:,i-1]
        cashflow[:,i-1]=np.where(exercise>continuation,value[:,i-1],cashflow[:,i]*discount )
    y=cashflow[:,1]*discount
    x1=S1[:,0]
    x2=S2[:,0]
    dataset=np.hstack((x1,x2))
    poly_reg=PolynomialFeatures(degree = 3)
    x_reg=poly_reg.fit_transform(dataset)
    model=linear_model.LinearRegression()
    model.fit(x_reg,y)
    dataset1=np.hstack((S10,S20))    
    x_pre=poly_reg.fit_transform(dataset1.reshape(1, -1))
    price=model.predict(x_pre)
   
    return price

def call_calculation(S10,S20,K,sigma1,sigma2,r,T,M,N,rho,alpha1,alpha2):
    deltat=T/(N-1)
    discount=np.exp(-r*deltat)
    [S1,S2]=samplepath(S10,S20,sigma1,sigma2,r,T,M,N,rho,alpha1,alpha2)
    value=mat(np.zeros((M,N)))
    maxS=np.where(S1[:,-1]>S2[:,-1],S1[:,-1],S2[:,-1])
    value[:,-1]=maxS-K
    value[:,-1][value[:,-1]<0]=0
    cashflow=mat(np.zeros((M,N)))
    cashflow[:,-1]=value[:,-1]

    for i in range(N-1,1,-1):
        maxS=np.where(S1[:,i-1]>S2[:,i-1],S1[:,i-1],S2[:,i-1])
        value[:,i-1]=maxS-K
        value[:,i-1][value[:,i-1]<0]=0
        x1=np.where(value[:,i-1]>0,S1[:,i-1],0)
        x2=np.where(value[:,i-1]>0,S2[:,i-1],0)
        y=cashflow[:,i]*discount       
        continuation=leastsquare(x1,x2,y,value[:,i-1])
        exercise=value[:,i-1]
        cashflow[:,i-1]=np.where(exercise>continuation,value[:,i-1],cashflow[:,i]*discount )
    y=cashflow[:,1]*discount
    x1=S1[:,0]
    x2=S2[:,0]
    dataset=np.hstack((x1,x2))
    poly_reg=PolynomialFeatures(degree = 3)
    x_reg=poly_reg.fit_transform(dataset)
    model=linear_model.LinearRegression()
    model.fit(x_reg,y)
    dataset1=np.hstack((S10,S20))    
    x_pre=poly_reg.fit_transform(dataset1)
    price=model.predict(x_pre)
   
    return price

simulation=np.zeros((100,1))
for i in range(0,100):
    simulation[i,0]=put_calculation(40,40,40,0.2,0.2,r,1,M,N,rho,alpha1,alpha2)
np.mean(simulation)
np.std(simulation)









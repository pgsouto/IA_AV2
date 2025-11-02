import numpy as np
import matplotlib.pyplot as plt
from time import time

class MultilayerPerceptron:
    def __init__(self,X_train:np.ndarray, Y_train:np.ndarray, topology:list, learning_rate = 1e-3, tol = 1e-12, max_epoch = 10000):
        '''
        X_train (p x N)
        Y_train (C x N) ou (1 x N)
        '''
        
        self.p, self.N = X_train.shape
        self.m = Y_train.shape[0]
        self.X_train = np.vstack((
            -np.ones((1,self.N)), X_train
        ))
        self.D = Y_train
        
        self.tol = tol
        self.lr = learning_rate
        
        topology.append(self.m)
        print(topology)
        self.W = [None]*len(topology)
        Z = 0
        for i in range(len(self.W)):
            if i == 0:
                W = np.random.random_sample((topology[i],self.p+1))-.5
            else:
                W = np.random.random_sample((topology[i], topology[i-1]+1))-.5
            self.W[i] = W
            Z += W.size
        print(f"Rede MLP com {Z} parâmetros")
        self.y = [None]*len(topology)
        self.u = [None]*len(topology)
        self.delta = [None]*len(topology)
        self.max_epoch = max_epoch
    
    def g(self,u):
        return (1-np.exp(-u))/(1+np.exp(-u))
    
    def g_d(self, u):
        y = self.g(u)
        return .5*(1 - y**2)
    
    def forward(self,x):
        for i,W in enumerate(self.W):
            if i == 0:
                self.u[i] = W@x                
            else:
                yb = np.vstack((
                    -np.ones((1,1)), self.y[i-1]
                ))
                self.u[i] = W@yb
            self.y[i] = self.g(self.u[i])
    
    def predict(self, x):
        self.forward(x)
        MC = np.zeros((self.m,self.m))
        return self.y[-1]
    
    def EQM(self):
        s = 0
        for k in range(self.N):
            x_k = self.X_train[:,k].reshape(self.p+1,1)
            self.forward(x_k)
            y = self.y[-1]
            d = self.D[:,k].reshape(self.m,1)
            e = d - y
            s += np.sum(e**2)
            
        return s/(2*self.N)
            
    def backward(self,e,x):
        for i in range(len(self.W)-1,-1,-1):
            if i == len(self.W)-1:
                yb = np.vstack((
                   -1,
                    self.y[i-1]
                ))
                self.delta[i] = self.g_d(self.u[i]) * e
                self.W[i] = self.W[i] + self.lr*(self.delta[i]@yb.T)
            elif i == 0:
                Wnb = self.W[i+1][:,1:]
                self.delta[i] = self.g_d(self.u[i]) * (Wnb.T@self.delta[i+1])
                self.W[i] = self.W[i] + self.lr*(self.delta[i]@x.T)                
            else:
                yb = np.vstack((
                   -1,
                    self.y[i-1]
                ))
                Wnb = self.W[i+1][:,1:]
                self.delta[i] = self.g_d(self.u[i]) * (Wnb.T@self.delta[i+1])
                self.W[i] = self.W[i] + self.lr*(self.delta[i]@yb.T)
                
    
    def fit(self):
        epoch = 0
        EQM = self.EQM()
        print(f'EQM: {EQM:.15f}, época: {epoch}')
        while epoch < self.max_epoch and EQM > self.tol:
            t1 = time()
            for k in range(self.N):
                x_k = self.X_train[:,k].reshape(self.p+1,1)
                #Forward
                self.forward(x_k)
                y = self.y[-1]
                d = self.D[:,k].reshape(self.m,1)
                e = d - y
                #Backward
                self.backward(e,x_k)
            t2 = time()
            EQM = self.EQM()
            print(f'EQM: {EQM:.15f}, época: {epoch}, Tempo: {t2-t1:.5f}s')            
            epoch+=1
        
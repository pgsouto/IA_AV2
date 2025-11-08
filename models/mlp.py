import numpy as np
from time import time
class MLP:
    def __init__(self, X_treino, Y_treino, topology, learning_rate=1e-3, tol = 1e-12,max_epoch=10000):
        self.p, self.N = X_treino.shape
        self.m = Y_treino.shape[0]
        self.X_treino = np.vstack((
            -np.ones((1,self.N)), X_treino
        ))

        self.D = Y_treino

        self.tol = tol
        self.lr = learning_rate

        topology.append(self.m)

        self.y = [None] * len(topology)

        self.W = [None]*len(topology)

        for i in range(len(self.W)):
            if i == 0:
                W = np.random.random_sample((topology[i],self.p+1))-.5
            else:
                W = np.random.random_sample((topology[i], topology[i-1]+1))-.5
            self.W[i] = W

        self.y = [None]*len(topology)
        self.u = [None]*len(topology)
        self.delta = [None]*len(topology)
        self.max_epoch = max_epoch

        self.atingiu_epoca_maxima = False
        self.atingiu_EQM_minimo = False
        self.EQM_atual = 0


    def g(self,u):
        return (1-np.exp(-u))/(1+np.exp(-u))
    
    def g_d(self, u):
        y = self.g(u)
        return .5*(1 - y**2)


    def foward(self, x_t):
        for j in range(len(self.W)):
            if j == 0:
                self.u[j] = self.W[j]@x_t
            else:
                yb = np.vstack((
                    -np.ones((1,1)), self.y[j-1]
                ))
                self.u[j] = self.W[j]@yb
            self.y[j] = self.g(self.u[j])

    def backward(self, x_t, e):
        for j in range(len(self.W)-1, -1, -1):
            if j+1 == len(self.W):
                yb = np.vstack((
                   -1,
                    self.y[j-1]
                ))
                self.delta[j] = self.g_d(self.u[j])*e
                self.W[j] = self.W[j] + self.lr*self.delta[j]@yb.T
            elif j == 0:
                Wnb = self.W[j+1][:,1:]
                self.delta[j] = self.g_d(self.u[j]) * (Wnb.T@self.delta[j+1])
                self.W[j] = self.W[j] + self.lr*(self.delta[j]@x_t.T)
            else:
                yb = np.vstack((
                   -1,
                    self.y[j-1]
                ))
                Wb = self.W[j+1][:,1:].T
                self.delta[j] = self.g_d(self.u[j])
                self.delta[j] = self.g_d(self.u[j]) * (Wb@self.delta[j+1])
                self.W[j] = self.W[j] + self.lr*(self.delta[j]@yb.T)
       
        


    def EQM(self):
        s = 0
        for t in range(self.N):
            x_t = self.X_treino[:, t].reshape(self.p+1, 1)
            self.foward(x_t)
            y = self.y[-1]
            d = self.D[:, t].reshape(self.m,1)
            e = d - y
            s += np.sum(e**2) 
        return s / (2*self.N)
    
    def fit(self):
        for epoch in range(self.max_epoch):
            for t in range(self.N):
                x_t = self.X_treino[:, t].reshape(self.p+1, 1)
                self.foward(x_t)
                y = self.y[-1]
                d = self.D[:, t].reshape(self.m, 1)
                e = d - y
                self.backward(x_t, e)
            self.EQM_atual = self.EQM()
            #print(f"epoca = {epoch}, EQM = {self.EQM_atual}")
            if self.EQM_atual < self.tol:
                self.atingiu_EQM_minimo = True
                return
            


        self.atingiu_epoca_maxima = True
        
    def predict(self, x_t):
        x_t = x_t.reshape(self.p, 1)
        x_t = np.vstack((
            -np.ones((1,1)), x_t
        ))
        self.foward(x_t)
        y_t = self.y[-1]
        if y_t[0] > y_t[1]:
            return 1
        else:
            return -1

import numpy as np

class Adaline:
    def __init__(self, eta = 0.01, max_epochs=100, eps=1e-6, random_state = None):
        self.eta = eta
        self.max_epochs = max_epochs
        self.eps = eps
        self.random_state = random_state
        self.w = None
        self.eqm_list = []

    def fit(self, X, y):
        N, p = X.shape
        rng = np.random.default_rng(self.random_state)
        X = np.insert(X, 0, -1, axis=1) #aqui eu adiciono a coluna de interceptos de peso -1 a matriz de entradas
        self.w = rng.normal(scale=0.01, size=(p+1,))
        previous_eqm = np.inf

        for epoch in range(self.max_epochs):
            for i in range(N):
                xi = X[i]
                ui = np.dot(self.w, xi)
                ei = y[i] - ui
                self.w = self.w + self.eta * ei * xi
            
            u = X @ self.w
            eqm = np.mean((y - u)**2) / 2
            self.eqm_list.append(eqm)

            if abs(previous_eqm - eqm) <= self.eps:
                print(f"Treinamento de modelo adaline convergiu na época {epoch + 1}.")
                break
            previous_eqm = eqm

        else:
            print("Treinamento do modelo adaline atingiu o limite de épocas sem convergir.")
        return self
    
    def predict(self, X):
        X = np.insert(X, 0, -1, axis=1)
        u = X @ self.w
        y_pred = np.where(u >= 0, 1, -1)
        return y_pred

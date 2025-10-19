import numpy as np

class Perceptron:
    def __init__(self, eta = 0.01, max_epochs = 50):
        self.eta = eta
        self.max_epochs = max_epochs
        self.w = None
        self.errors_by_epoch = []

    def signal(self, u):
        """Função de ativação degrau"""
        return 1 if u >=0 else -1
    
    def fit(self, X, d):
        """Função que treina o perceptron com base na matriz de entradas X e vetor de rótulos d (resultado esperado)"""
        X = np.insert(X, 0, -1, axis=1) #aqui eu adiciono a coluna de interceptos de peso -1 a matriz de entradas
        self.w = np.zeros(X.shape[1]) #inicializo um vetor de pesos w com valores 0
        self.errors_by_epoch = []

        for epoch in range(self.max_epochs):
            total_error = 0
            for i in range(len(X)):
                u = np.dot(self.w, X[i])
                y = self.signal(u)
                e = d[i] - y
                self.w += self.eta * e * X[i]
                total_error += abs(e)
            self.errors_by_epoch.append(total_error)
            if total_error == 0:
                print(f"Treinamento do modelo perceptron convergiu na época {epoch + 1}.")
                break
        else:
            print("Treinamento do modelo perceptron atingiu o limite de épocas sem convergir.")

    def predict(self, X):
        "Função que classifica novas amostras"
        X = np.insert(X, 0, -1, axis=1)
        y_pred = []
        for x in X:
            u = np.dot(self.w, x)
            y_pred.append(self.signal(u))
        return np.array(y_pred)
    
    def score(self, X, d):
        """Calcula a acurácia do modelo"""
        y_pred = self.predict(X)
        return np.mean(y_pred == d)

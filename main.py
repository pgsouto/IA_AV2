import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from models.perceptron import Perceptron
from models.adaline import Adaline

# ===============================
# Funções auxiliares
# ===============================
def matriz_confusao(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == -1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))
    return np.array([[TN, FP], [FN, TP]])


#carregamento do dataset spiral_d.csv
data = np.loadtxt('./datasets/spiral_d.csv', delimiter=',')

#separação de informações do dataset
initial_X = data[:, :-1]
classes = data[:, -1]
entry_x1 = data[:, 0]
entry_x2 = data[:, 1]

#X normalizado
X_normalized = (initial_X - initial_X.mean(axis=0)) / initial_X.std(axis=0)

#seperação de classes para scatter plots
x1_classe0 = entry_x1[classes == -1]
x1_classe1 = entry_x1[classes == 1]

x2_classe0 = entry_x2[classes == -1]
x2_classe1 = entry_x2[classes == 1]

#Visualização do gráfico de espalhamento das entradas por classe
plt.figure(figsize=(6, 6))

plt.scatter(x1_classe0, x2_classe0, color="green", label="Classe -1")
plt.scatter(x1_classe1, x2_classe1, color="red", label="Classe 1")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Distribuição das classes do dataset spiral_d")
plt.legend()
plt.show()

#elaboração do perceptron

#--------------------------------------
#Valudação por método de Monte Carlo
#--------------------------------------

R = 500
accuracies_perceptron = []
accuracies_adaline = []

perceptron = Perceptron(eta=0.1, max_epochs=100)
adaline = Adaline(eta=0.01, max_epochs=200, eps=1e-6)

for montecarlo_round in range(R):
    #Eambaralhar dados
    permuted_indexes = np.random.permutation(len(X_normalized))
    X_permuted = X_normalized[permuted_indexes]
    y_permuted = classes[permuted_indexes]

    #Divisão em dados de treino e de teste
    split = int(0.8*len(X_permuted))

    X_train = X_permuted[:split]
    X_test = X_permuted[split:]

    y_train = y_permuted[:split]
    y_test = y_permuted[split:]

    #Treinamento do perceptron simples
    perceptron.fit(X_train, y_train)

    #Teste do percptron simples
    y_pred_perceptron = perceptron.predict(X_test)
    acc_perceptron = np.mean(y_pred_perceptron == y_test)

    accuracies_perceptron.append(acc_perceptron)

    #Treinamento do adaline
    adaline.fit(X_train, y_train)

    #teste do adaline
    y_pred_adaline = adaline.predict(X_test)
    acc_adaline = np.mean(y_pred_adaline == y_test)

    accuracies_adaline.append(acc_adaline)

    if montecarlo_round == 0:
        cm_perceptron = matriz_confusao(y_test, y_pred_perceptron)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm_perceptron, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        plt.title(f"Matriz de confusão (Perceptron) - rodada {montecarlo_round+1}")
        plt.show()

        cm_adaline = matriz_confusao(y_test, y_pred_adaline)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm_adaline, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        plt.title(f"Matriz de confusão Adaline - rodada {montecarlo_round+1}")
        plt.show()
# Estatísticas da acurácia
print("=== Estatísticas da acurácia: Perceptron ===")
print(f"Média: {np.mean(accuracies_perceptron):.3f}")
print(f"Desvio padrão: {np.std(accuracies_perceptron):.3f}")
print(f"Maior valor: {np.max(accuracies_perceptron):.3f}")
print(f"Menor valor: {np.min(accuracies_perceptron):.3f}")

print("=== Estatísticas da acurácia Adaline ===")
print(f"Média: {np.mean(accuracies_adaline):.3f}")
print(f"Desvio padrão: {np.std(accuracies_adaline):.3f}")
print(f"Maior valor: {np.max(accuracies_adaline):.3f}")
print(f"Menor valor: {np.min(accuracies_adaline):.3f}")

# Curva de aprendizado da última rodada
plt.figure()
plt.plot(range(1, len(perceptron.errors_by_epoch)+1), perceptron.errors_by_epoch, marker='o')
plt.xlabel("Épocas")
plt.ylabel("Erro total")
plt.title("Curva de aprendizado - última rodada")
plt.show()

plt.figure()
plt.plot(range(1, len(adaline.eqm_list)+1), adaline.eqm_list, marker='o')
plt.xlabel("Épocas")
plt.ylabel("EQM")
plt.title("Curva de aprendizado Adaline - última rodada")
plt.show()

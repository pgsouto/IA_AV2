import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tabulate import tabulate

from models.perceptron import Perceptron
from models.adaline import Adaline
from models.mlp import MLP
# ===============================
# Funções auxiliares
# ===============================
def create_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == -1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))
    return np.array([[TN, FP], [FN, TP]])

def calculate_measures(confusion_matrix):
    TN, FP = confusion_matrix[0,0], confusion_matrix[0,1]
    FN, TP = confusion_matrix[1,0], confusion_matrix[1,1]

    accuracy = (TP + TN) / (TP + TN + FP + FN) #acurácia
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0 #sensibilidade
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0 #especificidade
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0 #precisão
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return accuracy, recall, specificity, precision, f1_score

def show_measures_table(nome, acur, sensi, espec, prec, f1):
    headers = ["Métrica", "Média", "Desvio", "Mínimo", "Máximo"]
    table = [
        ["Acurácia", np.mean(acur), np.std(acur), np.min(acur), np.max(acur)],
        ["Sensibilidade", np.mean(sensi), np.std(sensi), np.min(sensi), np.max(sensi)],
        ["Especificidade", np.mean(espec), np.std(espec), np.min(espec), np.max(espec)],
        ["Precisão", np.mean(prec), np.std(prec), np.min(prec), np.max(prec)],
        ["F1-score", np.mean(f1), np.std(f1), np.min(f1), np.max(f1)]
    ]
    
    print(f"=== Métricas {nome} ===")
    print(tabulate(table, headers=headers, floatfmt=".3f"))
    print()

# ====================================
# Carregamento e tratamento dos dados 
# ====================================

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
x1_class0 = entry_x1[classes == -1]
x1_class1 = entry_x1[classes == 1]

x2_class0 = entry_x2[classes == -1]
x2_class1 = entry_x2[classes == 1]

#================================================================
#Visualização do gráfico de espalhamento das entradas por classe
#================================================================
'''plt.figure(figsize=(6, 6))

plt.scatter(x1_class0, x2_class0, color="green", label="Classe -1")
plt.scatter(x1_class1, x2_class1, color="red", label="Classe 1")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Distribuição das classes do dataset spiral_d")
plt.legend()
plt.show()'''

#=======================================
#Validação por método de Monte Carlo
#=======================================

#total de rodadas 
R = 1

#inicialização das listas que registrarão as métricas de desempenho para cada modelo 
accuracies_perceptron = []
recalls_perceptron = []
specificities_perceptron = []
precisions_perceptron = []
f1s_perceptron = []

accuracies_adaline = []
recalls_adaline = []
specificities_adaline = []
precisions_adaline = []
f1s_adaline = []

acuracies_mlp = []
recalls_mlp = []
specificities_mlp = []
precisions_mlp = []
f1s_mlp = []

#instancias dos modelos redes neurais implementados
perceptron = Perceptron(eta=0.1, max_epochs=100)
adaline = Adaline(eta=0.01, max_epochs=100, eps=1e-6)

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

    cm_perceptron = create_confusion_matrix(y_test, y_pred_perceptron)
    acc, sensi, espec, prec, f1 = calculate_measures(cm_perceptron)

    accuracies_perceptron.append(acc)
    recalls_perceptron.append(sensi)
    specificities_perceptron.append(espec)
    precisions_perceptron.append(prec)
    f1s_perceptron.append(f1)

    #Treinamento do adaline
    adaline.fit(X_train, y_train)

    #teste do adaline
    y_pred_adaline = adaline.predict(X_test)

    cm_adaline = create_confusion_matrix(y_test, y_pred_adaline)
    acc, sensi, espec, prec, f1 = calculate_measures(cm_adaline)
    
    accuracies_adaline.append(acc)
    recalls_adaline.append(sensi)
    specificities_adaline.append(espec)
    precisions_adaline.append(prec)
    f1s_adaline.append(f1)

    #Treinamento do MLP
    print(y_train.shape)
    n1 = np.sum(y_train[:]==1)
    n2 = np.sum(y_train[:]==-1)
    Y_MLP_train = np.zeros((2, 1120))

    for i in range(y_train.shape[0]):  
        if y_train[i] == 1:
            Y_MLP_train[0, i] = 1
            Y_MLP_train[1, i] = -1
        else:
            Y_MLP_train[0, i] = -1
            Y_MLP_train[1, i] = 1
        
    mlp = MLP(X_train.T, Y_MLP_train, [1000, 1000, 1000, 1000, 500, 250, 50], learning_rate=0.001, tol=1e-12, max_epoch=10)
    mlp.fit()
    print(f"Último EQM do MLP: {mlp.EQM_atual}")

    #Teste do MLP
    q_acertos = 0
    for t in range(y_test.shape[0]):
        y_t = mlp.predict(X_train.T[:, i])
        d_t = y_test[t]
        if y_t == d_t:
            q_acertos += 1
    print(f"ACURÁCIA MLP: {q_acertos / y_test.shape[0]}")

    
    #print(f"Acuracia MLP: {q_acertos / y_test.shape[0]}")
        

    '''if montecarlo_round == 0:
        cm_perceptron = create_confusion_matrix(y_test, y_pred_perceptron)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm_perceptron, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        plt.title(f"Matriz de confusão (Perceptron) - rodada {montecarlo_round+1}")
        plt.show()

        cm_adaline = create_confusion_matrix(y_test, y_pred_adaline)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm_adaline, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        plt.title(f"Matriz de confusão Adaline - rodada {montecarlo_round+1}")
        plt.show()'''
# Estatísticas da acurácia
show_measures_table("Perceptron", accuracies_perceptron, recalls_perceptron, specificities_perceptron, precisions_perceptron, f1s_perceptron)
show_measures_table("Adaline", accuracies_adaline, recalls_adaline, specificities_adaline, precisions_adaline, f1s_adaline)


'''# Curva de aprendizado da última rodada
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
plt.show()'''

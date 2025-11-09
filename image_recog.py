import cv2
import numpy as np
import os
import random
from tabulate import tabulate

#modelos
from models.adaline import Adaline
from models.perceptron import Perceptron
from models.mlp import MLP

path_to_img_file = './datasets/RecFac'
people_list = os.listdir(path_to_img_file)
R = 80

X = np.empty((R*R, 0))
C = len(people_list)
Y = np.empty((C, 0))
i = 0

# === ==========================
# Função de predição multiclasse 
# ============================= 
# obs* é necessária para adaptar a característica de retorno binário dos neuronios simples

def predict_multiclass(models, x): 
    x = x.reshape(1, -1) 
    scores = [np.dot(m.w[1:], x.T) - m.w[0] for m in models] # ativações lineares 
    return np.argmax(scores)


for person in people_list:
    imgs_by_person_list = os.listdir(f'{path_to_img_file}/{person}')
    label = -np.ones((C, len(imgs_by_person_list)))
    label[i,:] = 1
    i+=1
    Y = np.hstack((
        Y,label
    ))
    for image in imgs_by_person_list:
        img = cv2.imread(f'{path_to_img_file}/{person}/{image}', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (R,R))
        x = img.flatten().reshape(R*R, 1)
        X = np.hstack((
            X, x
        ))

X = X.T
Y = Y


# Normalização Min-Max: Escala todos os valores de 0-255 para 0-1
X = X / 255.0

print(f"\nDataset carregado: {X.shape[0]} imagens com {X.shape[1]} features cada.\n")

total_rounds = 10

perceptron_accuracies = []
adaline_accuracies = []
mlp_accuracies = []
#======================
# Monte carlo
#======================
for montecarlo_round in range(total_rounds):

    #=============================
    #Split de treino e teste (80%)
    #=============================
    N = X.shape[0]
    indexes = np.arange(N)
    np.random.shuffle(indexes)

    split = int(0.8 * N)
    train_idx = indexes[:split]
    test_idx = indexes[split:]

    X_train = X[train_idx, :]
    X_test = X[test_idx, :]

    Y_train = Y[:, train_idx]
    Y_test = Y[:, test_idx]

    #===============
    #Treinamento Perceptron
    #===============
    perceptron_models = []

    for c in range(C):
        y_c = Y_train[c, :]
        p = Perceptron(eta=0.001, max_epochs=100)
        p.fit(X_train, y_c)
        perceptron_models.append(p)
    #====================
    #Teste perceptron
    #===================
    hits_perceptron = 0
    for i in range(X_test.shape[0]):
        pred = predict_multiclass(perceptron_models, X_test[i])
        real = np.argmax(Y_test[:, i])

        if pred == real:
            hits_perceptron += 1

    acc_perceptron = hits_perceptron / X_test.shape[0]
    perceptron_accuracies.append(acc_perceptron)

    adaline_models = []

    for c in range(C):
        y_c = Y_train[c, :]
        a = Adaline(eta=0.0001, max_epochs=100)
        a.fit(X_train, y_c)
        adaline_models.append(a)

    hits_adaline = 0
    for i in range(X_test.shape[0]):
        # A função predict_multiclass funciona perfeitamente com modelos Adaline
        pred = predict_multiclass(adaline_models, X_test[i]) 
        real = np.argmax(Y_test[:, i])
        if pred == real:
            hits_adaline += 1

    acc_a = hits_adaline / X_test.shape[0]
    adaline_accuracies.append(acc_a)

    #====================
    #Treinamento do MLP
    #===================
    mlp = MLP(X_train.T, Y_train, [1024, 512, 256], 0.001, 1e-12, 50)
    mlp.fit()
    
    #mlp.predict2()

    #====================
    #Teste do MLP
    #===================
    q_acertos = 0
    for i in range(Y_test.shape[1]):
        d_t = Y_test[:, i]
        i_d = 0
        for j in range(d_t.shape[0]):
            if d_t[j] == 1:
                i_d = j
                break
        x_t = X_test[i, :].T
        y_t = mlp.predict2(x_t)
        if i_d == y_t:
            q_acertos += 1
    mlp_accuracies.append(q_acertos / Y_test.shape[1])
    print(mlp_accuracies[montecarlo_round])

# --- Resultados finais ---
print("\n===== RESULTADOS FINAIS: Acurácia Perceptron =====")
print(f"Acurácias: {np.round(perceptron_accuracies, 4)}")
print(f"Média: {np.mean(perceptron_accuracies):.2%}")
print(f"Desvio padrão: {np.std(perceptron_accuracies):.2%}")
print(f"Máximo: {np.max(perceptron_accuracies):.2%}")
print(f"Mínimo: {np.min(perceptron_accuracies):.2%}")

print("\n===== RESULTADOS FINAIS: Acurácia Adaline =====")
print(f"Acurácias: {np.round(adaline_accuracies, 4)}")
print(f"Média: {np.mean(adaline_accuracies):.2%}")
print(f"Desvio padrão: {np.std(adaline_accuracies):.2%}")
print(f"Máximo: {np.max(adaline_accuracies):.2%}")
print(f"Mínimo: {np.min(adaline_accuracies):.2%}")

print("\n===== RESULTADOS FINAIS: Acurácia MLP =====")
print(f"Acurácias: {np.round(mlp_accuracies, 4)}")
print(f"Média: {np.mean(mlp_accuracies):.2}")
print(f"Desvio padrão: {np.std(mlp_accuracies):.2}")
print(f"Máximo: {np.max(mlp_accuracies):.2}")
print(f"Mínimo: {np.min(mlp_accuracies):.2}")
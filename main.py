import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

#carregamento do dataset spiral_d.csv
data = np.loadtxt('./datasets/spiral_d.csv', delimiter=',')

#separação de informações do dataset
initial_X = data[:, :-1]
classes = data[:, -1]
entry_x1 = data[:, 0]
entry_x2 = data[:, 1]

x1_classe0 = entry_x1[classes == -1]
x1_classe1 = entry_x1[classes == 1]

x2_classe0 = entry_x2[classes == -1]
x2_classe1 = entry_x2[classes == 1]

#Visualização do gráfico de espalhamento das entradas por classe
plt.figure(figsize=(6, 6))

plt.scatter(x1_classe0, x2_classe0, color="green", label="Classe 0")
plt.scatter(x1_classe1, x2_classe1, color="red", label="Classe 1")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Distribuição das classes do dataset spiral_d")
plt.legend()
plt.show()

#elaboração do perceptron



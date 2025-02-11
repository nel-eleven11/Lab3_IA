# este codigo es casi el mismo pero en lugar de usar 3 clases en la columna objetivo se usarán solo 2
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()

X_selected = iris.data[:, [0, 1]]  # seleccionar columnas especificas


y = iris.target

## Arreglo  usando solo 2 clasificadores
X_filtered = X_selected[y != 2]
y_filtered = y[y != 2]

trainingX, testX, trainY, testY = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

perceptron = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
perceptron.fit(trainingX, trainY)

predictions = perceptron.predict(testX)

results = classification_report(testY, predictions)

print(results)

x_min, x_max = trainingX[:, 0].min() - 1, trainingX[:, 0].max() + 1
y_min, y_max = trainingX[:, 1].min() - 1, trainingX[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                np.arange(y_min, y_max, 0.01))

Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# grafica
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(trainingX[:, 0], trainingX[:, 1], c=trainY, edgecolors='k', marker='o', cmap=plt.cm.Paired)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Perceptron Decision Boundary')
plt.show()


# Hay algo muy importante a tener en cuenta y es que la columna objetivo tiene 3 posibles valores "setosa, versicolor, virginica"
# Es por eso que la neurona/perceptron esta teniendo una gran dificultad en las predicciones
# ya que el perceptron es un clasificador binario
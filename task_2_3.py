# file for a simple implentation about perceptrons
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report

from sklearn.datasets import load_iris

iris = load_iris()

X_selected = iris.data[:, [0, 1]]  # Co

y = iris.target

trainingX, testX, trainY, testY = train_test_split(X_selected, y, test_size=0.2, random_state=42)

perceptron = Perceptron(max_iter=100, eta0=0.1, random_state=42)
perceptron.fit(trainingX, trainY)

predictions = perceptron.predict(testX)

results = classification_report(testY, predictions)

print(results)
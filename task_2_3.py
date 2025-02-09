# file for a simple implentation about perceptrons
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_iris

iris = load_iris()

X_selected = iris.data[:, [0, 1]]  # Co

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

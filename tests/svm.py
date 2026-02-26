import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from mlpack.svm import LinearSVM

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
print(y)
y[y == 0] = -1
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)

print(X_train.shape)
print(y_train.shape)

svm = LinearSVM()

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

print(confusion_matrix(y_test, y_pred))

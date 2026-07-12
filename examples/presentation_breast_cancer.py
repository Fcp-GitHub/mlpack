"""
Comparison of different models of mlpack with their counterparts of sklearn.
"""

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import (
    Perceptron,
    LogisticRegression
)
from sklearn.svm import (
    SVC,        # For Gaussian kernel (rbf)
    LinearSVC   # For linear kernel
)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from mlpack import perceptron, mlp, logistic_regression, svm
import mlpack.presentation as p

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# 1. Perceptron
p.pprint("Single-Layer Perceptron")
classifier1 = {
        "clf"  : Perceptron(max_iter=100),
        "name" : "sklearn" 
}

classifier2 = {
        "clf"  : perceptron.Perceptron(),
        "name" : "fcp"
}

p.present_dataset(classifier1, classifier2, X, y, standardization=False)
plt.show()

# 2. Logistic Regression
p.pprint("Logistic/Softmax Regression")
classifier1 = {
        "clf"  : LogisticRegression(),
        "name" : "sklearn"
}

classifier2 = {
        #"clf"  : fcp.OneVsAll(fcp.LogisticRegression),
        "clf"  : logistic_regression.LogisticRegression(patience=50),
        "name" : "fcp"
}

p.present_dataset(classifier1, classifier2, X, y)
plt.show()

# 3. Linear SVM
p.pprint("Linear Support Vector Machine")
classifier1 = {
        "clf"  : LinearSVC(),
        "name" : "sklearn"
}

classifier2 = {
        "clf"  : svm.LinearSVM(patience=1),
        "name" : "fcp"
}

p.present_dataset(classifier1, classifier2, X, y, svm_one_hot=True)
plt.show()

# 4. RBF SVM
p.pprint("RBF Support Vector Machine")
classifier1 = {
        "clf"  : SVC(),
        "name" : "sklearn"
}

classifier2 = {
        "clf"  : svm.GaussSVM(patience=1),
        "name" : "fcp"
}

p.present_dataset(classifier1, classifier2, X, y, svm_one_hot=True)
plt.show()

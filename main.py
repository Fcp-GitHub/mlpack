import numpy as np
import matplotlib.pyplot as plt

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

import fcp
import presentation as p

# 1. Perceptron
classifier1 = {
        "clf"  : Perceptron(max_iter=100),
        "name" : "sklearn" 
}

classifier2 = {
        "clf"  : fcp.OneVsAll(fcp.Perceptron),
        "name" : "fcp (OVA)"
}

p.present(classifier1, classifier2, standardization=False)
plt.show()

# 2. Logistic Regression
classifier1 = {
        "clf"  : LogisticRegression(),
        "name" : "sklearn"
}

classifier2 = {
        #"clf"  : fcp.OneVsAll(fcp.LogisticRegression),
        "clf"  : fcp.SoftmaxRegression(),
        "name" : "fcp (SoftmaxRegression)"
}

p.present(classifier1, classifier2)
plt.show()

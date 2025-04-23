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
p.pprint("Single-Layer Perceptron")
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
p.pprint("Logistic/Softmax Regression")
classifier1 = {
        "clf"  : LogisticRegression(),
        "name" : "sklearn"
}

classifier2 = {
        #"clf"  : fcp.OneVsAll(fcp.LogisticRegression),
        "clf"  : fcp.SoftmaxRegression(max_epochs=50),
        "name" : "fcp (SoftmaxRegression)"
}

p.present(classifier1, classifier2)
plt.show()

# 3. Linear SVM
p.pprint("Linear Support Vector Machine")
classifier1 = {
        "clf"  : LinearSVC(),
        "name" : "sklearn"
}

classifier2 = {
        "clf"  : fcp.MultiSVM(fcp.LinearSVM, max_epochs=1),
        "name" : "fcp (OVA)"
}

p.present(classifier1, classifier2, slow=True)
plt.show()

# 4. RBF SVM
p.pprint("RBF Support Vector Machine")
classifier1 = {
        "clf"  : SVC(),
        "name" : "sklearn"
}

classifier2 = {
        "clf"  : fcp.MultiSVM(fcp.GaussSVM, max_epochs=1),
        "name" : "fcp (OVA)"
}

p.present(classifier1, classifier2, slow=True)
plt.show()

# 5. MLP
p.pprint("Multi-Layer Perceptron")
classifier1 = {
        "clf"  : MLPClassifier(max_iter=300),
        "name" : "sklearn"
}

classifier2 = {
        "clf"  : fcp.MLP(64, 128, 10, learning_rate=0.3, epochs=200),
        "name" : "fcp (MLP)"
}

p.present(classifier1, classifier2)
plt.show()

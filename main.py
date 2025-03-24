import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
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

# Get digits data
X, y = load_digits(return_X_y=True)
# Define different percentages of test/held-out dataset
heldout = [0.95, 0.9, 0.75, 0.5, 0.01]
# Define different percentages of validation dataset
validation = 1. - np.array(heldout)

# Dictionary of all classifiers that will be tested and used
classifiers = [
    ("Perceptron", Perceptron(max_iter=110)),
    (
        "Logistic Regression",
        LogisticRegression(max_iter=110, solver="sag", tol=1e-1, C=1.e4 / X.shape[0])
    ),
    (
        "SVC (linear kernel)",
        LinearSVC()
    ),
    (
        "SVC (gaussian kernel)",
        SVC()
    ),
    (
        "Multi-Layer Perceptron",
        MLPClassifier()
    )
]

for name, clf in classifiers:
    print("training %s" % name)
    rng = np.random.RandomState(42)
    yy = []
    for i in heldout:
        yy_ = []
        for r in range(10):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=i, random_state=rng
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            yy_.append(1 - np.mean(y_pred == y_test))
        yy.append(np.mean(yy_))
    plt.plot(validation, yy, label=name)

plt.legend(loc="upper right")
plt.xlabel("Proportion train")
plt.ylabel("Test Error Rate")
plt.show()

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import numpy as np

from matplotlib import pyplot as plt

import fcp
import visualize as v

digits = load_digits()

plt.figure()
pca = PCA(n_components=2)
proj = pca.fit_transform(digits.data)
plt.scatter(proj[:, 0], proj[:, 1], c=digits.target, cmap="Paired")
plt.colorbar()

pca.fit_transform(digits.data)
plt.scatter(proj[:, 0], proj[:, 1], c=digits.target, cmap="Paired")
plt.colorbar()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

ova = fcp.OneVsAll(fcp.Perceptron, args=[0, 0.1, 10, 1e-3])
ova.train(X_train, y_train)
y_pred = ova.classify(X_test)
y_exp  = y_test


v.predictions_plot(X_test, y_pred, y_exp, num_images=128, num_subplots_x=8, num_subplots_y=16)
plt.show()

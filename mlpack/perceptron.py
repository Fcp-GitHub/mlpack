"""
Linear classifier: Perceptron.
"""


""" Third-party packages """
import numpy as np

""" mlpack resources """
from mlpack.linear_classifier import LinearClassifier
from mlpack.ova import OneVsAll


class Perceptron(LinearClassifier):
    def __init__(self, bias=0, learning_rate=0.3, patience=1000, tolerance=1e-3, *args, **kwargs):
        """
        Perceptron class constructor.


        Parameters
        ----------
        bias: bias that can be used to translate the decision boundary.
        learning_rate: learning rate used by the learning algorithm.
        patience: maximum number of epochs for the learning algorithm.
        """
        super().__init__(bias, learning_rate, patience, tolerance, *args, **kwargs)

    def activation_function(self, X: np.ndarray):
        """
        Perceptron's activation function. Used for predicting the confidence scores for samples.


        Parameters
        ----------
        X: a monodimensional np.ndarray.
        """
        return np.dot(X, self.weights) + self.bias

    def loss_function(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Perceptron's loss function.

        Parameters
        ----------
        X_train: np.ndarray containing the training data samples.
        y_train: a monodimensional np.ndarray.
        """
        _temp = -np.dot(y_train, np.matvec(X_train, self.weights))

        return np.max([0, _temp])

    def _internal_predict(self, X: np.ndarray):
        """
        Perceptron's classification algorithm.


        Parameters
        ----------
        X: np.ndarray with the samples to classify. Can be a vector or matrix.

        Notes
        -----
        All elements must have a final unit feature for the results to be accurate.

        Returns
        -------
        Scalar or np.ndarray vector based on X's shape. As the original Perceptron
        algorithms, returns 0 ("no") or 1 ("yes") based on the classification result.
        """
        score = self.activation_function(X)
        return np.heaviside(score, 0)

    def _internal_fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Rosenblatt's learning algorithm (1958) for the Perceptron classifier.


        Parameters
        ----------
        X_train: np.ndarray containing the training data samples
        y_train: np.ndarray containing the expected results for training

        Note
        -----
        This algorithm is for BINARY CLASSIFICATION. Use the OneVsAll class for
        multiclass classification.
        """
        _num_samples = self._get_num_samples(X_train)

        # 1. Compute predicted value
        _y_hat = self._internal_predict(X_train)    #TODO: there might be a better solution
        # 2. Update weigths accordingly 
        self.weights += ( self.eta / _num_samples ) * np.dot(X_train.T, (y_train - _y_hat))
        self.bias    += ( self.eta / _num_samples ) * np.sum(y_train - _y_hat)

        # 3. Update current tolerance
        _current_tol = 1/_num_samples * np.sum(np.abs(y_train - _y_hat))

        return _current_tol

    def __str__(self):
        return f"Perceptron(bias={self.bias}, learning_rate={self.eta}, patience={self.patience}, tolerance={self.tolerance})"

    def __repr__(self):
        return self.__str__()

#Perceptron

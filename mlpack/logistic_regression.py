"""
Linear classifiers: Logistic Regression and Softmax Regression.
"""

""" Third-party packages """
import numpy as np

""" mlpack resources """
from mlpack.linear_classifier import LinearClassifier


class LogisticRegression(LinearClassifier):
    def __init__(self, bias=0, learning_rate=0.3, patience=1000, tolerance=1e-3, regularization=1e-2, *args, **kwargs):
        """
        Logistic regression class constructor.


        Parameters
        ----------
        bias: bias that can be used to translate the decision boundary.
        learning_rate: learning rate used by the learning algorithm.
        patience: maximum number of epochs for the learning algorithm.
        regularization: regularization factor.
        """
        self.regularization = regularization
        super().__init__(bias, learning_rate, patience, tolerance, *args, **kwargs)

    def sigmoid(self, z):
        """
        Sigmoid function applied to `z`.


        Parameters
        ----------
        z: ndarray or scalar value. 
        """
        # Clip value(s) in order to avoid underflows and overflows
        cz = np.clip(z, -5, +5)

        # Avoid overflow if z < 0
        if cz >= 0:
            return 1 / (1 + np.exp(-cz))
        else:   # Avoid overflow if cz >= 0
            e_z = np.exp(cz)
            return e_z / (1 + e_z)

    def activation_function(self, X: np.ndarray):
        r"""
        Activation function of the logistic regression classifier (S is the sigmoid function): 
        .. math::
                S(\boldsymbol X \cdot \boldsymbol w)


        Parameters
        ----------
        X: ndarray containing sample data.
        """
        z = np.dot(X, self.weights) + self.bias
        _res = None
        if np.isscalar(z):
            return self.sigmoid(z)
        else:
            return np.array([self.sigmoid(zi) for zi in z])

    def loss_function(self, X_train: np.ndarray, y_train: np.ndarray):
        r"""
        Log loss function:
        .. math::
            -\frac{1}{N}\sum_{i=1}^N\left[y_i\log(y_i') + (1-y_i)\log(1-y_i')\right]

        """
        # Get total number of samples
        N = self._get_num_samples(X_train)
        
        # Compute predicted values
        _s = self.activation_function(X_train)

        # Compute loss function
        return -np.sum(y_train * np.log(_s) + (1 - y_train) * np.log(1 - _s)) / N

    def _internal_predict(self, X: np.ndarray):
        """
        Logistic regression's classification algorithm.
        """
        _act = self.activation_function(X)
        _pred = np.array([1 if _a > 0.5 else 0 for _a in _act])
        return _pred


    def _internal_fit(self, X_train, y_train, num_iterations=50):
        """
        Logistic regression's learning algorithm.
        Learning is performed using the cross-entropy loss function.
        """
        # Minimize loss function using gradient descent algorithm
        # NOTE: GD is not the best algorithm for several reasons, so 
        #       a better one could be implemented in the future...

        _num_samples = self._get_num_samples(X_train)

        # Define / Initialize current tolerance value variable
        _current_tol = 0
        for _ in range(num_iterations):
            # 1. Compute activation function at sample X[i] (sigmoid of <w,x>)
            _s = self.activation_function(X_train)

            # 2. Compute gradient of loss function at sample X[i]
            _g = (1 / _num_samples) * ( np.dot(X_train.T, (_s - y_train)) + self.regularization * self.weights )
            _b = (1 / _num_samples) * np.sum(_s - y_train)

            # 3. Update weights
            self.weights = self.weights - (self.eta * _g)
            self.bias = self.bias - (self.eta * _b)

        # 4. Update current tolerance
        _current_tol = 1/_num_samples * np.sum(np.abs(y_train - _s))

        return _current_tol

    def __str__(self):
        return f"LogisticRegression(bias={self.bias}, learning_rate={self.eta}, patience={self.patience}, tolerance={self.tolerance}, regularization={self.regularization}"

    def __repr__(self):
        return self.__str__()

#LogisticRegression


class SoftmaxRegression(LinearClassifier):

    def __init__(self, bias=0, learning_rate=0.3, patience=1000, tolerance=1e-3, regularization=1e-2, *args, **kwargs):
        """
        Softmax regression (a.k.a. multinomial logistic regression) class constructor.


        Parameters
        ----------
        bias: bias that can be used to translate the decision boundary.
        learning_rate: learning rate used by the learning algorithm.
        patience: maximum number of epochs for the learning algorithm.
        regularization: regularization factor.
        """
        self.regularization = regularization
        super().__init__(bias, learning_rate, patience, tolerance, *args, **kwargs)

    def softmax(self, z):
        """
        Softmax function applied to `z`.


        Parameters
        ----------
        z: ndarray or scalar value. 

        Notes
        -----
        Uses the safe softmax method in order to avoid the numerical instability of the 
        standard softmax function. In particular, translating all the exponents by the 
        largest score involved guarantees that the result is at most 1.

        See: https://en.wikipedia.org/wiki/Softmax_function#Numerical_algorithms.
        """
        _e = np.exp(z - np.max(z, axis=1, keepdims=True))   # Subtract max for numerical stability
        return _e / np.sum(_e, axis=1, keepdims=True)

    def activation_function(self, X: np.ndarray):
        r"""
        Activation function of the logistic regression classifier (S is the softmax function): 
        .. math::
                S(\boldsymbol X \cdot \boldsymbol w)


        Parameters
        ----------
        X: ndarray containing sample data.
        """
        _dot = np.dot(X, self.weights) + self.bias
        return self.softmax(_dot)

    def loss_function(self, X_train: np.ndarray, y_train: np.ndarray):
        r"""
        Cross-entropy loss function:
        .. math::
            -\sum_{i=1}^C y_i\log(y_i')

        """
        # Compute predicted values
        _s = self.activation_function(X_train)

        return -np.sum(np.sum(y_train * np.log(_s), axis=1))

    def _internal_predict(self, X: np.ndarray):
        _prob = self.activation_function(X)
        return np.argmax(_prob, axis=1)

    def _internal_fit(self, X_train: np.ndarray, y_train: np.ndarray, num_iterations=50):
        # NOTE: same as LogisticRegression's `_internal_fit`: GD is not the best algorithm
        _num_samples = self._get_num_samples(X_train)

        # Define / Initialize current tolerance value variable
        _current_tol = 0
        for _ in range(num_iterations):
            # 1. Compute activation function
            _s = self.activation_function(X_train)

            # 2. Compute gradient of loss function
            _g = (1 / _num_samples) * (np.dot(X_train.T, (_s - y_train)) + self.regularization * self.weights)
            _b = (1 / _num_samples) * np.sum(_s - y_train, axis=0, keepdims=True)

            # 3. Update weights
            self.weights = self.weights - (self.eta * _g)
            self.bias = self.bias - (self.eta * _b)

        # 4. Update current tolerance 
        _current_tol = 1/_num_samples * np.sum(np.abs(y_train - _s))

        return _current_tol
    
    def __str__():
        return f"SoftmaxRegression(bias={self.bias}, learning_rate={self.eta}, patience={self.patience}, tolerance={self.tolerance}, regularization={self.regularization}"

    def __repr__():
        return self.__str__()

#SoftmaxRegression

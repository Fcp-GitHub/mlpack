import numpy as np

from mlpack.classifier import Classifier

class LogisticRegression(Classifier):
    def __init__(self, bias=0, learning_rate=0.3, max_epochs=1000, tolerance=1e-3, regularization=1e-2, *args, **kwargs):
        self.regularization = regularization
        super().__init__(bias, learning_rate, max_epochs, tolerance, *args, **kwargs)
    
    def sigmoid(self, z):
        """
        Sigmoid function applied to `z`.
        """
        if z >= 0:
            return 1 / (1 + np.exp(-z))
        else:
            return np.exp(z) / (1 + np.exp(z))


    def activation_function(self, X: np.ndarray):
        """
        Activation function of the logistic regression classifier: the sigmoid 
        of the dot product of the weights and the sample.
        """
        z = np.dot(X, self.weights) + self.bias
        _res = None
        if np.isscalar(z):
            return self.sigmoid(z)
        else:
            return np.array([self.sigmoid(zi) for zi in z])


    def _internal_predict(self, X: np.ndarray):
        """
        Logistic regression's classification algorithm.
        """
        _act = self.activation_function(X)
        _pred = np.array([1 if _a > 0.5 else 0 for _a in _act])
        return _pred


    def _internal_fit(self, X_train, y_train, num_iterations=50):
        """Logistic regression's learning algorithm.
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

            # Compute loss function at sample _X[i]
            #_loss = np.sum( -y_train*np.log(_s) - (1-y_train)*np.log(1-_s) )

            # 2. Compute gradient of loss function at sample X[i]
            _g = (1 / _num_samples) * ( np.dot(X_train.T, (_s - y_train)) + self.regularization * self.weights )
            _b = (1 / _num_samples) * np.sum(_s - y_train)

            # 3. Update weights
            self.weights -= self.eta * _g
            self.bias -= self.eta * _b

        # 4. Update current tolerance
        _current_tol = 1/_num_samples * np.sum(np.abs(y_train - _s))

        return _current_tol

class SoftmaxRegression(Classifier):
    def __init__(self, bias=0, learning_rate=0.3, max_epochs=1000, tolerance=1e-3, regularization=1e-2, *args, **kwargs):
        self.regularization = regularization
        super().__init__(bias, learning_rate, max_epochs, tolerance, *args, **kwargs)

    def softmax(self, z):
        """
        Softmax function applied to `z`.
        """
        _e = np.exp(z - np.max(z, axis=1, keepdims=True))   # Subtract max for numerical stability
        return _e / np.sum(_e, axis=1, keepdims=True)

    def activation_function(self, X: np.ndarray):
        _dot = np.dot(X, self.weights) + self.bias
        return self.softmax(_dot)

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

            #TODO: Compute loss function
            # ...

            # 2. Compute gradient of loss function
            _g = (1 / _num_samples) * (np.dot(X_train.T, (_s - y_train)) + self.regularization * self.weights)
            _b = (1 / _num_samples) * np.sum(_s - y_train, axis=0, keepdims=True)

            # 3. Update weights
            self.weights -= self.eta * _g
            self.bias = self.bias - self.eta * _b

        # 4. Update current tolerance 
        _current_tol = 1/_num_samples * np.sum(np.abs(y_train - _s))

        return _current_tol



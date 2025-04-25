"""
Abstract Base Class used for different classifiers.
"""

import abc
import warnings
import numpy as np

class Classifier(abc.ABC):
    """
    Base class for all classifiers.
    """
    def __init__(self, bias, learning_rate, max_epochs, tolerance, warnings_on=False):

        if (max_epochs is None) or (max_epochs <= 0) or (int(max_epochs) != max_epochs):
            raise ValueError(f'{self.__str__}: `max_epochs` must be a positive natural number.')
        if (tolerance is None) or (tolerance <= 0): 
           raise ValueError(f'{self.__str__}: `tolerance` must be a positive real number.')

        self.bias = bias
        self.eta = learning_rate
        self.max_epochs = int(max_epochs)
        self.tol = tolerance
        self.weights = None
        self.labels = None
        self.warnings_on = warnings_on
        
        # For multiclass classifiers
        self.num_classes = None

    def _cc_data(self, X: np.ndarray, y: np.ndarray = None):
        """
        Check and convert data if needed.
        Internal method.

        Returns
        --------
        _X: input data properly converted.
        """
        # Check if X is a np.ndarray instance
        if isinstance(X, np.ndarray):
            _X = X
        else:
            _X = np.asarray(X)

        # Check if y, if given, is a np.ndarray instance
        if y is not None:
            if isinstance(y, np.ndarray):
                _y = y
            else:
                _y = np.asarray(y)

        # If X is a vector...
        if _is_vector:
            _X = np.append(_X, 1)
        else:
            _X = np.c_[ X, np.ones(_num_samples) ]

        return _X, _y

    def _is_vector(self, X: np.ndarray):
        """
        Check if given `X` is a vector or not.
        Internal method.
        """
        return X.ndim == 1

    def _get_num_samples(self, X: np.ndarray):
        """
        Get number of samples by inspection of `X`.
        """
        return 1 if self._is_vector(X) else X.shape[0]

    def _get_num_features(self, X: np.ndarray):
        """
        Get number of features by inspection of `X`.
        """
        return X.shape[-1]

    def _get_num_classes(self, y: np.ndarray):
        """
        Get number of classes from the desired output of a training dataset.
        """
        return len(np.unique(y))

    @abc.abstractmethod
    def activation_function(self, X: np.ndarray):
        pass

    @abc.abstractmethod
    def _internal_predict(self, X: np.ndarray):
        pass 

    def predict(self, X: np.ndarray):
        if self.weights is None or self.bias is None:
                raise RuntimeError("Model not trained yet. Call `fit()` first.") 
        return self._internal_predict(X)

    @abc.abstractmethod
    def _internal_fit(self, X_train: np.ndarray, y_train: np.ndarray):
        pass

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):

        """ Preparation """

        # Input validation
        if not isinstance(y_train, np.ndarray):
            y_train = np.asarray(y_train)

        # Get number of classes
        self.num_classes = self._get_num_classes(y_train)

        # Get number of features
        _num_features = self._get_num_features(X_train)

        # Initialize weights and bias based on the number of classes that are needed
        if self.num_classes == 2:
            self.weights = np.random.rand(_num_features)
            # Bias already good
        else:
            self.weights = np.random.rand(_num_features, self.num_classes)
            self.bias = np.full((1, self.num_classes), self.bias)

        _y = y_train
        if self.num_classes > 2:
        #    # Convert labels to one-hot encoding in order to:
        #    # 1. No false relationships between classes.
        #    # 2. Easy comparison with probabilities.
        #    # 3. Have clear targets while learning.
        #    #TODO: might not be a really clear / readable solution
        #    # Basically:
        #    # Create an identity matrix of size `self.num_classes`x`self.num_classes`
        #    # Flatten the `y_train` array
        #    # Use the flattened `y_train` array as an array of indices for the identity matrix
        #    # Store the specified rows in the `_y` array
            _y = np.eye(self.num_classes)[y_train.reshape(-1)]

        """ Learning Algorithm """

        # Initialize current tolerance
        _current_tol = np.inf
        # Loop through all epochs or until the desired tolerance is reached
        for _ in range(self.max_epochs):
            # Check if desired tolerance was reached
            #if _current_tol > self.tol:
            if not np.all(_current_tol < self.tol):
                # Loop through training data set using classifier-
                # specific learning algorithm and get current tolerance value
                _current_tol = self._internal_fit(X_train, _y)
            else:
                return

        if self.warnings_on:
            # If function hasn't returned yet, it means that the number of epochs
            # wasn't enough to reach convergence
            warnings.warn("Convergence wasn't reached within the specified number of epochs!")

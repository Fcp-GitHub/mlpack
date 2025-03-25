import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

class OneVsAll:
    """
    Implementation of One-Vs-Rest strategy for binary classifiers.
    """
    def __init__(self, classifier, args=None):
        self.classifier = classifier
        self.classifier_args = args
        self.cdict = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train each classifier with its own learning algorithm.
        """
        # Input validation
        if not isinstance((X_train, y_train), np.ndarray):
            X_train = np.asarray(X_train)
            y_train = np.asarray(y_train)

        # Save unique values of y_train (classes)
        classes = np.unique_counts(y_train).values
        # Get number of needed classifiers (number of classes)
        num_classifiers = classes.shape[0]
        # Definition of the array of classifiers
        self.cdict = {}
        if self.classifier_args is not None:
            for label in classes:
                self.cdict[label] = self.classifier(*self.classifier_args)
        else:
            for label in classes:
                self.cdict[label] = self.classifier()

        # Iterate through all the different classes
        for label,classifier in self.cdict.items():
            new_y_train = np.where(y_train == label, 1, 0) 
            classifier.train(X_train, new_y_train)


    def classify(self, X):
        """
        Classify samples using One-Vs-Rest strategy.
        """
        # Input validation 
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        _is_vector = X.ndim == 1

        # Confidence score: the highest will decide the prediction to be returned
        highest_score = 0
        # Classify with each classifier and report the highest confidence score
        # Distinguish between single-sample (vector) and multiple-samples (matrix)
        # case
        if _is_vector:
            # VECTOR
            # Best prediction based on confidence score
            best_pred = 0
            chosen_label = 0 

            for label,classifier in self.cdict.items():
                new_score = classifier.activation_function(X)
                if new_score > highest_score:
                    highest_score = new_score
                    chosen_label = label

            return chosen_label

        else:
            # MATRIX
            _num_samples = X.shape[0]
            best_predictions = np.zeros(_num_samples)
            
            for i,sample in zip(range(0,_num_samples), X):
                best_predictions[i] = self.classify(sample) 
            
            return best_predictions


class Perceptron:
    def __init__(self, bias=0, learning_rate=0.3, max_epochs=1000, tolerance=1e-3):
        """
        Perceptron class constructor.


        Parameters
        ----------
        bias: bias that can be used to translate the decision boundary.
        learning_rate:
        max_epochs:
        iter_error:
        """
        if (max_epochs is None) or (max_epochs <= 0) or (not isinstance(max_epochs, int)):
            raise ValueError("Perceptron: `max_epochs` must be a positive natural number.")
        if (tolerance is None) or (tolerance <= 0): 
           raise ValueError("Perceptron: `tolerance` must be a positive real number.")

        self.bias       = bias
        self.eta        = learning_rate
        self.max_epochs = max_epochs
        self.tol        = tolerance
        self.weights    = None
        #self.labels    = None

    def _convert_to_useful_data(self, X: np.ndarray):
        """
        Internal method for internal conversion and analysis of inputted data.


        Returns
        --------
        _X: input data properly converted.
        _is_vector: boolean value that is `True` if X is a vector (single sample), 
                    `False` otherwise.
        _num_samples: number of samples contained in X. 
                      `_num_samples` is equal to `1` if `_is_vector` is `True`.
        """
        # Check if X is a np.ndarray instance
        _X = None
        if not isinstance(X, np.ndarray):
           _X = np.asarray(X)
        else:
            _X = X

        # Check if X is a vector or a matrix
        _is_vector = _X.ndim == 1

        # Get number of samples 
        _num_samples = 0
        if _is_vector:
            _num_samples = 1
        else:
            _num_samples = _X.shape[0]

        # If X is a vector...
        if _is_vector:
            _X = np.append(_X, 1)
        else:
            _X = np.c_[ X, np.ones(_num_samples) ]

        return _X, _is_vector, _num_samples

    def activation_function(self, X: np.ndarray):
        """
        Perceptron's activation function. Used for predicting the confidence scores for samples.


        Parameters
        ----------
        X: a monodimensional np.ndarray.
        """
        _X = self._convert_to_useful_data(X)[0]
        return np.dot(self.weights,_X)

    def classify(self, X: np.ndarray):
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
        # Input validation
        _X, _is_vector, _num_samples = self._convert_to_useful_data(X) 

        # Classification algorithm
        classification = np.zeros(_num_samples)
            
        # For every sample: 
        for i in range(0,_num_samples):
            # Compute scalar product: <w,x>
            if not _is_vector:
                _dot = np.dot(X[i], self.weights)
            else:
                _dot = np.dot(X, self.weights)
            # Compute Heaviside function
            _res = np.heaviside(_dot, 0)
            # Append result
            classification[i] = _res

        return classification


    def train(self, X_train: np.ndarray, y_train: np.ndarray):
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
        # Input validation
        if not isinstance(y_train, np.ndarray):
            y_train = np.asarray(y_train)
        
        _X_train, _is_vector, _num_samples = self._convert_to_useful_data(X_train)

        # Number of features per sample
        _num_features = _X_train.shape[0] if _is_vector else _X_train.shape[1]

        # Initizialize weights
        self.weights = np.random.rand(_num_features) #np.zeros(_num_features)
        self.weights[-1] = self.bias    # Append bias
        
        # Deduce the two labels and store them for later
        #self.labels = np.unique_counts(y_train).values

        # Loop through all the epochs until loop number `max_epochs` is reached
        # or until we've reached the desired tolerance `tolerance`
        for i in range(0, self.max_epochs):
            # Define current tolerance
            _current_tol = np.inf
            # Check if we've reached desired tolerance
            if _current_tol > self.tol:
                # Prepare current tolerance for incoming epoch
                _current_tol = 0
                # Loop through the training data set
                for sample, y_des in zip(_X_train, y_train):
                    # 1. Compute predicted value
                    _y_hat = self.classify(sample)
                    # 2. Update weigths accordingly
                    self.weights += self.eta * (y_des - _y_hat) * sample
                    # 3. Update current tolerance
                    _current_tol += 1/_num_samples * np.abs(y_des - _y_hat)
            else:
                break

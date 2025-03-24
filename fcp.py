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
        self.carray = None

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
        self.carray = np.ndarray((num_classifiers,), dtype=object)
        if self.classifier_args is not None:
            for i in range(0,num_classifiers):
                self.carray[i] = self.classifier(*self.classifier_args)
        else:
            for i in range(0,num_classifiers):
                self.carray[i] = self.classifier()

        # Iterate through all the different classes
        for label,classifier in zip(classes,self.carray):
            new_y_train = np.where(y_train == label, label, 0)
            classifier.train(X_train,new_y_train)


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

            for classifier in self.carray:
                new_score = classifier.activation_function(X)
                if new_score > highest_score:
                    highest_score = new_score
                    best_pred = classifier.classify(X)

            return best_pred

        else:
            # MATRIX
            _num_samples = X.shape[0]
            best_predictions = np.zeros(_num_samples)
            
            for i,sample in zip(range(0,_num_samples), X):
                best_predictions[i] = self.classify(sample) 
            
            return best_predictions


class Perceptron:
    def __init__(self, bias=0, learning_rate=0.3, max_iterations=None, iter_error=None):
        """
        Perceptron class constructor.


        Parameters
        ----------
        bias: bias that can be used to translate the decision boundary.
        learning_rate:
        max_iterations:
        iter_error:
        """
        self.bias     = bias
        self.eta      = learning_rate
        self.gamma    = iter_error
        self.max_iter = max_iterations
        self.weights  = None
        self.labels   = None

    def activation_function(self, X):
        """
        Perceptron's activation function. Used for predicting the confidence scores for samples.
        """
        return np.dot(self.weights,X)

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
        Scalar or np.ndarray vector based on X's shape.
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        # Classification algorithm
        _is_vector = X.ndim == 1

        # Distinguish between vector and matrix case:
        if _is_vector:
            # VECTOR
            # 1. Initialize classification to zero
            classification = 0

            # 2. Apply formula:
            # 2.1. Compute scalar product: <w,x>
            _dot = np.dot(X, self.weights)
            _res = np.heaviside(_dot, 0)
            # 2.2. Result
            if _res > 0:
                classification = self.labels[1]
            else:
                classification = self.labels[0]
            
            return classification
        else:
            # MATRIX
            _num_samples = X.shape[0]
            # 1. Initialize classification array to zero
            classification = np.zeros(_num_samples)  
            
            # 2. For every sample: 
            for i in range(0,_num_samples):
                # 2.1. Compute scalar product: <w,x>
                _dot = np.dot(X[i], self.weights)
                # 2.2. Compute Heaviside function
                _res = np.heaviside(_dot, 0)
                # 2.3. Append result
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
        if not isinstance((X_train, y_train), np.ndarray):
            X_train = np.asarray(X_train)
            y_train = np.asarray(y_train)
        # Check if X is a vector
        _is_vector = X_train.ndim == 1

        # Testing set size
        _tset_size = y_train.shape[0]

        # Number of features
        num_features = X_train.shape[0] if X_train.ndim == 1 else X_train.shape[1]

        # Initizialize weights
        self.weights = np.zeros(num_features)
        self.weights[-1] = self.bias    # Append bias
        print(self.bias)
        
        # Deduce the two labels and store them for later
        self.labels = np.unique_counts(y_train).values

        if self.max_iter is not None:
            for i in range(0, self.max_iter):
                # 1. Compute predicted value
                _y_hat = self.classify(X_train[i])
                # 2. Update weigths accordingly
                self.weights += self.eta * (y_train[i] - _y_hat) * X_train[i]

        elif self.gamma is not None:
            _current_eta = np.inf
            while _current_eta < np.eta:
                _current_eta = np.inf
                for sample,y_des in zip(X_train, y_train):
                    # 1. Compute predicted value
                    _y_hat = self.classify(sample)
                    # 2. Update weights accordingly
                    self.weights += self.eta * (y_des - _y_hat) * sample
                    # 3. Update current eta
                    _current_eta += 1/_tset_size * np.abs(y_des - _y_hat) * sample
        
        else:
            for sample,y_des in zip(X_train, y_train):
                    # 1. Compute predicted value
                    _y_hat = self.classify(sample)
                    # 2. Update weights accordingly
                    self.weights += self.eta * (y_des - _y_hat) * sample

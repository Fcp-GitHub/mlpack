import abc
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from concurrent.futures import ProcessPoolExecutor

class Classifier(abc.ABC):
    """
    Base class for all classifiers of `fcp` module.
    """
    def __init__(self, bias, learning_rate, max_epochs, tolerance, warnings_on=False):

        if (max_epochs is None) or (max_epochs <= 0) or (int(max_epochs) != max_epochs):
            raise ValueError(f'{self.__str__}: `max_epochs` must be a positive natural number.')
        if (tolerance is None) or (tolerance <= 0): 
           raise ValueError(f'{self.__str__}: `tolerance` must be a positive real number.')

        self.bias = bias
        self.learning_rate = learning_rate
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
            # Convert labels to one-hot encoding in order to:
            # 1. No false relationships between classes.
            # 2. Easy comparison with probabilities.
            # 3. Have clear targets while learning.
            #TODO: might not be a really clear / readable solution
            # Basically:
            # Create an identity matrix of size `self.num_classes`x`self.num_classes`
            # Flatten the `y_train` array
            # Use the flattened `y_train` array as an array of indices for the identity matrix
            # Store the specified rows in the `_y` array
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

class OneVsAll:
    """
    Implementation of One-Vs-Rest strategy for binary classifiers.
    

    The 'OneVsAll' class uses 'concurrent.futures.ProcessPoolExecutor'
    (added with python 3.2) in order to speed up execution via multiprocessing.
    """
    def __init__(self, classifier: Classifier, args=None):
        self.classifier = classifier
        self.classifier_args = args
        self.cdict = None

    def _run_cpu_tasks_in_parallel(self, tasks, fnargs):
        """
        Run tasks concurrently in order to avoid as much overhead as possible.


        Parameters
        ----------
        tasks : list of functions to call
        fnargs: list of arguments to be used per function

        Return value
        ------------
        Returns the list of return values if the tasks have one, otherwise a list 
        of `None` is returned.
        """
        with ProcessPoolExecutor() as executor:
            # Submit all tasks for parallel execution
            futures = [executor.submit(task,*fnarg) for task,fnarg in zip(tasks,fnargs)]
            results = [None]*len(futures)
            for i,future in zip(range(len(futures)), futures):
                results[i] = future.result() # Get result
        return results

    def _parallel_fit(self, classifier : Classifier, fnargs):
        """
        Internal method for parallelization of fitting method.
        """
        #print(f"{classifier} now running...")
        classifier.fit(*fnargs)     # Execute learning algorithm
        return classifier.weights   # Return updated weights

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train each classifier with its own set of classes.
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

        fnargs = [None]*num_classifiers
        tasks  = [None]*num_classifiers

        # Iterate through all the different classes and save function
        # arguments for later
        for i,(label,clf) in zip(range(num_classifiers), self.cdict.items()):
            new_y_train = np.where(y_train == label, 1, 0) 
            fnargs[i] = [clf, [X_train, new_y_train]]
            tasks[i]  = self._parallel_fit
        
        # Run learning algorithm for all classifiers (hopefully) concurrently
        results = self._run_cpu_tasks_in_parallel(tasks, fnargs)
        for clf,result in zip(self.cdict.values(), results):
            clf.weights = result

    def activation_function(self, X: np.ndarray):
        return np.array([clf.activation_function(X) for clf in self.cdict.values()]).T

    def predict(self, X):
        """
        Classify samples using One-Vs-Rest strategy.
        """
        # Input validation 
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        # Classify with each classifier and report the highest confidence score
        _prob = self.activation_function(X)
        return np.argmax(_prob, axis=1)

        
class Perceptron(Classifier):
    def __init__(self, bias=0, learning_rate=0.3, max_epochs=1000, tolerance=1e-3, *args, **kwargs):
        """
        Perceptron class constructor.


        Parameters
        ----------
        bias: bias that can be used to translate the decision boundary.
        learning_rate:
        max_epochs:
        iter_error:
        """
        super().__init__(bias, learning_rate, max_epochs, tolerance, *args, **kwargs)

    def activation_function(self, X: np.ndarray):
        """
        Perceptron's activation function. Used for predicting the confidence scores for samples.


        Parameters
        ----------
        X: a monodimensional np.ndarray.
        """
        _dot = np.dot(X, self.weights)
        return _dot + self.bias

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
        _act = self.activation_function(X)
        return np.heaviside(_act, 0)

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
        self.weights += ( self.learning_rate / _num_samples ) * np.dot(X_train.T, (y_train - _y_hat))
        self.bias    += ( self.learning_rate / _num_samples ) * np.sum(y_train - _y_hat)

        # 3. Update current tolerance
        _current_tol = 1/_num_samples * np.sum(np.abs(y_train - _y_hat))

        return _current_tol


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


    def _internal_fit(self, X_train, y_train, num_iterations=100):
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
            self.weights -= self.learning_rate * _g
            self.bias -= self.learning_rate * _b

        # 4. Update current tolerance
        _current_tol = 1/_num_samples * np.sum(np.abs(y_train - _s))

        return _current_tol

class SoftmaxRegression(Classifier):
    def __init__(self, bias=0, learning_rate=0.3, max_epochs=1e3, tolerance=1e-3, regularization=1e-2, *args, **kwargs):
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

    def _internal_fit(self, X_train: np.ndarray, y_train: np.ndarray, num_iterations=100):
        # NOTE: same as LogisticRegression's `_internal_fit`: GD is not the best algorithm
        _num_samples = self._get_num_samples(X_train)

        # Define / Initialize current tolerance value variable
        _current_tol = 0
        for _ in range(num_iterations):
            # 1. Compute activation function
            _s = self.activation_function(X_train)

            # Compute loss function
            # ...

            # 2. Compute gradient of loss function
            _g = (1 / _num_samples) * (np.dot(X_train.T, (_s - y_train)) + self.regularization * self.weights)
            _b = (1 / _num_samples) * np.sum(_s - y_train, axis=0, keepdims=True)

            # 3. Update weights
            self.weights -= self.learning_rate * _g
            self.bias = self.bias - self.learning_rate * _b

        # 4. Update current tolerance 
        _current_tol = 1/_num_samples * np.sum(np.abs(y_train - _s))

        return _current_tol

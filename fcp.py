import abc
import warnings
import numpy as np
import copy
from scipy.optimize import minimize, Bounds
from scipy.spatial import distance
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import matplotlib.pyplot as plt
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

class OneVsAll:
    """
    Implementation of One-Vs-Rest strategy for binary classifiers.
    

    The 'OneVsAll' class uses 'concurrent.futures.ProcessPoolExecutor'
    (added with python 3.2) in order to speed up execution via multiprocessing.
    """
    def __init__(self, classifier: Classifier, args=None, svm_labels=False, workers=None):
        self.classifier = classifier
        self.classifier_args = args
        self.cdict = None
        self.svm_labels = svm_labels
        self.workers = workers

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
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            # Submit all tasks for parallel execution
            futures = [executor.submit(task,*fnarg) for task,fnarg in zip(tasks,fnargs)]
            results = [None]*len(futures)
            for i, future in enumerate(futures):
                results[i] = future.result() # Get result
        return results

    def _parallel_fit(self, classifier : Classifier, fnargs):
        """
        Internal method for parallelization of fitting method.
        """
        #print(f"{classifier} now running...")
        classifier.fit(*fnargs)     # Execute learning algorithm
        return classifier.weights, classifier.bias   # Return updated weights

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
            new_y_train = np.where(y_train == label, 1, 0 if not self.svm_labels else -1) 
            fnargs[i] = [clf, [X_train, new_y_train]]
            tasks[i]  = self._parallel_fit
        
        # Run learning algorithm for all classifiers (hopefully) concurrently
        #print("Running tasks...")
        results = self._run_cpu_tasks_in_parallel(tasks, fnargs)
        for clf,result in zip(self.cdict.values(), results):
            clf.weights, clf.bias = result

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
        return np.dot(X, self.weights) + self.bias

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
        self.weights += ( self.eta / _num_samples ) * np.dot(X_train.T, (y_train - _y_hat))
        self.bias    += ( self.eta / _num_samples ) * np.sum(y_train - _y_hat)

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

class SVM(Classifier):
    def __init__(self, bias=0, learning_rate=0.3, max_epochs=1000, tolerance=1e-3, support_vectors_tol=1e-11, regularization=1, *args, **kwargs):
        super().__init__(bias, learning_rate, 1, tolerance, *args, **kwargs)
        self.lmul = None    # Lagrange multipliers
        self.is_sv = None
        self.sv_i = None    # Support vectors' indices
        self.sv_tol = support_vectors_tol   # Tolerance used to identify support vectors
        self.regularization = regularization

        self.Xt, self.yt = None, None

    def activation_function(self, X: np.ndarray):
        return self.kernel(X, self.weights) + self.bias

    @abc.abstractclassmethod
    def kernel(self, xi, xj):
        pass

    @abc.abstractclassmethod
    def gram(self, X: np.ndarray):
        pass

    def dual(self, l: np.ndarray, K: np.ndarray, y: np.ndarray):
        ly = l * y
        return l.sum() - 0.5 * np.dot(ly.T, np.dot(K,ly)) 

    def dual_gradient(self, l: np.ndarray, K: np.ndarray, y: np.ndarray):
        """Calculates the gradient of the dual objective function."""
        return np.ones_like(l) - np.dot(K, l * y) * y

    def _internal_predict(self, X: np.ndarray):
        #_act = self.activation_function(X)
        #return np.sign(_act)
        xs, ys = self.Xt[self.sv_i, np.newaxis], self.yt[self.sv_i]
        # Support vectors
        l, y, _X = self.lmul[self.is_sv], self.yt[self.is_sv], self.Xt[self.is_sv]
        # Compute bias
        self.bias = ys - np.sum(l * y * self.kernel(_X, xs), axis=0)
        # Compute score
        score = np.sum(l * y * self.kernel(_X, X), axis=0)
        return np.sign(score).astype(int), score
        

    def _internal_fit_scipy(self, X_train: np.ndarray, y_train: np.ndarray):
        #TODO: for now the solution is given only solving the dual problem. There should be a parameter to choose if either the primal or the dual has to be solved.
        _num_samples = self._get_num_samples(X_train)

        # Define initial parameters for the Lagrange multipliers
        l0 = np.random.rand(_num_samples) * 0.1#np.zeros(_num_samples)

        # Compute kernel
        K = self.gram(X_train)

        # Define constraint 
        eq_cons = {
                'type' : 'eq',
                'fun'  : lambda l : np.dot(l, y_train)
        }

        # Define bounds for the Lagrange multipliers
        bounds = Bounds(0, self.regularization)

        # Call scipy optimization routine
        #print("Minimize!")
        res = minimize(
                fun = lambda l : -self.dual(l, K, y_train),
                jac = lambda l : -self.dual_gradient(l, K, y_train),
                x0 = l0,
                method = 'SLSQP',
                constraints = eq_cons,
                bounds = bounds,
                options={'maxiter': 1000, 'ftol': 1e-6}
        )
        #print("Minimized")

        # Check for algorithm success
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        # Save estimated parameters
        self.lmul = res.x

        # Compute weights
        # First, identify support vectors (values with non-zero Lagrange multiplier)
        self.sv_i = np.where(self.lmul > self.sv_tol)[0]
        _sl = self.lmul[self.sv_i]
        _sv = X_train[self.sv_i]
        _sy = y_train[self.sv_i]
        
        self.weights = np.sum(
                _sl[:, np.newaxis] *
                _sy[:, np.newaxis] *
                _sv, axis=0)

        # Compute the actual weights and the bias
        self.bias = _sy - np.dot(_sv, self.weights)
        self.bias = np.mean(self.bias)

        _p = self.predict(X_train)

        return 1/_num_samples * np.sum(np.abs(y_train - _p))

    def _internal_fit(self, X_train: np.ndarray, y_train: np.ndarray):
        _num_samples = self._get_num_samples(X_train)

        # Initialize values and compute matrix H
        _y = y_train.reshape(-1, 1).astype(np.double)   # Has to be a column vector
        self.yt = _y
        self.Xt = X_train
        K = self.gram(X_train)

        # Convert into cvxopt format
        # 0.5 * x^T P x + q^T x
        P = cvxopt_matrix(_y @ _y.T * K)
        q = cvxopt_matrix(-np.ones((_num_samples,1)))

        # Gx <= h
        #G = cvxopt_matrix(-np.eye(_num_samples))
        G = cvxopt_matrix(np.vstack((-np.identity(_num_samples),np.identity(_num_samples))))
        #h = cvxopt_matrix(np.zeros(_num_samples))
        h = cvxopt_matrix(np.vstack((np.zeros((_num_samples,1)), np.ones((_num_samples,1)) * self.regularization)))

        # Ax = b
        A = cvxopt_matrix(_y.T)
        b = cvxopt_matrix(np.zeros(1))

        # Set optimizer parameters
        cvxopt_solvers.options['show_progress'] = False
        #cvxopt_solvers.options['abstol'] = 1e-10
        #cvxopt_solvers.options['reltol'] = 1e-10
        #cvxopt_solvers.options['feastol'] = 1e-10

        # Run solver and store results
        #print("Minimize!")
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        #print("Minimized")
        self.lmul = np.array(sol['x'])

        # Get boolean array that flags support vectors
        self.is_sv = ((self.lmul > self.sv_tol)&(self.lmul <= self.regularization)).squeeze()
        # Get indices of some support vectors
        self.sv_i = np.argmax((self.lmul > self.sv_tol)&(self.lmul <= self.regularization - self.sv_tol))

        # Compute weights 
        #self.weights = ((_y[self.sv_i] * self.lmul[self.sv_i]).T @ X_train[self.sv_i]).reshape(-1,1)

        # Compute bias
        #self.bias = _y[self.sv_i] - np.dot(X_train[self.sv_i], self.weights)
        #self.bias = self.bias[0]

        _p = self.predict(X_train)

        return 1/_num_samples * np.sum(np.abs(y_train - _p))


class LinearSVM(SVM):
    def kernel(self, xi, xj):
        return xi @ xj.T

    def gram(self, X: np.ndarray):
        return X @ X.T

class GaussSVM(SVM):
    def __init__(self, rbf=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rbf = rbf

    def kernel(self, xi, xj):
        #return np.exp(-self.rbf * np.abs(xi - xj)**2)
        return np.exp(-self.rbf * distance.cdist(xi, xj, 'sqeuclidean'))

    def gram(self, X: np.ndarray):
        #_dist = np.sum( (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis = 2)
        #return np.exp(-self.rbf * _dist)
        return np.exp(-self.rbf * distance.cdist(X, X, 'sqeuclidean'))

class MultiSVM:
    def __init__(self, clf: SVM, *args, **kwargs):
        self.clf = clf
        self.clfs = []
        self.args = args
        self.kwargs = kwargs
        self.nclasses = 0

    def _task(self, X_train, y_train, i):
        # Get data for the pair
        Xs, Ys = X_train, copy.copy(y_train)
        # Change labels for multiclass classification
        Ys[Ys != i], Ys[Ys == i] = -1, +1
        # Fit
        clf = self.clf(*self.args, **self.kwargs)
        clf.fit(Xs, Ys)

        # Save classifier
        self.clfs.append(clf)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.clfs = []
        self.nclasses = len(np.unique(y_train))

        for i in range(self.nclasses):
            # Get data for the pair
            Xs, Ys = X_train, copy.copy(y_train)
            # Change labels for multiclass classification
            Ys[Ys != i], Ys[Ys == i] = -1, +1
            # Fit
            clf = self.clf(*self.args, **self.kwargs)
            print(f"fit {i}", end='\r')
            clf.fit(Xs, Ys)

            # Save classifier
            self.clfs.append(clf)

    def predict(self, X: np.ndarray):
        _num_samples = X.shape[0]
        _preds = np.zeros((_num_samples, self.nclasses))
        
        for i, clf in enumerate(self.clfs):
            _, _preds[:, i] = clf.predict(X)

        return np.argmax(_preds, axis=1)

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, x):
        z = np.exp(x - np.max(x))
        return z / z.sum(axis=1, keepdims=True)

    def forward(self, X: np.ndarray):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.softmax(self.final_input)

        return self.final_output

    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray, learning_rate):
        output_error = output - y
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.hidden_output * (1 - self.hidden_output)
        
        self.weights_hidden_output -= learning_rate * np.dot(self.hidden_output.T, output_error)
        self.bias_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        self.weights_input_hidden -= learning_rate * np.dot(X.T, hidden_error)
        self.bias_hidden -= learning_rate * np.sum(hidden_error, axis=0, keepdims=True)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, max_epochs, learning_rate):
        for _ in range(max_epochs):
            output = self.forward(X_train)
            self.backward(X_train, y_train, output, learning_rate)

    def predict(self, X: np.ndarray):
        output = self.forward(X)
        return np.argmax(output, axis=1)

class Neural_Network:
    def __init__(self, n_in, n_hidden, n_out, max_epochs=100, learning_rate=1.2):
        # Network dimensions
        self.n_x = n_in
        self.n_h = n_hidden
        self.n_y = n_out

        self.max_epochs = max_epochs
        self.eta = learning_rate
        
        # Parameters initialization
        self.W1 = np.random.randn(self.n_h, self.n_x) * 0.01
        self.b1 = np.zeros((self.n_h, 1))
        self.W2 = np.random.randn(self.n_y, self.n_h) * 0.01
        self.b2 = np.zeros((self.n_y, 1))
    
    def sigmoid(self, z):
        """
        Sigmoid function applied to `z`.
        """
        #if z >= 0:
        return 1 / (1 + np.exp(-z))
        #else:
        #    return np.exp(z) / (1 + np.exp(z))


    def forward(self, X):
        """ Forward computation """
        self.Z1 = self.W1.dot(X.T) + self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = self.sigmoid(self.Z2)
    
    def back_prop(self,  X, Y):
        """ Back-progagate gradient of the loss """
        m = X.shape[0]
        self.dZ2 = self.A2 - Y
        self.dW2 = (1 / m) * np.dot(self.dZ2, self.A1.T)
        self.db2 = (1 / m) * np.sum(self.dZ2, axis=1, keepdims=True)
        self.dZ1 = np.multiply(np.dot(self.W2.T, self.dZ2), 1 - np.power(self.A1, 2))
        self.dW1 = (1 / m) * np.dot(self.dZ1, X)
        self.db1 = (1 / m) * np.sum(self.dZ1, axis=1, keepdims=True)

    def fit(self, X, Y):
        """ Complete process of learning, alternates forward pass,
            backward pass and parameters update """
        m = X.shape[0]
        for _ in range(self.max_epochs):
            self.forward(X)
            #loss = -np.sum(np.multiply(np.log(self.A2), Y) + np.multiply(np.log(1-self.A2),  (1 - Y))) / m
            self.back_prop(X, Y)

            self.W1 -= self.eta * self.dW1
            self.b1 -= self.eta * self.db1
            self.W2 -= self.eta * self.dW2
            self.b2 -= self.eta * self.db2

            #if e % 1000 == 0:
            #    print("Loss ",  e, " = ", loss)

    def predict(self, X):
        """ Compute predictions with just a forward pass """
        self.forward(X)
        return np.round(self.A2).astype(int)

#from sklearn.datasets import load_digits
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import accuracy_score
 
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=100, batch_size=32):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.eta = learning_rate
        self.epochs = epochs

        self.batch_size = batch_size

        # Weights for the input - hidden layer step
        self.weights_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))

        # Weights for the hidden layer - output step
        self.weights_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        z = np.exp(x - np.max(x, axis=1, keepdims=True))
        return z / np.sum(z, axis=1, keepdims=True)

    def forward(self, X):
        # Compute hidden layer step
        self.hidden_layer_input = np.dot(X, self.weights_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        # Compute output layer step
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_output) + self.bias_output
        self.output_layer_output = self.softmax(self.output_layer_input)

        # Return output
        return self.output_layer_output

    def backward(self, X, y, output):
        num_samples = X.shape[0]

        # Gradient of the loss function with respect to the output layer
        d_output = output - y

        # Gradient of the output layer weights and biases
        d_weights_output = np.dot(self.hidden_layer_output.T, d_output) / num_samples
        d_bias_output = np.sum(d_output, axis=0, keepdims=True) / num_samples

        # Gradient of the hidden layer
        d_hidden = np.dot(d_output, self.weights_output.T) * self.sigmoid_derivative(self.hidden_layer_output)

        # Gradient of the hidden layer weights and biases
        d_weights_hidden = np.dot(X.T, d_hidden) / num_samples
        d_bias_hidden = np.sum(d_hidden, axis=0, keepdims=True) / num_samples

        return d_weights_hidden, d_bias_hidden, d_weights_output, d_bias_output

    def update_parameters(self, d_weights_hidden, d_bias_hidden, d_weights_output, d_bias_output):
        self.weights_hidden -= self.eta * d_weights_hidden
        self.bias_hidden -= self.eta * d_bias_hidden
        self.weights_output -= self.eta * d_weights_output
        self.bias_output -= self.eta * d_bias_output

    def fit(self, X_train, y_train):
        num_classes = len(np.unique(y_train))
        y_train = np.eye(num_classes)[y_train.reshape(-1)]
        num_samples = X_train.shape[0]
        num_batches = num_samples // self.batch_size

        for _ in range(self.epochs):
            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            for i in range(0, num_samples, num_batches):
                #start = i * self.batch_size
                #end = (i + 1) * self.batch_size
                X_batch = X_shuffled[i:i+num_batches]#start:end]
                y_batch = y_shuffled[i:i+num_batches]#start:end]

                output = self.forward(X_batch)
                d_wh, d_bh, d_wo, d_bo = self.backward(X_batch, y_batch, output)
                self.update_parameters(d_wh, d_bh, d_wo, d_bo)

            #if (epoch + 1) % 10 == 0:
            #    output_train = self.forward(X_train)
            #    predictions_train = np.argmax(output_train, axis=1)
            #    accuracy_train = accuracy_score(np.argmax(y_train, axis=1), predictions_train)
            #    print(f"Epoch {epoch+1}/{self.epochs}, Training Accuracy: {accuracy_train:.4f}")

    def predict(self, X):
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        return predictions

"""
Abstract Base Class of all linear classifiers.
"""

import abc
import model
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.inspection import DecisionBoundaryDisplay

from mlpack import visualize


class LinearClassifier(abc.ABC, model.Model):
    """
    Base class for all linear classifiers.
    """
    def __init__(self, bias, learning_rate, patience, tolerance, *args, **kwargs):
        """
        `LinearClassifier` constructor.
        Performs input validation, calls `Model`'s constructor and initializes instance attributes.


        Parameters
        ----------

        bias: float 
            Bias term.

        learning_rate: float 
            Learning rate used by the learning algorithm.
        
        patience: int 
            Maximum number of epochs for the learning algorithm.

        tolerance: float 
            Number used for the convergence condition.
            Specifically, the learning algorithm stops if `patience` epochs have been
            passed or if:
            .. math::
                    (1 - \mathrm{F1-Score}) < \mathrm{tolerance}


        Notes
        -----
        The optional `*args` and `**kwargs` are passed to `Model`'s constructor.
        If `warnings_on` is set to `True` and the learning algorithm doesn't converge on time,
        a warning message gets printed.
        """

        # Input validation
        if (patience is None) or (patience <= 0) or (int(patience) != patience):
            raise ValueError(f'{self.__str__}: `patience` must be a positive natural number.')
        if (tolerance is None) or (tolerance <= 0): 
           raise ValueError(f'{self.__str__}: `tolerance` must be a positive real number.')

        # Call `Model`'s constructor
        super().__init__(*args, **kwargs)

        """ Instance Attributes """
        
        # Store hyperparameters 
        self.bias = bias
        self.eta  = learning_rate
        self.patience  = int(patience)
        self.tolerance = tolerance 
        
        self.weights = None
        self.loss    = None
        self.score   = None

        # Internal "function pointer" used to dispatch fit function
        # based on the classification problem (binary vs multiclass)
        self._internal_fit = None

        # Internal references to animations
        self.fit_anim = None    # For learning animation
        self.cls_anim = None    # For classification animation
    

    """ Abstract Methods """

    # Interface Methods
    @abc.abstractmethod
    def activation_function(self, X: np.ndarray):
        pass

    @abc.abstractmethod
    def loss_function(self, X: np.ndarray, y: np.ndarray):
        pass

    # Internal Methods
    @abc.abstractmethod
    def _internal_predict(self, X: np.ndarray):
        pass 

    @abc.abstractmethod
    def _fit_binary(self, X_train: np.ndarray, y_train: np.ndarray):
        pass

    @abc.abstractmethod
    def _fit_multiclass(self, X_train: np.ndarray, y_train: np.ndarray):
        pass
    

    """ Interface Methods """

    def predict(self, X: np.ndarray, animate: bool = False):
        """
        Apply classification algorithm to unknown sample(s) `X`.


        Parameters
        ----------

        X: ndarray of shape (samples, features)
            array of samples. If not an instance of `numpy.ndarray`, it gets
            internally converted to one.
        
        animate: bool
            animation flag.
        """

        # Input validation
        if self.weights is None or self.bias is None:
                raise RuntimeError("Model not trained yet. Call `fit()` first.") 

        # Avoid original array modifications
        _X = X

        if not isinstance(X, np.ndarray):
            _X = np.asarray(X)

        # Animation flag
        if animate == True:

            self.cls_anim = visualize.decision_boundary_grid(
                    self,
                    _X
            )

        return self._internal_predict(_X)


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = False, animate: bool = False):
        """
        Apply learning algorithm using samples `X_train` and labels `y_train`.


        Parameters
        ----------

        X_train: ndarray of shape (samples, features) 
            array of samples. If not an instance of `numpy.ndarray`, it gets
            internally converted to one.

        y_train: ndarray of shape (samples,)
            array of class labels. If not an instance of `numpy.ndarray`, it gets
            internally converted to one.

        verbose: bool
            verbosity flag.

        animate: bool
            animation flag.
        """

        """ Preparation """

        # Copy data in order to avoid altering original arrays
        _X, _y = X_train, y_train

        # Input "correction"
        if not isinstance(_y, np.ndarray):
            _y = np.asarray(_y)

        if not isinstance(_X, np.ndarray):
            _X = np.asarray(_X)

        # Get number of classes and features
        self.num_classes = self._get_num_classes(_y)

        _num_features = self._get_num_features(_X)

        # Based on classification problem, perform the
        # following operations:
        # - function dispatching 
        # - weights and bias initialization
        if self.num_classes == 2:
            self._internal_fit = self._fit_binary

            self.weights = np.random.rand(_num_features)
            # Bias already good
        else:
            self._internal_fit = self._fit_multiclass

            # Encode labels into one-hots for multiclass classification
            _y = self._get_one_hot(_y)

            self.weights = np.random.rand(
                    _num_features, self.num_classes
            )
            self.bias = np.full(
                    (1, self.num_classes), 
                    self.bias
            )


        # Array of partially-trained classifiers, used for the animation
        classifiers = None
        if animate == True:
            classifiers = []

        """ Learning Algorithm """

        # Initialize loss and score arrays
        self.loss  = np.zeros(self.patience)
        self.score = np.zeros(self.patience)

        # Loop through all epochs or 
        # until the desired tolerance is reached
        for epoch in range(self.patience):

            # Shuffle training data
            
            # Call specific learning algorithm
            self._internal_fit(_X, _y)

            # Save loss and F1-Score values
            self.loss[epoch]  = self.loss_function(_X, _y)

            _y_pred = self.predict(_X)
            self.score[epoch] = f1_score(_y, _y_pred) 

            if verbose == True:
                print(f"Epoch: {epoch+1}")
                print(f"\tLoss : {self.loss[epoch]:.4f}")
                print(f"\tScore: {self.score[epoch]:.4f}")

            if animate == True:
                classifiers.append(self)

            # Check if desired tolerance was reached
            if ( (1 - self.score[epoch]) < self.tolerance ):
                return
        
        if self.warnings_on:
            # If function hasn't returned yet, it means that
            # the number of epochs wasn't enough 
            # to reach convergence
            warnings.warn("Convergence wasn't reached within the specified number of epochs!")

        # If the animation flag is set to `True`, show the evolution of
        # the decision boundary through a grid of pairwise plots
        if animate == True:
            self.fit_anim = visualize.animate_decision_boundary(
                    classifiers,
                    _X
            )


    """ Interface Graphical Visualization Methods """

    def _plot(self, y_values, title: str, **style_kwargs):
        """
        Plot `y_values` over a natural linearly-spaced domain.


        Parameters
        ----------

        y_values: 
            values to plot on the Y-axis.

        title: 
            title to display on the top of the figure

        style_kwargs: 
            keyword arguments directly passed to 
            `matplotlib.pyplot.plot()` function.


        Notes
        ------
        This is an internal method.
        """

        # Instantiate a new `Figure` object
        plt.figure()

        # Plot values
        # Use user-specified keyword arguments for styling
        line, = plt.plot(y_values, **style_kwargs)

        # Basic styling
        plt.title(title)
        plt.xlabel("Epochs")
        plt.grid()
        plt.legend()

        return line
    
    def plot_loss_curve(self, **style_kwargs):
        """
        Plot loss values over the number of epochs passed.

        
        Parameters
        ----------

        **style_kwargs: 
            keyword arguments directly passed to 
            `matplotlib.pyplot.plot()` function.
        """
        # Plot loss values per number of epochs passed
        return self._plot(self.loss, "Loss function curve", **style_kwargs)

    def plot_score_curve(self, **style_kwargs):
        """
        Plot score values over the number of epochs passed.

        
        Parameters
        ----------
        **style_kwargs: 
            keyword arguments directly passed to 
            `matplotlib.pyplot.plot()` function.
        """
        # Plot score values per number of epochs passed
        return self._plot(self.score, "Score function curve", **style_kwargs)

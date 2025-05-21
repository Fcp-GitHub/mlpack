"""
Abstract Base Class of all the learning models.
"""

import abc
import numpy as np

class Model(abc.ABC):
    """
    Base class for all models.
    """
    def __init__(self, warnings_on=False):  # TODO: add verbosity levels
        """
        ABC Model constructor.
        

        Parameters
        -----------
        warnings_on : Verbosity flag. If `True`, prints 
                      warnings if necessary.
        """

        # Verbosity flag
        self.warnings_on = warnings_on

        # For storing some metadata
        self.labels = None
        self.num_classes = None


    """ Utility functions """

    def _is_vector(self, X: np.ndarray):
        """
        Check if given `X` is a vector or not.


        Parameters
        ----------
        X: `numpy.ndarray` instance.

        
        Notes
        -----
        This is an internal method. `X` should be a `numpy.ndarray` instance, 
        otherwise a runtime exception will be raised.
        """
        return X.ndim == 1

    def _get_num_samples(self, X: np.ndarray):
        """
        Get number of samples by inspection of `X`.


        Parameters
        ----------
        X: `numpy.ndarray` instance.

        
        Notes
        -----
        This is an internal method. `X` should be a `numpy.ndarray` instance, 
        otherwise a runtime exception will be raised.        
        """
        return 1 if self._is_vector(X) else X.shape[0]

    def _get_num_features(self, X: np.ndarray):
        """
        Get number of features by inspection of `X`.
       

        Parameters
        ----------
        X: `numpy.ndarray` instance.

        
        Notes
        -----
        This is an internal method. `X` should be a `numpy.ndarray` instance, 
        otherwise a runtime exception will be raised. 
        """
        return X.shape[-1]

    def _get_num_classes(self, y: np.ndarray):
        """
        Get number of classes from the desired output of a training dataset.
       

        Parameters
        ----------
        y: `numpy.ndarray` instance.

        
        Notes
        -----
        This is an internal method. `y` should be a `numpy.ndarray` instance, 
        otherwise a runtime exception will be raised. 
        """
        return len(np.unique(y))

    def _get_one_hot(self, y: np.ndarray):
        """
        Encode class labels into one-hots.
        
        
        Parameters
        ----------
        y: `numpy.ndarray` instance.

        
        Notes
        -----
        Encoding labels to one-hots proves beneficial since it
        leads to the following benefits:
        1. No false relationships between classes.
        2. Easy comparison with probabilities.
        3. Have clear targets while learning.

        This is an internal method. `y` should be a `numpy.ndarray` instance, 
        otherwise a runtime exception will be raised.
        """

        # TODO: might not be a really clear / 
        #       readable solution...
        # Basically:
        # 1. Create an identity matrix of size: 
        #    `self.num_classes`*`self.num_classes`
        # 2. Flatten the `y` array
        # 3. Use the flattened `y` array as an array of
        #    indices for the identity matrix
        # 4. Return the specified rows 
        return np.eye(self.num_classes)[y.reshape(-1)]


    """ Abstract Methods """

    @abc.abstractmethod
    def __str__():
        pass

    @abc.abstractmethod
    def __repr__():
        pass

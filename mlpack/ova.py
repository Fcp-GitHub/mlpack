"""
OVA strategy implementation with multiprocessing parallelism.
"""
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from mlpack.model import Model

class OneVsAll:
    """
    Implementation of One-Vs-All (or One-Vs-Rest) strategy for binary classifiers.
    

    NOTES
    -----
    The 'OneVsAll' class uses 'concurrent.futures.ProcessPoolExecutor'
    (added with python 3.2) in order to speed up execution via multiprocessing.

    In order to see why multiprocessing is preferred over multithreading see 
    for example: https://stackoverflow.com/a/51829082.
    """
    def __init__(self, classifier: Model, args=None, svm_labels=False, workers=None):
        """
        Class constructor.

        
        Parameters
        ----------
        classifier
        args
        svm_labels
        workers: if `None` uses `os.get_cpu_count()`.
        """

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

    def _parallel_fit(self, classifier : Model, fnargs):
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

#OneVsAll

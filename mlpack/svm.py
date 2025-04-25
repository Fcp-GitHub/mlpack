import abc
import numpy as np
import copy
from scipy.optimize import minimize, Bounds
from scipy.spatial import distance
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

from mlpack.classifier import Classifier

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



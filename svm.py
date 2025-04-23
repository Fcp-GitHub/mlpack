import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class LinearSVM_LagrangianDual:
    """
    Implements a linear Support Vector Machine using the Lagrangian dual formulation.
    """
    def __init__(self, C=1.0):
        self.C = C  # Regularization parameter
        self.alphas = None
        self.w = None
        self.b = None
        self.support_vectors_indices = None

    def _objective_function(self, alphas, y, K):
        """
        The objective function to maximize in the dual problem.
        """
        return np.sum(alphas) - 0.5 * alphas.T @ np.diag(y) @ K @ np.diag(y) @ alphas

    def _constraints(self, alphas):
        """
        The constraints for the dual problem.
        """
        return np.sum(alphas)

    def fit(self, X, y):
        """
        Fits the SVM model to the training data using the Lagrangian dual.

        Args:
            X (np.ndarray): Training data features (n_samples, n_features).
            y (np.ndarray): Training data labels (n_samples,). Should be +1 or -1.
        """
        n_samples = X.shape[0]

        # Gram matrix (linear kernel)
        K = X @ X.T

        # Define the bounds for alpha (0 <= alpha_i <= C)
        bounds = [(0, self.C) for _ in range(n_samples)]

        # Define the linear equality constraint (sum of alpha_i * y_i = 0)
        constraints = {'type': 'eq', 'fun': lambda a: np.sum(a * y)}

        # Initial guess for alpha
        initial_alphas = np.zeros(n_samples)

        # Minimize the negative of the objective function (since scipy.optimize.minimize minimizes)
        result = minimize(
            fun=lambda a: -self._objective_function(a, y, K),
            x0=initial_alphas,
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            raise ValueError("Optimization failed: {}".format(result.message))

        self.alphas = result.x

        # Identify support vectors (where alpha_i > 1e-5, a small tolerance)
        self.support_vectors_indices = np.where(self.alphas > 1e-5)[0]
        support_alphas = self.alphas[self.support_vectors_indices]
        support_vectors = X[self.support_vectors_indices]
        support_labels = y[self.support_vectors_indices]

        # Calculate the weight vector w
        self.w = np.sum(support_alphas[:, np.newaxis] * support_labels[:, np.newaxis] * support_vectors, axis=0)

        # Calculate the bias b
        # For numerical stability, we can average the bias computed from each support vector
        bias_sum = 0
        for i in range(len(support_vectors)):
            bias_sum += support_labels[i] - np.dot(self.w, support_vectors[i])
        self.b = bias_sum / len(support_vectors) if len(support_vectors) > 0 else 0.0

    def predict(self, X):
        """
        Predicts the labels for the given data.

        Args:
            X (np.ndarray): Data features (n_samples, n_features).

        Returns:
            np.ndarray: Predicted labels (n_samples,).
        """
        if self.w is None or self.b is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        return np.sign(np.dot(X, self.w) + self.b)

if __name__ == '__main__':
    # Generate some synthetic data
    np.random.seed(42)
    X = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 8], [5, 6]])
    y = np.array([-1, -1, -1, 1, 1, 1])

    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the linear SVM with Lagrangian duality
    svm = LinearSVM_LagrangianDual(C=1.0)
    svm.fit(X_train, y_train)

    # Make predictions
    predictions = svm.predict(X_test)
    print("Predictions:", predictions)
    print("True labels:", y_test)

    # Print the learned weights and bias
    print("Weights (w):", svm.w)
    print("Bias (b):", svm.b)
    print("Support vector indices:", svm.support_vectors_indices)

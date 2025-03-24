"""
perceptron.py -- basic binary classifier. 
"""

from sklearn.datasets import load_digits                # built-in hand-written digits dataset
from sklearn.model_selection import train_test_split    # split dataset into training and testing sets
from sklearn.linear_model import Perceptron             # perceptron
from sklearn.metrics import accuracy_score              # compute accuracy of the classifier
from sklearn.metrics import classification_report       # classification metrics

import fcp

# Load dataset as a (data, target) tuple of two ndarrays:
#   data:   contains a 2D ndarray of shape (1797,64) with each row
#           representing the features
#   target: ndarray of shape (1797) which contains the target samples
X, y = load_digits(return_X_y=True)

# Split dataset into training/validation and testing/heldout data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,  # 20 % of dataset is for testing
                                                    random_state=42 # Seed of random number generator
                                                                    # Passing an integer ensures reproducibility of the results
                                                    )   

# Perceptron classifier
ova = fcp.OneVsAll(fcp.Perceptron)
perceptron = Perceptron(max_iter=100,       # Maximum number of epochs
                        eta0=0.1,           # Learning rate
                        random_state=42     # Seed of random number generator
                        )

# Train classifier
ova.train(X_train, y_train)
perceptron.fit(
        X_train,    # Training data
        y_train     # Target values
        )    

# Predict class labels for samples in test dataset 
fcp_y_pred = ova.classify(X_test)
y_pred = perceptron.predict(X_test)

print(fcp_y_pred)
print(y_pred)

# Compute accuracy of predictions
accuracy = accuracy_score(y_test, fcp_y_pred)
print(f'Accuracy: {accuracy}')
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Classification report
class_report = classification_report(y_test, fcp_y_pred)
print("Classification Report:\n", class_report)
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

import numpy as np
import matplotlib.pyplot as plt
import mlpack.visualize as v

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def pprint(title: str):
    print()
    print(title.center(30))
    print('-'*30)

def present(classifier1: dict, classifier2: dict, standardization=True, slow=False):
    """
    Demonstrate `classifier1` (sklearn classifier) and `classifier2` (fcp classifier) in order to compare them using the handwritten
    digits dataset that comes with the scikit-learn library.


    The presentation follows the following order:
        - Accuracy scores
        - Classification reports
        - Confusion matrices
        - Learning curve
        - Results visualization

    Parameters:
    - `classifier1` and `classifier2` must be two dictionaries containing (in order):
        - `clf` : the actual classifier class, with methods `fit` and `predict`.
        - `name`: a string containing a user-friendly name of the classifier.
    - `standardization` : whether to perform data standardization or not.
    """

    # Validate input
    if classifier1 is None:
        raise ValueError("present: `classifier1` is None.")
    if classifier2 is None:
        raise ValueError("present: `classifier2` is None.")

    # Load parameters separately
    c1, name1 = classifier1.values()
    c2, name2 = classifier2.values()

    # Load dataset
    X, y = load_digits(return_X_y=True)

    # Split dataset into training/validation and testing/heldout data
    X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.3, # 30 % of the total dataset is for testing
            random_state=42 # Seed of rnd generator in order to ensure reproducibility of the results
            )

    # Data standardization, if required
    sc = None
    if standardization:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test  = sc.transform(X_test)

    # Get number of classes (10 in this case)
    num_classes = len(np.unique(y))

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
    #_y = np.eye(num_classes)[y_train.reshape(-1)]

    # Train classifiers    
    c1.fit(X_train, y_train)
    c2.fit(X_train, y_train)
    
    # Make them classify data
    pred1 = c1.predict(X_test)
    pred2 = c2.predict(X_test)

    #TODO: print side-by-side  
    # Accuracy scores
    pprint("ACCURACY SCORES")
    accuracy = accuracy_score(y_test, pred1)
    print(f'Accuracy of {name1}: {accuracy}')
    accuracy = accuracy_score(y_test, pred2)
    print(f'Accuracy of {name2}: {accuracy}')

    # Classification reports
    pprint("CLASSIFICATION REPORT")
    class_report = classification_report(y_test, pred1)
    print(f'Classification report for {name1}:\n', class_report)
    class_report = classification_report(y_test, pred2)
    print(f'Classification report for {name2}:\n', class_report)
    
    # Confusion matrices
    pprint("CONFUSION MATRICES")
    cm = confusion_matrix(y_test, pred1)
    print(f'Confusion matrix for {name1}:\n', cm)
    cm = confusion_matrix(y_test, pred2)
    print(f'Confusion matrix for {name2}:\n', cm)

    # Learning curve
    # Different percentages of held-out data
    heldout = [0.95, 0.9, 0.75, 0.5, 0.01]
    # Different percentages of validation dataset
    validation = 1. - np.array(heldout)
    
    # Number of times to fit and evaluate the same estimator with same heldout
    rounds = 3

    plt.figure("Learning curve")

    for classifier in [classifier1, classifier2]:
        clf, name = classifier.values()
        res = []
        for i in heldout:
            print(f'Training {name} with {i}% heldout value')
            _res = []
            for r in range(rounds if not slow else 1):
                X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=i, random_state=42
                        )
                if standardization:
                    #TODO: This surely could be done better
                    _sc = StandardScaler()
                    X_train = _sc.fit_transform(X_train)
                    X_test  = _sc.transform(X_test)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                _res.append(1 - np.mean(y_pred == y_test))
            res.append(np.mean(_res))
        plt.plot(validation, res, label=name)
    plt.legend(loc="upper right")
    plt.xlabel("Proportion train")
    plt.ylabel("Test Error Rate")

    # Results Visualization  
    pprint("RESULTS VISUALIZATION")
    if standardization:
        X_test = sc.inverse_transform(X_test)
    v.predictions_plot(X_test, pred1, y_test, title=name1)
    v.predictions_plot(X_test, pred2, y_test, title=name2)

    print()


def present_dataset(classifier1: dict, classifier2: dict, X: np.ndarray, y:np.ndarray, standardization=True, svm_one_hot=False, slow=False):
    """
    Demonstrate `classifier1` (sklearn classifier) and `classifier2` (fcp classifier) in order to compare them using an arbitrary dataset passed via the `X` and `y` arrays.


    The presentation follows the following order:
        - Accuracy scores
        - Classification reports
        - Confusion matrices
        - Learning curve

    Parameters:
    - `classifier1` and `classifier2` must be two dictionaries containing (in order):
        - `clf` : the actual classifier class, with methods `fit` and `predict`.
        - `name`: a string containing a user-friendly name of the classifier.
    - `X` and `y`: dataset
    - `svm_one_hot`: whether to perform one-hot encoding for SVM classes (labels [-1, 1])
    - `standardization` : whether to perform data standardization or not.
    """

    # Validate input
    if classifier1 is None:
        raise ValueError("present: `classifier1` is None.")
    if classifier2 is None:
        raise ValueError("present: `classifier2` is None.")

    # Load parameters separately
    c1, name1 = classifier1.values()
    c2, name2 = classifier2.values()

    # One-hot encoding into range [-1, +1], if required
    _y = y
    if svm_one_hot:
        _y[_y == 0] = -1

    # Split dataset into training/validation and testing/heldout data
    X_train, X_test, y_train, y_test = train_test_split(
            X, _y,
            test_size=0.3, # 30 % of the total dataset is for testing
            random_state=42 # Seed of rnd generator in order to ensure reproducibility of the results
            )

    # Data standardization, if required
    sc = None
    if standardization:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test  = sc.transform(X_test)

    # Get number of classes
    num_classes = len(np.unique(y))

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
    #_y = np.eye(num_classes)[y_train.reshape(-1)]

    # Train classifiers    
    c1.fit(X_train, y_train)
    c2.fit(X_train, y_train)
    
    # Make them classify data
    pred1 = c1.predict(X_test)
    pred2 = c2.predict(X_test)

    #TODO: print side-by-side  
    # Accuracy scores
    pprint("ACCURACY SCORES")
    accuracy = accuracy_score(y_test, pred1)
    print(f'Accuracy of {name1}: {accuracy}')
    accuracy = accuracy_score(y_test, pred2)
    print(f'Accuracy of {name2}: {accuracy}')

    # Classification reports
    pprint("CLASSIFICATION REPORT")
    class_report = classification_report(y_test, pred1)
    print(f'Classification report for {name1}:\n', class_report)
    class_report = classification_report(y_test, pred2)
    print(f'Classification report for {name2}:\n', class_report)
    
    # Confusion matrices
    pprint("CONFUSION MATRICES")
    cm = confusion_matrix(y_test, pred1)
    print(f'Confusion matrix for {name1}:\n', cm)
    cm = confusion_matrix(y_test, pred2)
    print(f'Confusion matrix for {name2}:\n', cm)

    # Learning curve
    # Different percentages of held-out data
    heldout = [0.95, 0.9, 0.75, 0.5, 0.01]
    # Different percentages of validation dataset
    validation = 1. - np.array(heldout)
    
    # Number of times to fit and evaluate the same estimator with same heldout
    rounds = 3

    plt.figure("Learning curve")

    for classifier in [classifier1, classifier2]:
        clf, name = classifier.values()
        res = []
        for i in heldout:
            print(f'Training {name} with {i}% heldout value')
            _res = []
            for r in range(rounds if not slow else 1):
                X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=i, random_state=42
                        )
                if standardization:
                    #TODO: This surely could be done better
                    _sc = StandardScaler()
                    X_train = _sc.fit_transform(X_train)
                    X_test  = _sc.transform(X_test)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                _res.append(1 - np.mean(y_pred == y_test))
            res.append(np.mean(_res))
        plt.plot(validation, res, label=name)
    plt.legend(loc="upper right")
    plt.xlabel("Proportion train")
    plt.ylabel("Test Error Rate")

    # Results Visualization  
    #pprint("RESULTS VISUALIZATION")
    #if standardization:
    #    X_test = sc.inverse_transform(X_test)
    #v.predictions_plot(X_test, pred1, y_test, title=name1)
    #v.predictions_plot(X_test, pred2, y_test, title=name2)

    print()

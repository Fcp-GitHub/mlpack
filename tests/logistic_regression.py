from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC

from mlpack.mlp import Neural_Network

X, y = load_breast_cancer(return_X_y=True)
#y[y == 0] = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#lg = fcp.LogisticRegression(learning_rate=1e-2, max_epochs=100, regularization=1e-2)
#lg = fcp.Perceptron()
#lg = fcp.LinearSVM(max_epochs=1, regularization=1)
#lg = LinearSVC()
lg = Neural_Network(30, 10, 1, 10, 0.3)

lg.fit(X_train, y_train)

pred = lg.predict(X_test)[0]
print(pred)

class_report = classification_report(y_test, pred)
print("Classification report:\n", class_report)

cm = confusion_matrix(y_test, pred)
print("Confusion matrix:\n", cm)

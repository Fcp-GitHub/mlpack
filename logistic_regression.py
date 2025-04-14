from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import fcp

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lg = fcp.LogisticRegression(learning_rate=1e-2, max_epochs=100, regularization=1e-2)

lg.fit(X_train, y_train)

pred = lg.predict(X_test)
print(pred)

class_report = classification_report(y_test, pred)
print("Classification report:\n", class_report)

cm = confusion_matrix(y_test, pred)
print("Confusion matrix:\n", cm)

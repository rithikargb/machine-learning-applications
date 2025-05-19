import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target
X = X / 255.0

X_sample, y_sample = X[:20000], y[:20000]
y_sample = y_sample.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.25, random_state=42)

linear_svm = LinearSVC(max_iter=10000, random_state=42)
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)

rbf_svm = SVC(kernel='rbf', gamma='scale', random_state=42)
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)

def evaluate_model(model_name, y_test, y_pred):
    print(f"Evaluation: {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

evaluate_model("Linear SVM", y_test, y_pred_linear)
evaluate_model("Non-linear SVM (RBF)", y_test, y_pred_rbf)

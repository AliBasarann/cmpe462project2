from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import numpy as np


def read_data(path):
    lines = []
    X = []
    y = []
    with open(path, "r") as f:
        while(True):
            line = f.readline()
            if line:
                line = line.replace("\n", "")
                line = line.split(",")
                y.append(line[1])
                x = list(map(float, line[2:]))
                X.append(x)
                lines.append(line)
            else:
                break
    return np.array(X), np.array(y)

def train_linear_classifier_with_features(k):
    selected_features = X_train[:, top_feature_indices[:k]]
    selected_test_features = X_test[:, top_feature_indices[:k]]
    
    lr_clf = LogisticRegression(max_iter=1000)
    lr_clf.fit(selected_features, y_train)
    
    y_pred_lr = lr_clf.predict(selected_test_features)
    
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    return accuracy_lr

class NaiveBayes:
    def __init__(self):
        self.class_probabilities = {}
        self.mean = {}
        self.variance = {}

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]
            self.class_probabilities[c] = len(X_c) / n_samples

            self.mean[c] = np.mean(X_c, axis=0)
            self.variance[c] = np.var(X_c, axis=0)

    def calculate_gaussian(self, class_label, x):
        mean = self.mean[class_label]
        variance = self.variance[class_label]
        numerator = np.exp(-((x - mean) ** 2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                prior = self.class_probabilities[c]
                likelihood = np.prod(self.calculate_gaussian(c, x))
                posterior = prior * likelihood
                posteriors[c] = posterior
            predictions.append(max(posteriors, key=posteriors.get))
        return predictions

def train_and_visualize_tree(depth):
    # Initialize DecisionTreeClassifier with specified depth
    dt_clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    
    # Train the classifier
    dt_clf.fit(X_train, y_train)
    
    # Visualize the learned tree
    plt.figure(figsize=(10, 6))
    plot_tree(dt_clf, filled=True, feature_names=[f"feature_{i}" for i in range(X_train.shape[1])], class_names=[str(i) for i in range(10)])
    plt.title(f"Decision Tree with Max Depth {depth}")
    plt.show()
    return dt_clf


if __name__ == "__main__":
    X, y = read_data("data/wdbc.data")

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize DecisionTreeClassifier
    depths_to_test = [2, 5, 8]  # You can add more depths as needed
    for depth in depths_to_test:
        dt_clf = train_and_visualize_tree(depth)

        # Predict on the test set
        y_pred = dt_clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for decision tree with depth {depth}:", accuracy)

    dt_clf = DecisionTreeClassifier(max_depth=depth, random_state=42)    
    dt_clf.fit(X_train, y_train)
    y_pred = dt_clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for decision tree:", accuracy)

    classifier = NaiveBayes()
    classifier.fit(X_train, y_train)

    test_predictions = classifier.predict(X_test)
    accuracy = np.mean(test_predictions == y_test)
    print("Naive bayes test accuracy:", accuracy)

    feature_importances = dt_clf.feature_importances_

    # Sort the feature importances in descending order and get indices of the top features
    top_feature_indices = np.argsort(feature_importances)[::-1]
    num_features_to_test = [5, 10, 15, 20]
    for num_features in num_features_to_test:
        accuracy_lr = train_linear_classifier_with_features(num_features)
        print(f"Accuracy with {num_features} features:", accuracy_lr)
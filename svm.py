import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
import time
import sys
from prepare_dataset import load_dataset, flatten_data_and_scale, extract_features

def svm_linear_train(X, y, C=0.2, max_iters=100):
    n_samples, n_features = X.shape
    X = np.hstack((X, np.ones((n_samples, 1))))
    n_features += 1

    P = matrix(np.eye(n_features))
    P[-1, -1] = 0
    q = matrix(np.zeros(n_features))
    G = matrix(np.vstack([-np.diag(y) @ X, -np.eye(n_features)]))
    h = matrix(np.hstack([-np.ones(n_samples), np.zeros(n_features)]))
    h[:n_samples] *= C

    solvers.options['maxiters'] = max_iters
    sol = solvers.qp(P, q, G, h)
    w = np.array(sol['x']).flatten()
    return w[:-1], w[-1]

def train_one_vs_all_linear(X, y, classes, max_iters=100):
    models = {}
    start_time = time.time()
    for c in classes:
        yc = np.where(y == c, 1, -1)
        w, b = svm_linear_train(X, yc, max_iters=max_iters)
        models[c] = (w, b)
    training_time = time.time() - start_time
    print(f"Custom SVM Training Time: {training_time:.2f} seconds")
    return models, training_time

def predict(X, models):
    class_labels = list(models.keys())
    scores = np.array([np.dot(X, models[c][0]) + models[c][1] for c in class_labels])
    predicted_indices = scores.argmax(axis=0)
    predictions = np.array([class_labels[idx] for idx in predicted_indices])
    return predictions

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Part 1(a): Linear SVM
def train_one_vs_all_linear_and_evaluate(X_train, Y_train, X_test, Y_test, classes, max_iters=100):
    models, training_time = train_one_vs_all_linear(X_train, Y_train, classes, max_iters=max_iters)
    train_preds = predict(X_train, models)
    test_preds = predict(X_test, models)
    train_accuracy = calculate_accuracy(Y_train, train_preds)
    test_accuracy = calculate_accuracy(Y_test, test_preds)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
    return train_accuracy, test_accuracy, training_time

# Part 1(b): Linear SVM with sklearn
def train_and_evaluate_with_sklearn_linear(X_train, Y_train, X_test, Y_test, C=0.2):
    svc = LinearSVC(C=C, dual=False, random_state=42, max_iter=10000)
    start_time = time.time()
    svc.fit(X_train, Y_train)
    training_time = time.time() - start_time
    Y_train_pred = svc.predict(X_train)
    Y_test_pred = svc.predict(X_test)
    train_accuracy = accuracy_score(Y_train, Y_train_pred)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)
    print(f"Sklearn SVM Training Time: {training_time:.2f} seconds")
    print(f"Sklearn SVM Training Accuracy for C={C}: {train_accuracy * 100:.2f}")
    print(f"Sklearn SVM Test Accuracy for C={C}: {test_accuracy * 100:.2f}")
    return svc, training_time



def gaussian_kernel(x, y, gamma):
    return np.exp(-gamma * np.linalg.norm(x-y)**2)

def compute_kernel_matrix(X, gamma):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i, n_samples):
            K[i, j] = gaussian_kernel(X[i], X[j], gamma)
            K[j, i] = K[i, j]  # The kernel matrix is symmetric
    return K


def fit_nonlinear_svm_class(X, Y, C=0.1, gamma=0.05, max_iters=100):
    n_samples, n_features = X.shape

    print("Computing Gaussian Kernel Matrix")
    K = compute_kernel_matrix(X, gamma)
    print("Finished Computing Kernel Matrix")

    P = matrix(np.outer(Y, Y) * K)
    q = matrix(-np.ones(n_samples))
    A = matrix(Y, (1, n_samples), 'd')
    b = matrix(0.0)
    G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
    h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * C)))

    solvers.options['maxiters'] = max_iters
    solution = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(solution['x']).flatten()
    support_indices = (alphas > 1e-5)

    return {
        'support_vectors': X[support_indices],
        'support_labels': Y[support_indices],
        'alphas': alphas[support_indices],
        'gamma': gamma,
        'kernel': lambda x, y: gaussian_kernel(x, y, gamma)
    }
def predict_nonlinear(models, x):
    scores = [np.sum(model['alphas'] * model['support_labels'] *
                     np.array([model['kernel'](sv, x) for sv in model['support_vectors']])
                    ) for model in models]
    return np.argmax(scores)
# Part1 (c)
def train_one_vs_all_nonlinear_and_evaluate(X_train, Y_train, X_test, Y_test, classes, C=0.1, max_iters=100):
    X_train = X_train[:2000]
    Y_train = Y_train[:2000]
    models = []
    s = time.time()
    for digit in classes:
        labels_transformed = np.where(Y_train == digit, 1, -1)
        model = fit_nonlinear_svm_class(X_train, labels_transformed, C=C, max_iters=max_iters)
        models.append(model)
    training_time = time.time() - s
    print(f"Training time with custom non-linear SVM: {training_time:.2f} seconds")
    predictions = []
    for i in range(len(X_test)):
        test_sample = X_test[i]
        prediction_index = predict_nonlinear(models, test_sample)
        predicted_digit = classes[prediction_index]
        predictions.append(Y_test[i] == predicted_digit)
    train_preds = []   
    for i in range(len(X_train)):
        train_sample = X_train[i]
        prediction_index = predict_nonlinear(models, train_sample)
        predicted_digit = classes[prediction_index]
        train_preds.append(Y_train[i] == predicted_digit)
    train_accuracy = np.mean(train_preds)
    test_accuracy = np.mean(predictions)
    print(f"Training Accuracy with custom non-linear SVM: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy with custom non-linear SVM: {test_accuracy * 100:.2f}%")
    return models,  training_time

# Part 1(d): Non-linear SVM with sklearn
def train_and_evaluate_with_sklearn_nonlinear(X_train, Y_train, X_test, Y_test, C=0.1, gamma=0.05):
    svc = SVC(C=C, gamma=gamma, kernel='poly', random_state=42)
    start_time = time.time()
    svc.fit(X_train, Y_train)
    training_time = time.time() - start_time
    
    Y_train_pred = svc.predict(X_train)
    train_accuracy = accuracy_score(Y_train, Y_train_pred)
    
    Y_test_pred = svc.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)
    
    print(f"Training time with scikit-learn non-linear SVM: {training_time:.2f} seconds")
    print(f"Training Accuracy with scikit-learn non-linear SVM: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy with scikit-learn non-linear SVM: {test_accuracy * 100:.2f}%")
    
    return svc, training_time, train_accuracy, test_accuracy

def main():
    train_data, test_data = load_dataset()
    (X_train, Y_train, X_test, Y_test) = flatten_data_and_scale(train_data=train_data, test_data=test_data)
    print("Data loaded and images flattened")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    classes = np.unique(Y_train)
    is_extract_features = False
    max_iters = 100
    if (len(sys.argv) > 1):
        is_extract_features = bool(sys.argv[1])
    if is_extract_features:
        print("Extracting features using PCA...")
        X_train, X_test = extract_features(X_train, X_test)
        max_iters = 50
    print("Training with one-vs-all linear SVM...")
    train_one_vs_all_linear_and_evaluate(X_train, Y_train, X_test, Y_test, classes, max_iters=max_iters)

    # print("Training with scikit-learn SVM...")
    # train_and_evaluate_with_sklearn_linear(X_train, Y_train, X_test, Y_test)

    # print("Training with custom non-linear SVM...")
    # train_one_vs_all_nonlinear_and_evaluate(X_train, Y_train, X_test, Y_test, classes, max_iters=max_iters)
    
    # print("Training with scikit-learn non-linear SVM...")
    # train_and_evaluate_with_sklearn_nonlinear(X_train, Y_train, X_test, Y_test)
if __name__ == '__main__':
    main()
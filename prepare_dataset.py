import pandas as pd
from sklearn.decomposition import PCA

def load_dataset(train_path='data/mnist_train.csv', test_path='data/mnist_test.csv', digits=[2,3,8,9]):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    filtered_train_data = train_data[train_data['label'].isin(digits)]
    filtered_test_data = test_data[test_data['label'].isin(digits)]
    return filtered_train_data, filtered_test_data

def flatten_data_and_scale(train_data, test_data):
    X_train = train_data.iloc[:, 1:].values.reshape(train_data.shape[0], -1)
    Y_train = train_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values.reshape(test_data.shape[0], -1)
    Y_test = test_data.iloc[:, 0].values
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, Y_train, X_test, Y_test

def extract_features(X_train, X_test, n_components=0.99):
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca
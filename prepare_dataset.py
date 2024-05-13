import pandas as pd

def load_dataset(train_path='data/mnist_train.csv', test_path='data/mnist_test.csv', digits=[2,3,8,9]):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    filtered_train_data = train_data[train_data['label'].isin(digits)]
    filtered_test_data = test_data[test_data['label'].isin(digits)]
    return filtered_train_data, filtered_test_data
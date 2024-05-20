import numpy as np
from prepare_dataset import load_dataset, extract_features
from sklearn.metrics import silhouette_score



def initialize_centroids(data, k):
    centroids_idx = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[centroids_idx]
    return centroids

def assign_to_clusters_cosine(data, centroids):
    similarities = np.dot(data, centroids.T) / (np.linalg.norm(data, axis=1)[:, None] * np.linalg.norm(centroids, axis=1))
    clusters = np.argmax(similarities, axis=1)
    return clusters


def assign_to_clusters(data, centroids):
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    clusters = np.argmin(distances, axis=0)
    return clusters

def update_centroids(data, clusters, k):
    centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
    return centroids

def kmeans(data, k, max_iterations=100, is_cosine=False):
    centroids = initialize_centroids(data, k)
    for i in range(max_iterations):
        if is_cosine:
            clusters = assign_to_clusters_cosine(data, centroids)
        else:
            clusters = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return clusters, centroids


def evaluate_clustering(data, clusters):
    silhouette_avg = silhouette_score(data, clusters)
    return silhouette_avg

def find_num_of_labels(y_train, labels):
    num_of_labels = {2:{}, 3:{}, 8:{}, 9:{}}
    for i in range(len(y_train)):
        if labels[i] not in num_of_labels[y_train[i]]:
            num_of_labels[y_train[i]][labels[i]] = 1
        else:
            num_of_labels[y_train[i]][labels[i]] += 1
    return num_of_labels

def find_most_common_label(num_of_labels):
    most_common_label = {2:0, 3:0, 8:0, 9:0}
    for key in num_of_labels:
        most_common_label[key] = max(num_of_labels[key], key=num_of_labels[key].get)
    return most_common_label

def find_accuracy(y_train, labels, most_common_label):
    correct = 0
    for i in range(len(y_train)):
        if labels[i] == most_common_label[y_train[i]]:
            correct += 1
    return correct/len(y_train)

def cluster_testset(test_data, centroids):
    distances = np.sqrt(((test_data - centroids[:, np.newaxis])**2).sum(axis=2))
    clusters = np.argmin(distances, axis=0)
    return clusters



def compute_error(data, clusters, centroids):
    error = 0
    for i in range(k):
        cluster_points = data[clusters == i]
        error += np.sum((cluster_points - centroids[i])**2)
    return error

def normalize_data(data):
    data = data / 255
    return data

if __name__ == "__main__":
    train_data, test_data = load_dataset()
    X_train = train_data.drop('label', axis=1).values
    X_train = normalize_data(X_train)
    y_train = train_data['label'].values
    

    X_test = test_data.drop('label', axis=1).values
    X_test = normalize_data(X_test)
    y_test = test_data['label'].values


    k = 4 
    clusters, centroids = kmeans(X_train, k)
    most_common_label = find_most_common_label(find_num_of_labels(y_train, clusters))
    accuracy = find_accuracy(y_train, clusters, most_common_label)
    print("Train Accuracy: " + str(accuracy))

    error = compute_error(X_train, clusters, centroids)
    print(f'Train SSE: {error}')


    clusters = cluster_testset(X_test, centroids)
    most_common_label = find_most_common_label(find_num_of_labels(y_test, clusters))
    accuracy = find_accuracy(y_test, clusters, most_common_label)
    print("Test Accuracy: " + str(accuracy))


    error = compute_error(X_test, clusters, centroids)
    print(f'Test SSE: {error}')



    train_data, test_data = extract_features(X_train, X_test)
    clusters, centroids = kmeans(train_data, k)
    most_common_label = find_most_common_label(find_num_of_labels(y_train, clusters))
    accuracy = find_accuracy(y_train, clusters, most_common_label)
    print("Train Accuracy with PCA: " + str(accuracy))

    error = compute_error(train_data, clusters, centroids)
    print(f'PCA Train SSE: {error}')


    clusters = cluster_testset(test_data, centroids)
    most_common_label = find_most_common_label(find_num_of_labels(y_test, clusters))
    accuracy = find_accuracy(y_test, clusters, most_common_label)
    print("Test Accuracy with PCA: " + str(accuracy))

    error = compute_error(test_data, clusters, centroids)
    print(f'PCA Test SSE: {error}')



    clusters, centroids = kmeans(X_train, k, is_cosine=True)
    most_common_label = find_most_common_label(find_num_of_labels(y_train, clusters))
    accuracy = find_accuracy(y_train, clusters, most_common_label)
    print("Train Accuracy cosine: " + str(accuracy))

    error = compute_error(X_train, clusters, centroids)
    print(f'Train SSE cosine: {error}')


    clusters = cluster_testset(X_test, centroids)
    find_num_of_labels(y_test, clusters)
    most_common_label = find_most_common_label(find_num_of_labels(y_test, clusters))
    accuracy = find_accuracy(y_test, clusters, most_common_label)
    print("Test Accuracy cosine: " + str(accuracy))

    error = compute_error(X_test, clusters, centroids)
    print(f'Test SSE cosine: {error}')


    train_data, test_data = extract_features(X_train, X_test)
    clusters, centroids = kmeans(train_data, k, is_cosine=True)
    most_common_label = find_most_common_label(find_num_of_labels(y_train, clusters))
    accuracy = find_accuracy(y_train, clusters, most_common_label)
    print("Train Accuracy with PCA cosine: " + str(accuracy))

    error = compute_error(train_data, clusters, centroids)
    print(f'PCA Train SSE cosine: {error}')

    clusters = cluster_testset(test_data, centroids)
    most_common_label = find_most_common_label(find_num_of_labels(y_test, clusters))
    accuracy = find_accuracy(y_test, clusters, most_common_label)
    print("Test Accuracy with PCA cosine: " + str(accuracy))

    error = compute_error(test_data, clusters, centroids)
    print(f'PCA Test SSE cosine: {error}')
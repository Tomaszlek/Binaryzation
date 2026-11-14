import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# łączenie danych treningowych i testowych w jedno
X = np.vstack((x_train, x_test))
y = np.hstack((y_train.flatten(), y_test.flatten()))

X_flat = X.reshape(X.shape[0], -1)

# standaryzacja cech, trzeba bo niektóre klasyfikatory są wrażliwe na skalę cech
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)

# redukcja wymiarów za pomocą PCA do 50 cech(przyspieszenie działania kodu, on jest i tak bardzo wolny)
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

print(f"Po redukcji wymiarów: {X_pca.shape}")

def knn_classify(train_X, train_y, test_X, k, metric):
    distances = cdist(test_X, train_X, metric=metric)
    neighbors = np.argsort(distances, axis=1)[:, :k]
    predictions = [
        Counter(train_y[neighbor_indices]).most_common(1)[0][0] for neighbor_indices in neighbors
    ]
    return predictions

def naive_bayes_classify(train_X, train_y, test_X):
    unique_classes = np.unique(train_y)
    class_probabilities = []

    for cls in unique_classes:
        cls_indices = np.where(train_y == cls)[0]
        X_cls = train_X[cls_indices]

        mean = X_cls.mean(axis=0)
        variance = X_cls.var(axis=0) + 1e-9  # Zapobieganie dzieleniu przez 0

        likelihoods = -0.5 * np.sum(((test_X - mean) ** 2) / variance + np.log(2 * np.pi * variance), axis=1)
        class_probabilities.append(likelihoods)

    # wybór klasy z największym prawdopodobieństwem
    class_probabilities = np.array(class_probabilities).T
    return np.argmax(class_probabilities, axis=1)


def leave_one_out(X, y, classifier, k=None, metric=None):
    correct = 0
    for i in range(len(X)):
        train_X = np.delete(X, i, axis=0)
        train_y = np.delete(y, i)
        test_X = X[i:i + 1]
        test_y = y[i]

        if classifier == knn_classify:
            pred = classifier(train_X, train_y, test_X, k, metric)[0]
        elif classifier == naive_bayes_classify:
            pred = classifier(train_X, train_y, test_X)[0]

        if pred == test_y:
            correct += 1
    return correct / len(X)


metrics = ["euclidean", "cityblock", "chebyshev", "braycurtis", "canberra", "minkowski"]
k_values = [1, 3, 5]

# Ze względu na czas, używam małego podzbioru danych
X_subset, _, y_subset, _ = train_test_split(X_pca, y, test_size=0.9, stratify=y, random_state=42)

results = []

# testy dla k-NN
for metric in metrics:
    for k in k_values:
        accuracy = leave_one_out(X_subset, y_subset, knn_classify, k=k, metric=metric)
        results.append(("k-NN", metric, k, accuracy))

# dla bayesa
accuracy = leave_one_out(X_subset, y_subset, naive_bayes_classify)
results.append(("Naive Bayes", "N/A", "N/A", accuracy))


results_df = pd.DataFrame(results, columns=["Classifier", "Metric", "k",  "Accuracy"])
print(results_df.sort_values(by="Accuracy", ascending=False))
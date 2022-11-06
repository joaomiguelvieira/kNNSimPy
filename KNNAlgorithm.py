import numpy as np


def classify(dataset, k):
    for i in range(dataset.n_test):
        # Calculate distances
        distances = np.sum((dataset.train - dataset.test[i]) ** 2, axis=1)

        # Get the indexes of the k smallest distances
        knn = np.argpartition(distances, -k)[:k]

        # Determine class
        classes = [dataset.train_classes[i] for i in knn]
        dataset.test_classes[i] = max(set(classes), key=classes.count)


def check_solution(dataset, filename):
    errors = 0

    with open(filename, 'r') as file:
        for i in range(dataset.n_test):
            ref = int(file.readline())
            if dataset.test_classes[i] != ref:
                errors += 1
                print(f"[ERROR] {i}: {dataset.test_classes[i]} != {ref}!")

    return errors

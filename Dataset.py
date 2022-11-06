import numpy as np


class Dataset:
    def __init__(self, filename, n_train, n_test, n_features, n_classes, data_type):
        # Save dataset parameters
        self.n_train = n_train
        self.n_test = n_test
        self.n_features = n_features
        self.n_classes = n_classes
        self.data_type = data_type

        # Load dataset from binary file
        if filename is not None:
            # Read raw data from file
            with open(filename, 'rb') as file:
                raw_train = file.read(n_train * n_features * 4)
                raw_test = file.read(n_test * n_features * 4)
                raw_train_classes = file.read(n_train * 4)

            # Transform raw data into numpy arrays
            if data_type == 'numpy':
                self.train = np.frombuffer(raw_train, dtype=np.float32).reshape((n_train, n_features))
                self.test = np.frombuffer(raw_test, dtype=np.float32).reshape((n_test, n_features))
                self.train_classes = np.frombuffer(raw_train_classes, dtype=np.int32)
            else:
                # TODO Transform raw data into tensors
                raise NotImplementedError
        else:
            # TODO Generate random dataset
            raise NotImplementedError

        # Allocate vector for test classes
        self.test_classes = np.zeros(n_test, dtype=np.int32)

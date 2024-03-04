import numpy as np

def parse_ct_data(path_to_ct_data):
    data = np.load(path_to_ct_data)
    X_train = data['X_train']; X_val = data['X_val']; X_test = data['X_test']
    y_train = data['y_train']; y_val = data['y_val']; y_test = data['y_test']

    idx_train = np.min(X_train, axis=0) != np.max(X_train, axis=0)

    # Remove columns with constant values
    X_train = X_train[: , idx_train]
    X_val = X_val[: , idx_train]
    X_test = X_test[: , idx_train]

    _, idx_train = np.unique(X_train, axis=1, return_index=True)

    # Remove duplicate columns
    idx_train.sort()

    X_train = X_train[:, idx_train]
    X_val = X_val[:, idx_train]
    X_test = X_test[:, idx_train]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def parse_ct_data_scaled(path_to_ct_data):
    X_train, X_val, X_test, y_train, y_val, y_test = parse_ct_data(path_to_ct_data)
    
    X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
    X_val = X_val / np.linalg.norm(X_val, axis=1, keepdims=True)
    X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

import numpy as np

def parse_ct_data(path_to_ct_data):
    data = np.loadtxt(path_to_ct_data, delimiter=",", skiprows=1)  # Skip first row as it's name of columns
    
    # Use unique patient IDs to split the data into training, validation, and test sets
    unique_patient_ids = np.unique(data[:, 0])  # Assuming the first column contains patient IDs
    groups = [np.where(data[:, 0] == patient_id)[0] for patient_id in unique_patient_ids]
    
    train_data = data[np.concatenate(groups[:73])]  # First 73 patients
    val_data = data[np.concatenate(groups[73:85])]  # Next 12 patients
    test_data = data[np.concatenate(groups[85:])]   # Last 12 patients
    
    # We now erase patientID column and separate X and y
    X_train, y_train = train_data[:, 1:-1], train_data[:, -1]
    X_val,   y_val   = val_data[:, 1:-1],   val_data[:, -1]
    X_test,  y_test  = test_data[:, 1:-1],  test_data[:, -1]
    
    # Shift and scale y to make training locations have zero mean and unit variance
    y_train_mean = np.mean(y_train)
    y_train_std = np.std(y_train)
    
    y_train = (y_train - y_train_mean) / y_train_std
    y_val = (y_val - y_train_mean) / y_train_std
    y_test = (y_test - y_train_mean) / y_train_std
    
    # Remove columns with constant values
    idx_train = np.min(X_train, axis=0) != np.max(X_train, axis=0)

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

def parse_ct_data_OLD(path_to_ct_data):
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

def parse_ct_data_scaled_OLD(path_to_ct_data):
    X_train, X_val, X_test, y_train, y_val, y_test = parse_ct_data_OLD(path_to_ct_data)
    
    X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
    X_val = X_val / np.linalg.norm(X_val, axis=1, keepdims=True)
    X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    import time
    t1 = time.time()
    X_train2, X_val2, X_test2, y_train2, y_val2, y_test2 = parse_ct_data_scaled_OLD('ct_data.npz')
    t2 = time.time()
    print(t2-t1)

    t1 = time.time()
    X_train, X_val, X_test, y_train, y_val, y_test = parse_ct_data_scaled('slice_localization_data.csv')
    t2 = time.time()
    print(t2-t1)

    #Compare absolute values
    print(np.allclose(X_train, X_train2))
    print(np.allclose(X_val, X_val2))
    print(np.allclose(X_test, X_test2))
    print(np.allclose(y_train, y_train2))
    print(np.allclose(y_val, y_val2))
    print(np.allclose(y_test, y_test2))

    print(np.mean(np.abs(X_train - X_train2)))
    print(np.mean(np.abs(X_val - X_val2)))
    print(np.mean(np.abs(X_test - X_test2)))
    print(np.mean(np.abs(y_train - y_train2)))
    print(np.mean(np.abs(y_val - y_val2)))
    print(np.mean(np.abs(y_test - y_test2)))
    
if __name__ == "__main__":
    main()